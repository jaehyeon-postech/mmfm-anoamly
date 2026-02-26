#!/usr/bin/env python3
"""
Stage 1: AnomalyOV Encoder Fine-tuning

Trains only the AnomalyOV module with Balanced BCE loss while keeping the
SigLIP backbone fully frozen. Uses pretrained_expert_7b.pth (or 05b) as the
starting point and fine-tunes on an industrial defect dataset.

Data format (JSON):
[
    {"image": "/abs/path/or/relative/to/image_folder/image.jpg", "label": 0},  # 0 = normal
    {"image": "/abs/path/or/relative/to/image_folder/image.jpg", "label": 1},  # 1 = anomaly
]

Split strategy:
    Train : train_normal[val_normal_count:] + syn_abnormal   (e.g. 4500 + 1000)
    Val   : train_normal[:val_normal_count] + real_abnormal[:val_real_abnormal_count]  (e.g. 500 + 24)
    Test  : test_normal + real_abnormal[val_real_abnormal_count:]   (e.g. 5000 + 92)
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, LinearLR, SequentialLR, ConstantLR
from PIL import Image
from tqdm import tqdm
from absl import logging, app, flags
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# add mmfm-anomaly root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.anomaly_expert import AnomalyOV
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower

FLAGS = flags.FLAGS


# ---------------------------------------------------------------------------
# Config helper (minimal cfg object required by SigLipVisionTower)
# ---------------------------------------------------------------------------
class _VisionTowerCfg:
    mm_tunable_parts: str = ""
    unfreeze_mm_vision_tower: bool = False


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnomalyDataset(Dataset):
    """
    Accepts either a path to a JSON file or a pre-built list of dicts.
    JSON / list format:
        [{"image": "rel_or_abs_path", "label": 0 or 1}, ...]
    """

    def __init__(self, data, image_folder: str, image_processor):
        if isinstance(data, str):
            with open(data) as f:
                self.data = json.load(f)
        else:
            self.data = data
        self.image_folder = image_folder
        self.image_processor = image_processor

        n_anomaly = sum(1 for d in self.data if d["label"] == 1)
        n_normal = len(self.data) - n_anomaly
        logging.info(f"Dataset: {len(self.data)} total  |  normal={n_normal}, anomaly={n_anomaly}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.image_folder, image_path)

        image = Image.open(image_path).convert("RGB")
        # SigLipImageProcessor -> [3, 384, 384]
        pixel_values = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        label = torch.tensor(float(item["label"]), dtype=torch.float32)
        return pixel_values, label

    def get_labels(self):
        return [d["label"] for d in self.data]


# ---------------------------------------------------------------------------
# Split builder
# ---------------------------------------------------------------------------
def build_splits(
    train_normal_path, test_normal_path, syn_abnormal_path, real_abnormal_path,
    train_normal_image_folder, test_normal_image_folder,
    syn_abnormal_image_folder, real_abnormal_image_folder,
    val_normal_count, train_real_abnormal_count, val_real_abnormal_count, seed,
):
    """
    Loads 4 data sources, shuffles where needed, and returns (train, val, test) lists.

    real_abnormal is allocated in order after shuffle:
      [0 : train_real_abnormal_count]                                  -> train
      [train_real_abnormal_count : train + val_real_abnormal_count]    -> val
      [train + val_real_abnormal_count :]                              -> test

    Train : train_normal[val_normal_count:] + syn_abnormal + real_abnormal[:train_real_abnormal_count]
    Val   : train_normal[:val_normal_count] + real_abnormal[train_real_abnormal_count : train+val]
    Test  : test_normal  + real_abnormal[train+val :]
    """
    def load_and_resolve(path, image_folder):
        with open(path) as f:
            data = json.load(f)
        if image_folder:
            data = [
                {**item, "image": os.path.join(image_folder, item["image"])
                         if not os.path.isabs(item["image"]) else item["image"]}
                for item in data
            ]
        return data

    rng = np.random.default_rng(seed)

    train_normal  = load_and_resolve(train_normal_path,  train_normal_image_folder)
    test_normal   = load_and_resolve(test_normal_path,   test_normal_image_folder)
    syn_abnormal  = load_and_resolve(syn_abnormal_path,  syn_abnormal_image_folder)
    real_abnormal = load_and_resolve(real_abnormal_path, real_abnormal_image_folder)

    # shuffle before split (reproducible)
    rng.shuffle(train_normal)
    rng.shuffle(real_abnormal)

    t = train_real_abnormal_count
    v = train_real_abnormal_count + val_real_abnormal_count

    train_data = train_normal[val_normal_count:] + syn_abnormal + real_abnormal[:t]
    val_data   = train_normal[:val_normal_count] + real_abnormal[t:v]
    test_data  = test_normal  + real_abnormal[v:]

    def _log(name, data):
        n_anom = sum(1 for d in data if d["label"] == 1)
        logging.info(f"  {name:<6}: {len(data):5d} total  |  normal={len(data)-n_anom}, anomaly={n_anom}")

    logging.info("Data splits:")
    _log("Train", train_data)
    _log("Val",   val_data)
    _log("Test",  test_data)

    return train_data, val_data, test_data


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def balanced_bce_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Balanced BCE loss (Paper Eq. 9).
    predictions: [B] (sigmoid-applied, range 0~1)
    labels:      [B] (0 or 1)
    """
    n_pos = labels.sum().clamp(min=1.0)
    n_neg = (1.0 - labels).sum().clamp(min=1.0)
    pos_weight = n_neg / n_pos  # upweight minority (anomaly) class

    per_sample_weight = torch.where(labels == 1, pos_weight, torch.ones_like(labels))
    loss = F.binary_cross_entropy(predictions, labels, weight=per_sample_weight)
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(anomaly_ov, vision_tower, dataloader, device, dtype):
    anomaly_ov.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for pixel_values, labels in tqdm(dataloader, desc="  Evaluating", dynamic_ncols=True, leave=False):
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            labels_dev = labels.to(device=device, dtype=torch.float32)
            batch_size = pixel_values.shape[0]

            ov_feats, sig_feats = vision_tower(pixel_values)
            split_sizes = [1] * batch_size

            _, _, final_prediction = anomaly_ov(ov_feats, sig_feats, split_sizes)
            predictions = final_prediction.squeeze(-1).float()

            total_loss += balanced_bce_loss(predictions, labels_dev).item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.float().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    binary_preds = (all_preds >= 0.5).astype(float)

    avg_loss = total_loss / len(dataloader)
    acc = (binary_preds == all_labels).mean()
    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_labels, binary_preds, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity : anomaly -> anomaly (hit rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Miss Rate   : anomaly -> normal  (1 - TPR)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity : normal  -> normal
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Alarm : normal  -> anomaly (1 - TNR)

    return avg_loss, acc, auroc, auprc, f1, tpr, fnr, tnr, fpr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv):
    del argv  # absl: unused non-flag arguments

    torch.manual_seed(FLAGS.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if FLAGS.bf16 else torch.float32
    logging.info(f"Device: {device}  |  dtype: {dtype}")

    if FLAGS.output_dir:
        os.makedirs(FLAGS.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. W&B init
    # ------------------------------------------------------------------
    use_wandb = FLAGS.wandb and WANDB_AVAILABLE
    if FLAGS.wandb and not WANDB_AVAILABLE:
        logging.warning("wandb not installed — skipping W&B logging.")
    if use_wandb:
        wandb.init(
            project=FLAGS.wandb_project,
            name=FLAGS.wandb_run_name or None,
            entity=FLAGS.wandb_entity or None,
            config={
                "train_normal_path": FLAGS.train_normal_path,
                "test_normal_path": FLAGS.test_normal_path,
                "syn_abnormal_path": FLAGS.syn_abnormal_path,
                "real_abnormal_path": FLAGS.real_abnormal_path,
                "val_normal_count": FLAGS.val_normal_count,
                "val_real_abnormal_count": FLAGS.val_real_abnormal_count,
                "vision_tower": FLAGS.vision_tower,
                "pretrained_expert": FLAGS.pretrained_expert,
                "num_epochs": FLAGS.num_epochs,
                "batch_size": FLAGS.batch_size,
                "lr": FLAGS.lr,
                "weight_decay": FLAGS.weight_decay,
                "grad_accum_steps": FLAGS.grad_accum_steps,
                "scheduler": FLAGS.scheduler,
                "warmup_steps": FLAGS.warmup_steps,
                "bf16": FLAGS.bf16,
                "seed": FLAGS.seed,
            },
        )
        logging.info(f"W&B run: {wandb.run.url}")

    # ------------------------------------------------------------------
    # 1. SigLIP Vision Tower (fully frozen)
    # ------------------------------------------------------------------
    logging.info(f"Loading SigLIP vision tower: {FLAGS.vision_tower}")
    vision_tower = SigLipVisionTower(FLAGS.vision_tower, _VisionTowerCfg(), delay_load=False)
    vision_tower.to(device=device, dtype=dtype)
    vision_tower.requires_grad_(False)
    vision_tower.eval()
    logging.info("SigLIP loaded and frozen.")

    # ------------------------------------------------------------------
    # 2. AnomalyOV (load pretrained checkpoint, fine-tune all params)
    # ------------------------------------------------------------------
    logging.info(f"Loading AnomalyOV from: {FLAGS.pretrained_expert}")
    anomaly_ov = AnomalyOV()
    anomaly_ov.load_zero_shot_weights(path=FLAGS.pretrained_expert)
    anomaly_ov.to(device=device, dtype=dtype)
    anomaly_ov.requires_grad_(True)
    anomaly_ov.train()

    n_params = sum(p.numel() for p in anomaly_ov.parameters() if p.requires_grad)
    logging.info(f"AnomalyOV trainable parameters: {n_params / 1e6:.2f}M")

    # ------------------------------------------------------------------
    # 3. Dataset & DataLoader
    # ------------------------------------------------------------------
    image_processor = vision_tower.image_processor

    train_data, val_data, test_data = build_splits(
        FLAGS.train_normal_path, FLAGS.test_normal_path,
        FLAGS.syn_abnormal_path, FLAGS.real_abnormal_path,
        FLAGS.train_normal_image_folder, FLAGS.test_normal_image_folder,
        FLAGS.syn_abnormal_image_folder, FLAGS.real_abnormal_image_folder,
        FLAGS.val_normal_count, FLAGS.train_real_abnormal_count, FLAGS.val_real_abnormal_count,
        FLAGS.seed,
    )

    # paths are already resolved in build_splits; pass image_folder=None
    dataset = AnomalyDataset(train_data, None, image_processor)

    if FLAGS.balanced_sampling:
        labels_list = dataset.get_labels()
        n_normal  = sum(1 for l in labels_list if l == 0)
        n_anomaly = sum(1 for l in labels_list if l == 1)
        class_weight = {0: 1.0 / n_normal, 1: 1.0 / n_anomaly}
        sample_weights = [class_weight[l] for l in labels_list]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        logging.info(f"Balanced sampling ON  |  normal_w={class_weight[0]:.6f}, anomaly_w={class_weight[1]:.6f}")
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            sampler=sampler,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        logging.info("Balanced sampling OFF  |  using Balanced BCE loss for class imbalance")
        dataloader = DataLoader(
            dataset,
            batch_size=FLAGS.batch_size,
            shuffle=True,
            num_workers=FLAGS.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    val_dataloader = DataLoader(
        AnomalyDataset(val_data, None, image_processor),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        AnomalyDataset(test_data, None, image_processor),
        batch_size=FLAGS.batch_size,
        shuffle=False,
        num_workers=FLAGS.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4. Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        anomaly_ov.parameters(),
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,
    )
    total_steps = len(dataloader) * FLAGS.num_epochs // FLAGS.grad_accum_steps
    warmup_steps = min(FLAGS.warmup_steps, total_steps)

    decay_steps = total_steps - warmup_steps
    steps_per_epoch = len(dataloader) // FLAGS.grad_accum_steps
    sched_name = FLAGS.scheduler.lower()
    if sched_name == "cosine":
        main_sched = CosineAnnealingLR(optimizer, T_max=max(decay_steps, 1), eta_min=FLAGS.lr * 0.01)
    elif sched_name == "cosine_restart":
        # Paper: restart iteration = half of single epoch
        t0 = max(steps_per_epoch // 2, 1)
        main_sched = CosineAnnealingWarmRestarts(optimizer, T_0=t0, T_mult=1, eta_min=FLAGS.lr * 0.01)
    elif sched_name == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(decay_steps, 1))
    elif sched_name == "constant":
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=max(decay_steps, 1))
    else:
        raise ValueError(f"Unknown scheduler: {FLAGS.scheduler}. Choose from: cosine, cosine_restart, linear, constant")

    if warmup_steps > 0:
        warmup_sched = LinearLR(optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, main_sched], milestones=[warmup_steps])
    else:
        scheduler = main_sched
    logging.info(f"Scheduler: {FLAGS.scheduler}  |  total_steps={total_steps}  |  warmup_steps={warmup_steps}")

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    logging.info("Starting training...")
    global_step = 0
    best_metric = -float("inf")  # val AUPRC; higher = better
    best_metrics_log = {}  # stores all metrics at best epoch

    for epoch in range(1, FLAGS.num_epochs + 1):
        anomaly_ov.train()
        epoch_loss = 0.0
        train_preds = []
        train_labels = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{FLAGS.num_epochs}", dynamic_ncols=True)
        optimizer.zero_grad()

        for step, (pixel_values, labels) in enumerate(pbar):
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            labels = labels.to(device=device, dtype=dtype)
            batch_size = pixel_values.shape[0]

            # ----------------------------------------------------------
            # SigLIP forward (no_grad: frozen backbone)
            # ov_feats:  [B, 729, 1152]
            # sig_feats: list of 4 x [B, 729, 1152]
            # ----------------------------------------------------------
            with torch.no_grad():
                ov_feats, sig_feats = vision_tower(pixel_values)

            # single-resolution images: each image has 1 "patch" (no AnyRes crop)
            split_sizes = [1] * batch_size

            # ----------------------------------------------------------
            # AnomalyOV forward
            # final_prediction: [B, 1] (sigmoid applied)
            # ----------------------------------------------------------
            _, _, final_prediction = anomaly_ov(ov_feats, sig_feats, split_sizes)
            predictions = final_prediction.squeeze(-1)  # [B]

            # ----------------------------------------------------------
            # Loss (balanced BCE or standard BCE depending on sampling mode)
            # ----------------------------------------------------------
            if FLAGS.balanced_sampling:
                loss = F.binary_cross_entropy(predictions, labels)
            else:
                loss = balanced_bce_loss(predictions, labels)
            loss_scaled = loss / FLAGS.grad_accum_steps
            loss_scaled.backward()

            if (step + 1) % FLAGS.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(anomaly_ov.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if use_wandb:
                    wandb.log({"train/loss_step": loss.item(), "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

            # ----------------------------------------------------------
            # Accumulate predictions for epoch-level metrics
            # ----------------------------------------------------------
            epoch_loss += loss.item()
            train_preds.extend(predictions.detach().float().cpu().numpy())
            train_labels.extend(labels.float().cpu().numpy())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # flush remaining accumulated gradients at end of epoch
        if len(dataloader) % FLAGS.grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(anomaly_ov.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        train_preds_np = np.array(train_preds)
        train_labels_np = np.array(train_labels)
        train_binary = (train_preds_np >= 0.5).astype(float)
        train_auroc = roc_auc_score(train_labels_np, train_preds_np)
        train_auprc = average_precision_score(train_labels_np, train_preds_np)
        train_f1 = f1_score(train_labels_np, train_binary, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(train_labels_np, train_binary, labels=[0, 1]).ravel()
        train_tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        train_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        train_tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        train_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        logging.info(
            f"Epoch {epoch}/{FLAGS.num_epochs}  |  loss={avg_loss:.4f}  |  "
            f"auroc={train_auroc:.4f}  |  auprc={train_auprc:.4f}  |  f1={train_f1:.4f}  |  "
            f"tpr={train_tpr:.4f}  |  fnr={train_fnr:.4f}  |  tnr={train_tnr:.4f}  |  fpr={train_fpr:.4f}"
        )

        epoch_log = {
            "train/loss": avg_loss, "train/auroc": train_auroc, "train/auprc": train_auprc,
            "train/f1": train_f1, "train/tpr": train_tpr, "train/fnr": train_fnr,
            "train/tnr": train_tnr, "train/fpr": train_fpr,
            "epoch": epoch,
        }

        # ----------------------------------------------------------
        # Validation Evaluation (used for best model selection)
        # ----------------------------------------------------------
        val_loss, val_acc, val_auroc, val_auprc, val_f1, val_tpr, val_fnr, val_tnr, val_fpr = evaluate(
            anomaly_ov, vision_tower, val_dataloader, device, dtype
        )
        logging.info(
            f"  [Val]   loss={val_loss:.4f}  |  acc={val_acc:.4f}  |  auroc={val_auroc:.4f}  |  auprc={val_auprc:.4f}  |  f1={val_f1:.4f}  |  "
            f"tpr={val_tpr:.4f}  |  fnr={val_fnr:.4f}  |  tnr={val_tnr:.4f}  |  fpr={val_fpr:.4f}"
        )
        anomaly_ov.train()
        epoch_log.update({
            "val/loss": val_loss, "val/acc": val_acc, "val/auroc": val_auroc, "val/auprc": val_auprc, "val/f1": val_f1,
            "val/tpr": val_tpr, "val/fnr": val_fnr, "val/tnr": val_tnr, "val/fpr": val_fpr,
        })

        if use_wandb:
            wandb.log(epoch_log, step=global_step)

        # ----------------------------------------------------------
        # Best Checkpoint (based on val AUPRC — no test set leakage)
        # ----------------------------------------------------------
        if val_auprc > best_metric:
            best_metric = val_auprc
            best_metrics_log = {k.replace("val/", "best/"): v
                                for k, v in epoch_log.items() if k.startswith("val/")}
            best_metrics_log["best/epoch"] = epoch
            if FLAGS.output_dir:
                best_path = os.path.join(FLAGS.output_dir, "anomaly_ov_ft_best.pth")
                torch.save(anomaly_ov.state_dict(), best_path)
                logging.info(f"  [Best] val_auprc={val_auprc:.4f}  ->  saved: {best_path}")

    logging.info("Training complete.")

    # ------------------------------------------------------------------
    # 6. Final Test Evaluation (once, using best checkpoint)
    # ------------------------------------------------------------------
    if FLAGS.output_dir:
        best_path = os.path.join(FLAGS.output_dir, "anomaly_ov_ft_best.pth")
        logging.info(f"Loading best checkpoint for final test evaluation: {best_path}")
        anomaly_ov.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_acc, test_auroc, test_auprc, test_f1, test_tpr, test_fnr, test_tnr, test_fpr = evaluate(
        anomaly_ov, vision_tower, test_dataloader, device, dtype
    )
    logging.info(
        f"[Final Test]  loss={test_loss:.4f}  |  acc={test_acc:.4f}  |  auroc={test_auroc:.4f}  |  auprc={test_auprc:.4f}  |  f1={test_f1:.4f}  |  "
        f"tpr={test_tpr:.4f}  |  fnr={test_fnr:.4f}  |  tnr={test_tnr:.4f}  |  fpr={test_fpr:.4f}"
    )

    if use_wandb:
        final_test_log = {
            "final_test/loss": test_loss, "final_test/acc": test_acc,
            "final_test/auroc": test_auroc, "final_test/auprc": test_auprc, "final_test/f1": test_f1,
            "final_test/tpr": test_tpr, "final_test/fnr": test_fnr,
            "final_test/tnr": test_tnr, "final_test/fpr": test_fpr,
        }
        if best_metrics_log:
            final_test_log.update(best_metrics_log)
        wandb.log(final_test_log)
        wandb.finish()


if __name__ == "__main__":
    # Data paths
    flags.DEFINE_string('train_normal_path',    "./annotations/train_normal_particle.json",  'path to train normal annotation JSON')
    flags.DEFINE_string('train_normal_image_folder', None, 'image root for train_normal (used for relative paths)')
    flags.DEFINE_string('test_normal_path',     "./annotations/test_normal_particle.json",   'path to test normal annotation JSON')
    flags.DEFINE_string('test_normal_image_folder',  None, 'image root for test_normal (used for relative paths)')
    flags.DEFINE_string('syn_abnormal_path',    "./annotations/syn_abnormal_particle.json",  'path to synthetic anomaly annotation JSON')
    flags.DEFINE_string('syn_abnormal_image_folder', None, 'image root for syn_abnormal (used for relative paths)')
    flags.DEFINE_string('real_abnormal_path',   "./annotations/real_abnormal_particle.json", 'path to real anomaly annotation JSON')
    flags.DEFINE_string('real_abnormal_image_folder', None, 'image root for real_abnormal (used for relative paths)')
    # Split sizes
    flags.DEFINE_integer('val_normal_count',           500, 'number of normal samples reserved for validation (from train_normal)')
    flags.DEFINE_integer('train_real_abnormal_count',    0, 'number of real anomaly samples added to training (from real_abnormal); 0 = train on synthetic only')
    flags.DEFINE_integer('val_real_abnormal_count',     12, 'number of real anomaly samples reserved for validation (from real_abnormal)')
    # Model
    flags.DEFINE_string('vision_tower', 'google/siglip-so400m-patch14-384', 'SigLIP model name or local path')
    flags.DEFINE_string('pretrained_expert', './pretrained_expert_7b.pth', 'pretrained AnomalyOV checkpoint (7b or 05b)')
    # Training
    flags.DEFINE_string( 'output_dir', None, 'directory to save checkpoints (skipped if not set)')
    flags.DEFINE_integer('num_epochs', 10, 'number of training epochs')
    flags.DEFINE_integer('batch_size', 64, 'batch size')
    flags.DEFINE_float(  'lr', 1e-4, 'learning rate')
    flags.DEFINE_float(  'weight_decay', 1e-4, 'AdamW weight decay')
    flags.DEFINE_integer('num_workers', 1, 'number of DataLoader workers')
    flags.DEFINE_integer('grad_accum_steps', 2, 'gradient accumulation steps')
    flags.DEFINE_string( 'scheduler', 'cosine', 'LR scheduler: cosine | cosine_restart | linear | constant')
    flags.DEFINE_integer('warmup_steps', 0, 'linear warmup optimizer steps before main scheduler kicks in')
    flags.DEFINE_bool(   'balanced_sampling', True, 'use WeightedRandomSampler for 1:1 class balance (uses standard BCE instead of balanced BCE)')
    # Misc
    flags.DEFINE_bool('bf16', True, 'use bfloat16')
    flags.DEFINE_integer('seed', 42, 'random seed')
    # W&B
    flags.DEFINE_bool('wandb', True, 'enable Weights & Biases logging')
    flags.DEFINE_string('wandb_project', 'test', 'W&B project name')
    flags.DEFINE_string('wandb_run_name', None, 'W&B run name (auto-generated if not set)')
    flags.DEFINE_string('wandb_entity', 'postech-log-mmfm', 'W&B entity (team/user, uses default if not set)')
    app.run(main)
