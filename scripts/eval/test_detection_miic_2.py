#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import warnings
import argparse
import pandas as pd
from PIL import Image

from transformers import (
    AutoConfig, AutoTokenizer, AutoImageProcessor,
    AutoModelForVision2Seq, AutoModel
)

from calc_accuracy import cal_metrics, cal_metrics_miic

warnings.filterwarnings("ignore")


def _load_data_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    image_paths, labels = [], []
    for _, row in df.iterrows():
        fn = str(row["new_filename"])
        lb = int(row["label"])
        image_paths.append({"image": fn})  # cal_metrics 호환용
        labels.append(lb)
    return image_paths, labels


def pick_image_token(tokenizer):
    # LLaVA에서 흔히 쓰는 후보들 순차 탐색
    candidates = ["<image>", "<image_1>", "<|image|>", "<img>"]
    for t in candidates:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is not None and tid != tokenizer.unk_token_id:
            return t
    return "<image>"  # 최후의 폴백


def load_hf_llava_like(repo: str, device: str, dtype: torch.dtype):
    print(f"[HF] loading: {repo}")

    cfg = AutoConfig.from_pretrained(repo, revision="main", trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision="main", use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    vision_repo = getattr(cfg, "vision_tower", None) or "openai/clip-vit-large-patch14-336"
    image_processor = AutoImageProcessor.from_pretrained(
        vision_repo, trust_remote_code=True
    )

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            repo,
            revision="main",
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception:
        model = AutoModel.from_pretrained(
            repo,
            revision="main",
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if hasattr(model, "get_input_embeddings"):
        try:
            vocab_model = model.get_input_embeddings().weight.size(0)
            if vocab_model != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass

    if hasattr(model, "generation_config"):
        if model.generation_config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config.eos_token_id is None and tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id

    model.eval()
    return tokenizer, image_processor, model, cfg


def run_generate(model, tokenizer, image_processor, image: Image.Image, prompt: str, dtype: torch.dtype, max_new_tokens: int = 16):
    # 텍스트/비전 각각 준비
    inputs_t = tokenizer(prompt, return_tensors="pt")
    inputs_v = image_processor(image, return_tensors="pt")

    # 디바이스/데이터형
    for k, v in inputs_t.items():
        inputs_t[k] = v.to(model.device)
    for k, v in inputs_v.items():
        if isinstance(v, torch.Tensor) and dtype == torch.float16:
            v = v.half()
        inputs_v[k] = v.to(model.device)

    # LLaVA는 input_ids(+attention_mask) + pixel_values 조합을 기대
    merged = {**inputs_t, **inputs_v}

    with torch.no_grad():
        out_ids = model.generate(
            **merged,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens
        )

    text = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    return text


def to_yes_no(text: str) -> str:
    s = (text or "").strip()
    last = s.splitlines()[-1].strip() if "\n" in s else s
    cand = last.lower()
    if "yes" in cand and "no" not in cand:
        return "Yes"
    if "no" in cand and "yes" not in cand:
        return "No"
    if "yes" in s.lower() and "no" not in s.lower():
        return "Yes"
    if "no" in s.lower() and "yes" not in s.lower():
        return "No"
    if "defect" in s.lower() and any(k in s.lower() for k in ["found", "exists", "present", "detected", "abnormal"]):
        return "Yes"
    if "no defect" in s.lower() or "no abnormal" in s.lower():
        return "No"
    return last if last in ["Yes", "No"] else "No"


def eval_model(args):
    repo = args.model_checkpoint or "ChenWeiLi/Anomaly_OV_0.5B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported())
             else (torch.float16 if device == "cuda" else torch.float32))

    tokenizer, image_processor, model, cfg = load_hf_llava_like(repo, device, dtype)

    # 데이터 로딩
    if args.csv_path:
        data, ground_truths = _load_data_from_csv(args.csv_path)
        base_dir = args.image_root
    else:
        with open(os.path.join(args.data_dir, args.bench_json), 'r') as f:
            data = json.load(f)
        ground_truths = [d.get('label', 0) for d in data]
        base_dir = args.data_dir

    # 이미지 토큰 포함 프롬프트
    img_tok = pick_image_token(tokenizer)
    prompt_tpl = f"{img_tok}\nAre there any defects for the object in the image? Reply with 'Yes' or 'No'."

    responses = []
    for idx, d in enumerate(data):
        image_path = os.path.join(base_dir, d["image"])
        image = Image.open(image_path).convert("RGB")

        text = run_generate(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image=image,
            prompt=prompt_tpl,  # ← 이미지 토큰 포함
            dtype=dtype,
            max_new_tokens=16
        )

        answer = to_yes_no(text)
        print(idx, answer, "| raw:", text.replace("\n", " ")[:120])
        responses.append(answer)

    results_path = args.save_path or "./detection_results_hf.json"
    with open(results_path, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"[Info] Saved responses to {results_path}")

    cal_metrics_miic(ground_truths, responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test detection performance of the model (HF-only, no anomaly encoder)")
    parser.add_argument("--data_dir", type=str, default='/home/jhjeong/sait_data', help="Path to your data directory")
    parser.add_argument("--bench_json", type=str, default='VisA/test_data.json', help="Path to your benchmark json file")
    parser.add_argument("--csv_path", type=str, default="/home/jhjeong/miic/test_dataset/label.csv", help="CSV file with columns: new_filename,label")
    parser.add_argument("--image_root", type=str, default="/home/jhjeong/miic/test_dataset/image", help="Root directory for images (used with csv_path)")
    parser.add_argument("--model_checkpoint", type=str, default="ChenWeiLi/Anomaly_OV_0.5B", help="Path or HF repo id for the pretrained model")
    parser.add_argument("--size", type=str, default='7b', help="(Unused) kept for interface compatibility")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
    args = parser.parse_args()

    eval_model(args)
