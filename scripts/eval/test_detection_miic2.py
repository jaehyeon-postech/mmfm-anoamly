#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust binary defect detection evaluator for VLMs.
Supports:
  - LLaVA-OneVision (e.g., lmms-lab/llava-onevision-qwen2-7b-ov)
  - InternVL2 (e.g., OpenGVLab/InternVL2-8B)  <-- trust_remote_code 필요
  - Qwen2-VL (generic Vision2Seq 경로)
"""

import os
import json
import copy
import inspect
import argparse
from typing import List, Dict, Tuple

import torch
import pandas as pd
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoModelForCausalLM,
    AutoImageProcessor,
)
from transformers import LlavaOnevisionForConditionalGeneration


# -------------------------------
# Utilities
# -------------------------------

def internvl2_generate(model, tokenizer, image_processor, image, question, max_new_tokens):
    """
    InternVL2 포크에서 chat()이 죄다 깨질 때 쓰는 '무조건 동작' 경로.
    - pixel_values: AutoImageProcessor로 생성
    - prompt: <img><IMG_CONTEXT></img> + 질문
    - model.generate(input_ids + pixel_values) 직접 호출
    """
    # 1) pixel_values
    enc = image_processor(images=image, return_tensors="pt")
    data = enc.data if hasattr(enc, "data") else enc
    pv = None
    for k in ("pixel_values", "images", "pixel_values_list", "pixel_inputs", "vision_tower_pixel_values"):
        if k in data:
            pv = data[k][0] if isinstance(data[k], (list, tuple)) else data[k]
            break
    if pv is None:
        keys = list(data.keys()) if isinstance(data, dict) else type(data)
        raise KeyError(f"[InternVL2] pixel_values not found. keys={keys}")
    if pv.ndim == 3:
        pv = pv.unsqueeze(0)
    pv = pv.to(model.device).to(model.dtype)

    # 2) 특수 토큰(있으면 그걸, 없으면 기본값)
    IMG_START = getattr(model, "img_start_token", "<img>")
    IMG_END   = getattr(model, "img_end_token",   "</img>")
    IMG_CTX   = getattr(model, "img_context_token", "<IMG_CONTEXT>")
    ats = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    if "<img>" in ats: IMG_START = "<img>"
    if "</img>" in ats: IMG_END = "</img>"
    if "<IMG_CONTEXT>" in ats: IMG_CTX = "<IMG_CONTEXT>"

    # 3) 프롬프트 + 토크나이즈
    prompt = f"{IMG_START}{IMG_CTX}{IMG_END}\n{question}"
    txt = tokenizer([prompt], return_tensors="pt")
    txt = {k: v.to(model.device) for k, v in txt.items()}

    # 4) 종료 토큰/패딩
    eos_id = tokenizer.eos_token_id or getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos_id, list) and eos_id:
        eos_id = eos_id[0]
    pad_id = tokenizer.pad_token_id or eos_id

    # 5) generate
    if not hasattr(model, "generate"):
        raise RuntimeError("This InternVL2 fork has no model.generate()")
    with torch.no_grad():
        out = model.generate(
            **txt,
            pixel_values=pv,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            do_sample=False,
        )

    # 6) 프롬프트 길이만큼 잘라 생성 부분만 디코드
    plen = txt["input_ids"].shape[1]
    gen_only = out[0, plen:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()

def pick_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_csv(csv_path: str) -> List[Dict]:
    df = pd.read_csv(csv_path)
    out = []
    for _, row in df.iterrows():
        out.append({"image": str(row["new_filename"]), "label": int(row["label"])})
    return out


def resize_long_edge(image: Image.Image, max_long: int = 1024) -> Image.Image:
    if max(image.size) <= max_long:
        return image
    if image.width >= image.height:
        new_w = max_long
        new_h = int(max_long * image.height / image.width)
    else:
        new_h = max_long
        new_w = int(max_long * image.width / image.height)
    return image.resize((new_w, new_h))


def infer_model_family(model_id: str) -> str:
    mid = model_id.lower()
    if "llava-onevision" in mid or "onevision" in mid:
        return "llava_ov"
    if "internvl" in mid:
        return "internvl2"
    if "qwen2-vl" in mid or "qwen2_vl" in mid:
        return "qwen2_vl"
    return "generic_vlm"


def compute_metrics(golds: List[int], preds: List[int]) -> Dict[str, float]:
    assert len(golds) == len(preds)
    tp = sum(1 for g, p in zip(golds, preds) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(golds, preds) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(golds, preds) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(golds, preds) if g == 1 and p == 0)
    total = len(golds)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    acc  = (tp + tn) / total if total else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "total": total}


def parse_yes_no(text: str) -> int:
    s = (text or "").strip().lower()
    if any(x in s for x in ["no defect", "no defects", "without defect", "no anomaly"]):
        return 0
    if any(x in s for x in ["has defect", "have defect", "defect exists", "anomaly detected"]):
        return 1
    if s.startswith("yes") or " yes" in s:
        return 1
    if s.startswith("no") or " no" in s:
        return 0
    return 0


# -------------------------------
# Prompt builders
# -------------------------------

def build_prompt_llava_ov(processor, question: str) -> str:
    # 줄바꿈은 반드시 \n 로: f-string 줄깨짐 방지
    messages = [{"role": "user", "content": f"<image>\n{question}"}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_prompt_qwen2(processor, image: Image.Image, question: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True, images=[image], tokenize=False)
    except TypeError:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_prompt_generic(processor, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        return question


# -------------------------------
# InternVL2 helpers
# -------------------------------

def _get_pixel_values_and_patches(image_processor, image, device, dtype):
    """
    InternVL2: 이미지 전처리는 AutoImageProcessor로만 한다.
    반환 딕셔너리에서 pixel_values 후보 키를 robust하게 탐색.
    """
    enc = image_processor(images=image, return_tensors="pt")
    data = enc.data if hasattr(enc, "data") else enc

    CAND_KEYS = (
        "pixel_values",
        "images",
        "pixel_values_list",
        "pixel_inputs",
        "vision_tower_pixel_values",
    )
    pixel_values = None
    for k in CAND_KEYS:
        if k in data:
            pixel_values = data[k]
            if isinstance(pixel_values, (list, tuple)):
                pixel_values = pixel_values[0]
            break

    if pixel_values is None:
        keys = list(data.keys()) if isinstance(data, dict) else type(data)
        raise KeyError(f"[InternVL2] pixel_values not found. Available keys: {keys}")

    num_patches_list = data.get("num_patches_list", None)
    return pixel_values.to(device).to(dtype), num_patches_list

def internvl2_chat(model, tokenizer, image_processor, image, question, max_new_tokens):
    """
    InternVL2 chat() 시그니처 자동 대응 + generate 미탑재 포크 폴백.
    1순위: (tokenizer, pixel_values, question, generation_config, ...)
           -> 일부 포크에서 language_model.generate 미탑재로 AttributeError 발생 가능
           -> 그 경우 image/question 또는 msgs 경로로 폴백
    """
    import inspect
    sig = inspect.signature(model.chat)
    params = set(sig.parameters.keys())

    # ----- pixel_values 준비 (배치 차원/타입 보장) -----
    enc = image_processor(images=image, return_tensors="pt")
    data = enc.data if hasattr(enc, "data") else enc
    pv = None
    for k in ("pixel_values", "images", "pixel_values_list", "pixel_inputs", "vision_tower_pixel_values"):
        if k in data:
            pv = data[k]
            if isinstance(pv, (list, tuple)):
                pv = pv[0]
            break
    if pv is None:
        keys = list(data.keys()) if isinstance(data, dict) else type(data)
        raise KeyError(f"[InternVL2] pixel_values not found. keys={keys}")
    if pv.ndim == 3:
        pv = pv.unsqueeze(0)
    pv = pv.to(model.device).to(model.dtype)

    # ----- generation_config(dict) 준비 -----
    gen_cfg = {}
    try:
        gen_cfg_obj = copy.deepcopy(model.generation_config)
        try:
            gen_cfg = gen_cfg_obj.to_dict()
        except Exception:
            gen_cfg = dict(getattr(gen_cfg_obj, "__dict__", {}))
    except Exception:
        pass
    gen_cfg["max_new_tokens"] = int(max_new_tokens)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if isinstance(eos_id, list) and eos_id:
        eos_id = eos_id[0]
    if eos_id is not None:
        gen_cfg.setdefault("eos_token_id", eos_id)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_id is None and eos_id is not None:
        pad_id = eos_id
    if pad_id is not None:
        gen_cfg.setdefault("pad_token_id", pad_id)

    for k in ("temperature", "top_p", "top_k"):
        if k in gen_cfg and gen_cfg[k] is None:
            gen_cfg.pop(k, None)

    # ----- 1순위 시그니처 시도 -----
    if {"tokenizer", "pixel_values", "question", "generation_config"}.issubset(params):
        try:
            out = model.chat(
                tokenizer=tokenizer,
                pixel_values=pv,
                question=question,
                generation_config=gen_cfg,
                return_history=False,
            )
            return out[0] if isinstance(out, (tuple, list)) and len(out) >= 1 else out
        except AttributeError as e:
            # language_model.generate 미탑재 포크 → 폴백 시도
            if "generate" not in str(e):
                raise
            # 아래 폴백 경로로 진행
        except TypeError as e:
            # 일부 포크에서 매개변수 불일치 → 폴백 경로로 진행
            pass

    # ----- 폴백 1: (image, question, tokenizer) -----
    if {"image", "question", "tokenizer"}.issubset(params):
        try:
            return model.chat(image=image, question=question, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        except Exception:
            pass

    # ----- 폴백 2: (image, msgs, tokenizer) -----
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ],
    }]
    if {"image", "msgs", "tokenizer"}.issubset(params):
        try:
            return model.chat(image=image, msgs=messages, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        except Exception:
            pass

    # ----- 폴백 3: (msgs, tokenizer) -----
    if {"msgs", "tokenizer"}.issubset(params):
        try:
            return model.chat(msgs=messages, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        except Exception:
            pass

    # 모두 실패
    raise RuntimeError(
        "InternVL2 chat() failed across signatures. "
        "This fork lacks language_model.generate and no alternate chat path succeeded. "
        f"Detected signature: {sig}"
    )


# -------------------------------
# Inference
# -------------------------------

def run_inference(image: Image.Image,
                  processor,
                  model,
                  tokenizer,
                  family: str,
                  max_new_tokens: int = 64,
                  image_processor=None) -> str:
    """
    - LLaVA/Generic: processor(images=..., text=...) + generate
    - InternVL2: image_processor로 pixel_values 생성 후 model.chat(...)
    """
    question = "Are there any defects for the object in the image? Please reply with 'Yes' or 'No'."

    if family == "internvl2":
        if image_processor is None:
            raise ValueError("InternVL2 requires `image_processor` (AutoImageProcessor).")
        text = internvl2_generate(model, tokenizer, image_processor, image, 
                                "Are there any defects for the object in the image? Please reply with 'Yes' or 'No'.",
                                max_new_tokens)
        return text


    # Build family-specific prompt
    if family == "llava_ov":
        prompt = build_prompt_llava_ov(processor, question)
    elif family == "qwen2_vl":
        prompt = build_prompt_qwen2(processor, image, question)
    else:
        prompt = build_prompt_generic(processor, question)

    # Encode and generate (Vision2Seq-style)
    encoded = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in encoded.items()}

    # Some models dislike unknown keys; keep safe subset
    allowed = {"input_ids", "attention_mask", "pixel_values", "pixel_attention_mask", "position_ids"}
    inputs = {k: v for k, v in inputs.items() if k in allowed}

    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    if "input_ids" in inputs and inputs["input_ids"].ndim == 2:
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = out[0, prompt_len:]
    else:
        gen_ids = out[0]

    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    if text.lower().startswith("assistant"):
        text = text.split("\n", 1)[-1].strip()
    return text


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Binary defect detection with VLMs (robust)")
    ap.add_argument("--csv_path", required=True, help="CSV with columns: new_filename,label")
    ap.add_argument("--image_root", required=True, help="Directory containing images")
    ap.add_argument("--model_id", default="lmms-lab/llava-onevision-qwen2-7b-ov")
    ap.add_argument("--model_family", default=None, choices=[None, "llava_ov", "internvl2", "qwen2_vl", "generic_vlm"])
    ap.add_argument("--save_json", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--resize_long_edge", type=int, default=1024)
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    device, dtype = pick_device_and_dtype()
    device_map = "auto" if device == "cuda" else None

    family = args.model_family or infer_model_family(args.model_id)
    need_trust = bool(args.trust_remote_code or family == "internvl2" or "opengvlab/internvl2" in args.model_id.lower())

    proc_kwargs = {"trust_remote_code": need_trust}
    tok_kwargs  = {"use_fast": False, "trust_remote_code": need_trust}
    mdl_kwargs  = {"dtype": dtype, "device_map": device_map, "trust_remote_code": need_trust}
    if args.revision:
        proc_kwargs["revision"] = args.revision
        tok_kwargs["revision"]  = args.revision
        mdl_kwargs["revision"]  = args.revision

    processor = AutoProcessor.from_pretrained(args.model_id, **proc_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, **tok_kwargs)

    # InternVL2용 이미지 프로세서(다른 패밀리에서도 있어도 무방)
    ip_kwargs = {"trust_remote_code": True}
    if args.revision:
        ip_kwargs["revision"] = args.revision
    image_processor = AutoImageProcessor.from_pretrained(args.model_id, **ip_kwargs)

    # Load model by family
    if family == "llava_ov":
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(args.model_id, **mdl_kwargs)
    elif family == "internvl2":
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **mdl_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(args.model_id, **mdl_kwargs)
    model.eval()

    # Sanity: loaded class must match family
    cls = type(model).__name__.lower()
    if family == "llava_ov" and "llava" not in cls:
        raise RuntimeError(f"[Mismatch] family=llava_ov but loaded class is {type(model).__name__}.")
    if family == "internvl2" and "internvl" not in cls:
        raise RuntimeError(f"[Mismatch] family=internvl2 but loaded class is {type(model).__name__}.")

    data = load_csv(args.csv_path)
    golds: List[int] = []
    preds: List[int] = []
    raw_texts: List[str] = []

    for i, item in enumerate(data):
        img_path = os.path.join(args.image_root, item["image"])
        if not os.path.exists(img_path):
            print(f"[Warn] Not found: {img_path}. Skip.")
            continue
        image = Image.open(img_path).convert("RGB")
        image = resize_long_edge(image, args.resize_long_edge)

        text = run_inference(
            image=image,
            processor=processor,
            model=model,
            tokenizer=tokenizer,
            family=family,
            max_new_tokens=args.max_new_tokens,
            image_processor=image_processor,  # InternVL2에서 사용, 그 외 무시
        )

        pred = parse_yes_no(text)
        gold = int(item["label"])
        raw_texts.append(text)
        preds.append(pred)
        golds.append(gold)

        print(f"{i:05d} | gt={gold} pred={pred} | {os.path.basename(img_path)} | out: {text}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(raw_texts, f, indent=2, ensure_ascii=False)
        print(f"[Info] Saved raw responses to {args.save_json}")

    m = compute_metrics(golds, preds)
    print("\n===== Metrics =====")
    for k, v in m.items():
        print(f"{k:>10s}: {v:.4f}" if isinstance(v, float) else f"{k:>10s}: {v}")


if __name__ == "__main__":
    main()
