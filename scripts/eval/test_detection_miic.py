#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from transformers import AutoTokenizer, LlavaOnevisionForConditionalGeneration, AutoProcessor
from llava.model.anomaly_expert import AnomalyOV

from PIL import Image
import requests
import copy
import torch

import json
import os
import sys
import warnings
import argparse
import pandas as pd  # ← CSV 읽기용

from calc_accuracy import cal_metrics, cal_metrics_miic

warnings.filterwarnings("ignore")


def _load_data_from_csv(csv_path: str):
    """
    CSV 스키마:
      - new_filename: 이미지 파일명
      - label: 0(정상, 정답=No) / 1(비정상, 정답=Yes)
    cal_metrics(data, responses) 호환을 위해
    [{'image': <filename>, 'label': 0/1}, ...] 형태로 반환
    """
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        fn = str(row["new_filename"])
        lb = int(row["label"])
        data.append({"image": fn, "label": lb})
    return data


def eval_model(args):
    pretrained = args.model_checkpoint
    if "anomaly" in pretrained :
        use_anomaly = True
    else:
        use_anomaly = False


    model_name = "llava_qwen"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None
    if device == "cuda" and torch.cuda.is_bf16_supported():
        torch_dtype = "bfloat16"
    elif device == "cuda":
        torch_dtype = "float16"
    else:
        torch_dtype = "float32"

    # 원본의 overwrite_config 유지
    overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 152064}
    overwrite_config = {'vocab_size': 152064}

    print(f"Starting evaluation with checkpoint: {pretrained}")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name,
        device_map=device_map, cache_dir='./cache',
        torch_dtype=torch_dtype,
        overwrite_config=overwrite_config
    )

    if args.size != '7b':
        model.lm_head.weight = model.model.embed_tokens.weight
        print("Testing 0.5B model, set lm_head weight to embed_tokens weight")
        anomaly_encoder_weight_path = './pretrained_expert_05b.pth'
    else:
        print("Testing 7B model, no need to set lm_head weight")
        anomaly_encoder_weight_path = './pretrained_expert_7b.pth'

    if use_anomaly:
        anomaly_encoder = AnomalyOV()
        anomaly_encoder.load_zero_shot_weights(path=anomaly_encoder_weight_path)
        anomaly_encoder.freeze_layers()
        anomaly_encoder.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=model.device)
        anomaly_encoder.requires_grad_(False)
        anomaly_encoder.eval()

        model.set_anomaly_encoder(anomaly_encoder)
    else:

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype=torch_dtype, device_map=device_map
        )
        processor = AutoProcessor.from_pretrained(pretrained)
        # HF 경로에선 tokenizer를 processor에서 가져와 일관 디코딩
        tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False)
        model.eval()

        # 생성기 함수 (HF 경로)
        def run_inference(image: Image.Image) -> str:
            # 동일 리사이즈 정책 유지(선택)
            if max(image.size) > 1024:
                if image.width > image.height:
                    new_width = 1024
                    new_height = int(1024 * image.height / image.width)
                else:
                    new_height = 1024
                    new_width = int(1024 * image.width / image.height)
                image = image.resize((new_width, new_height))

            prompt = "Are there any defects for the object in the image? Please reply with 'Yes' or 'No'."
            inputs = processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                )
            text = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
            return text

    model.eval()

    responses = []

    # ======= 여기만 변경: CSV 우선 사용 =======
    if args.csv_path:
        # CSV를 읽어 원본 JSON과 동일한 스키마로 구성
        data = _load_data_from_csv(args.csv_path)
        base_dir = args.image_root  # 이미지 루트
    else:
        with open(os.path.join(args.data_dir, args.bench_json), 'r') as f:
            data = json.load(f)
        base_dir = args.data_dir
    # =========================================

    for index, d in enumerate(data):
        image_path = os.path.join(base_dir, d['image'])
        image = Image.open(image_path).convert("RGB")

        # 원본 리사이즈 로직
        if max(image.size) > 1024:
            if image.width > image.height:
                new_width = 1024
                new_height = int(1024 * image.height / image.width)
            else:
                new_height = 1024
                new_width = int(1024 * image.width / image.height)
            image = image.resize((new_width, new_height))

        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device) for _image in image_tensor]

        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + "\nAre there any defects for the object in the image? Please reply with 'Yes' or 'No'."
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size]

        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(index, text_outputs[0])
        responses.append(text_outputs[0])

    # 저장
    results_path = args.save_path if args.save_path else f'./detection_results_{args.size}.json'
    with open(results_path, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"[Info] Saved responses to {results_path}")

    # cal_metrics는 (data, responses) 시그니처 유지
    cal_metrics_miic(data, responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test detection performance of the model")
    # JSON 경로(기존 방식 유지용)
    parser.add_argument("--data_dir", type=str, default='/home/jhjeong/sait_data', help="Path to your data directory")
    parser.add_argument("--bench_json", type=str, default='VisA/test_data.json', help="Path to your benchmark json file")
    # CSV 경로(신규)
    parser.add_argument("--csv_path", type=str, default="/home/jhjeong/miic/test_dataset/label.csv", help="CSV file with columns: new_filename,label")
    parser.add_argument("--image_root", type=str, default="/home/jhjeong/miic/test_dataset/image", help="Root directory for images (used with csv_path)")
    # 모델/기타
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to your pretrained model")
    parser.add_argument("--size", type=str, default='7b', help="Model size")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
    args = parser.parse_args()

    eval_model(args)
