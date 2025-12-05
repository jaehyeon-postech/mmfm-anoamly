
"""
MIIC Dataset을 활용한 In-Context Learning Anomaly Detection

In-Context Learning (ICL) 방식:
1. 정상(normal) 샘플 몇 개와 비정상(defect) 샘플 몇 개를 예시로 제공
2. 모델이 예시를 통해 패턴을 학습하여 새로운 테스트 이미지의 defect 여부를 판단
"""

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

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
import pandas as pd
import random
from typing import List, Dict

warnings.filterwarnings("ignore")


def _load_data_from_csv(csv_path: str):
    """
    CSV 스키마:
      - new_filename: 이미지 파일명
      - label: 0(정상, 정답=No) / 1(비정상, 정답=Yes)
    """
    df = pd.read_csv(csv_path)
    data = []
    for _, row in df.iterrows():
        fn = str(row["new_filename"])
        lb = int(row["label"])
        data.append({"image": fn, "label": lb})
    return data


def prepare_icl_examples_from_support(support_csv: str, num_normal: int = 2, num_defect: int = 2, seed: int = 42):
    """
    별도의 support/train set에서 In-Context Learning 예시 샘플 준비
    
    Args:
        support_csv: support set CSV 경로 (train set)
        num_normal: 정상 예시 개수
        num_defect: 비정상 예시 개수
        seed: 랜덤 시드
    
    Returns:
        normal_examples: 정상 예시 리스트
        defect_examples: 비정상 예시 리스트
    """
    random.seed(seed)
    
    # Support set 로드
    support_data = _load_data_from_csv(support_csv)
    
    # 정상/비정상 샘플 분리
    normal_samples = [d for d in support_data if d['label'] == 0]
    defect_samples = [d for d in support_data if d['label'] == 1]
    
    # 랜덤 샘플링
    normal_examples = random.sample(normal_samples, min(num_normal, len(normal_samples)))
    defect_examples = random.sample(defect_samples, min(num_defect, len(defect_samples)))
    
    return normal_examples, defect_examples


def prepare_icl_examples_from_paths(normal_paths: List[str], defect_paths: List[str]):
    """
    직접 지정된 이미지 경로로부터 ICL 예시 준비
    
    Args:
        normal_paths: 정상 이미지 경로 리스트
        defect_paths: 비정상 이미지 경로 리스트
    
    Returns:
        normal_examples: 정상 예시 리스트
        defect_examples: 비정상 예시 리스트
    """
    normal_examples = [{"image": path, "label": 0} for path in normal_paths]
    defect_examples = [{"image": path, "label": 1} for path in defect_paths]
    
    return normal_examples, defect_examples


def prepare_icl_examples_from_directory(icl_dir: str, num_normal: int = 2, num_defect: int = 2, seed: int = 42):
    """
    디렉토리 구조에서 ICL 예시 준비 (normal/, abnormal/ 폴더)
    
    Args:
        icl_dir: ICL 예시 디렉토리 경로 (normal/, abnormal/ 폴더 포함)
        num_normal: 정상 예시 개수
        num_defect: 비정상 예시 개수
        seed: 랜덤 시드
    
    Returns:
        normal_examples: 정상 예시 리스트
        defect_examples: 비정상 예시 리스트
    """
    import glob
    
    random.seed(seed)
    
    # normal/ 폴더에서 이미지 찾기
    normal_dir = os.path.join(icl_dir, "normal")
    normal_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        normal_images.extend(glob.glob(os.path.join(normal_dir, ext)))
    
    # abnormal/ 폴더에서 이미지 찾기
    abnormal_dir = os.path.join(icl_dir, "abnormal")
    abnormal_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        abnormal_images.extend(glob.glob(os.path.join(abnormal_dir, ext)))
    
    # 랜덤 샘플링
    selected_normal = random.sample(normal_images, min(num_normal, len(normal_images)))
    selected_abnormal = random.sample(abnormal_images, min(num_defect, len(abnormal_images)))
    
    normal_examples = [{"image": path, "label": 0} for path in selected_normal]
    defect_examples = [{"image": path, "label": 1} for path in selected_abnormal]
    
    return normal_examples, defect_examples


def build_icl_system_prompt(normal_examples: List[Dict], defect_examples: List[Dict], 
                           image_token: str = DEFAULT_IMAGE_TOKEN):
    """
    ICL을 위한 System Prompt 구성 (Qwen 스타일)
    System prompt에 task 설명 + 예시 image-text pairs 포함
    """
    if len(normal_examples) == 0 and len(defect_examples) == 0:
        # Zero-shot: task 설명만
        return """You are an expert in analyzing scanning electron microscope (SEM) images of semiconductors to detect defects. 
Your task is to determine whether there are any defects or abnormalities in the given SEM image.
Reply with 'Yes' if there are defects, or 'No' if the sample is normal."""
    
    # Few-shot: task 설명 + 예시 (image-text pairs)
    system_parts = [
        "You are an expert in analyzing scanning electron microscope (SEM) images of semiconductors to detect defects.",
        "Your task is to determine whether there are any defects or abnormalities in the given SEM image.",
        "",
        "Here are some examples:",
        ""
    ]
    
    example_num = 1
    
    # 정상 예시 - 이미지와 텍스트 쌍
    for i, ex in enumerate(normal_examples):
        system_parts.append(f"Example {example_num}:")
        system_parts.append(image_token)  # 예시 이미지 토큰
        system_parts.append("Answer: No")
        system_parts.append("")
        example_num += 1
    
    # 비정상 예시 - 이미지와 텍스트 쌍
    for i, ex in enumerate(defect_examples):
        system_parts.append(f"Example {example_num}:")
        system_parts.append(image_token)  # 예시 이미지 토큰
        system_parts.append("Answer: Yes")
        system_parts.append("")
        example_num += 1
    
    system_parts.append("Based on these examples, analyze the following SEM image and reply with 'Yes' or 'No'.")
    
    return "\n".join(system_parts)


def build_icl_user_prompt(image_token: str = DEFAULT_IMAGE_TOKEN):
    """
    ICL을 위한 User Prompt 구성
    테스트 이미지 + 질문만 (예시는 system에 포함됨)
    """
    return f"{image_token}\nAre there any defects in this SEM image? Answer (Yes/No):"


def build_icl_prompt(normal_examples: List[Dict], defect_examples: List[Dict], 
                     image_token: str = DEFAULT_IMAGE_TOKEN):
    """
    Backward compatibility를 위한 wrapper
    실제로는 system prompt + user prompt 방식을 사용할 것
    """
    # Zero-shot
    if len(normal_examples) == 0 and len(defect_examples) == 0:
        return f"{image_token}\nAre there any defects in this SEM image? Please reply with 'Yes' or 'No'."
    
    # Few-shot: 예시 이미지들 + 테스트 질문
    prompt_parts = []
    
    # 예시 이미지들 (이미지만, 설명은 system prompt에)
    for ex in normal_examples + defect_examples:
        prompt_parts.append(image_token)
    
    # 테스트 이미지와 질문
    prompt_parts.append(image_token)
    prompt_parts.append("\nAre there any defects in this SEM image? Answer (Yes/No):")
    
    return "\n".join(prompt_parts)


def eval_model_with_icl(args):
    pretrained = args.model_checkpoint
    model_name = "llava_qwen"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_map = "auto" if device == "cuda" else None
    if device == "cuda" and torch.cuda.is_bf16_supported():
        torch_dtype = "bfloat16"
    elif device == "cuda":
        torch_dtype = "float16"
    else:
        torch_dtype = "float32"

    overwrite_config = {'vocab_size': 152064}

    print(f"Starting In-Context Learning evaluation with checkpoint: {pretrained}")
    print(f"ICL Settings: {args.num_normal} normal examples, {args.num_defect} defect examples")

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, None, model_name,
        device_map=device_map, cache_dir='./cache',
        torch_dtype=torch_dtype,
        overwrite_config=overwrite_config
    )

    # Anomaly encoder 설정
    if args.size != '7b':
        model.lm_head.weight = model.model.embed_tokens.weight
        print("Testing 0.5B model, set lm_head weight to embed_tokens weight")
        anomaly_encoder_weight_path = './pretrained_expert_05b.pth'
    else:
        print("Testing 7B model, no need to set lm_head weight")
        anomaly_encoder_weight_path = './pretrained_expert_7b.pth'

    anomaly_encoder = AnomalyOV()
    anomaly_encoder.load_zero_shot_weights(path=anomaly_encoder_weight_path)
    anomaly_encoder.freeze_layers()
    anomaly_encoder.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=model.device)
    anomaly_encoder.requires_grad_(False)
    anomaly_encoder.eval()

    model.set_anomaly_encoder(anomaly_encoder)
    model.eval()

    # 데이터 로드 (test set)
    if args.csv_path:
        data = _load_data_from_csv(args.csv_path)
        base_dir = args.image_root
    else:
        with open(os.path.join(args.data_dir, args.bench_json), 'r') as f:
            data = json.load(f)
        base_dir = args.data_dir
    
    # 일부 비율만 평가 (stratified sampling to maintain label balance)
    if args.eval_ratio < 1.0:
        random.seed(args.seed)
        normal_data = [d for d in data if d['label'] == 0]
        abnormal_data = [d for d in data if d['label'] == 1]
        
        num_normal = max(1, int(len(normal_data) * args.eval_ratio))
        num_abnormal = max(1, int(len(abnormal_data) * args.eval_ratio))
        
        sampled_normal = random.sample(normal_data, num_normal)
        sampled_abnormal = random.sample(abnormal_data, num_abnormal)
        
        data = sampled_normal + sampled_abnormal
        random.shuffle(data)
        
        print(f"\n[Sampling] Using {args.eval_ratio*100:.1f}% of test data:")
        print(f"  - Normal: {num_normal}/{len(normal_data)}")
        print(f"  - Abnormal: {num_abnormal}/{len(abnormal_data)}")
        print(f"  - Total: {len(data)}/{len(normal_data) + len(abnormal_data)}\n")

    # ICL 예시 준비 (support set 또는 직접 지정된 경로에서)
    normal_examples = []
    defect_examples = []
    support_base_dir = None
    
    if args.icl_example_dir:
        # Option 1: 디렉토리 구조에서 예시 로드 (normal/, abnormal/ 폴더)
        print(f"Loading ICL examples from directory: {args.icl_example_dir}")
        normal_examples, defect_examples = prepare_icl_examples_from_directory(
            args.icl_example_dir,
            num_normal=args.num_normal,
            num_defect=args.num_defect,
            seed=args.seed
        )
        support_base_dir = None  # 이미 절대 경로
    elif args.support_csv:
        # Option 2: 별도의 support/train set에서 예시 샘플링
        print(f"Loading ICL examples from support set: {args.support_csv}")
        normal_examples, defect_examples = prepare_icl_examples_from_support(
            args.support_csv,
            num_normal=args.num_normal, 
            num_defect=args.num_defect,
            seed=args.seed
        )
        support_base_dir = args.support_image_root if args.support_image_root else base_dir
    elif args.normal_examples and args.defect_examples:
        # Option 3: 직접 지정된 이미지 경로 사용
        print(f"Using manually specified ICL examples")
        normal_examples, defect_examples = prepare_icl_examples_from_paths(
            args.normal_examples,
            args.defect_examples
        )
        support_base_dir = args.support_image_root if args.support_image_root else base_dir
    elif args.num_normal > 0 or args.num_defect > 0:
        # Option 4: test set에서 샘플링 (data leakage 주의!)
        print(f"WARNING: Sampling ICL examples from test set (potential data leakage!)")
        print(f"Consider using --icl_example_dir, --support_csv or --normal_examples/--defect_examples instead")
        from random import sample as random_sample
        random.seed(args.seed)
        normal_samples = [d for d in data if d['label'] == 0]
        defect_samples = [d for d in data if d['label'] == 1]
        normal_examples = random_sample(normal_samples, min(args.num_normal, len(normal_samples)))
        defect_examples = random_sample(defect_samples, min(args.num_defect, len(defect_samples)))
        support_base_dir = base_dir
    else:
        # Option 5: Zero-shot (no ICL examples)
        print(f"\n[Zero-shot Mode] No ICL examples provided. Running zero-shot evaluation.")
        normal_examples = []
        defect_examples = []
    
    # ICL 예시 이미지 파일명 추출 (test set에서 제외하기 위해)
    icl_image_basenames = set()
    for ex in normal_examples + defect_examples:
        img_path = ex['image']
        basename = os.path.basename(img_path)
        icl_image_basenames.add(basename)
    
    # Test set에서 ICL 예시로 사용된 이미지 제외
    original_data_len = len(data)
    data = [d for d in data if os.path.basename(d['image']) not in icl_image_basenames]
    excluded_count = original_data_len - len(data)
    
    if excluded_count > 0:
        print(f"\n[Info] Excluded {excluded_count} ICL example images from test set")
        print(f"  Test set size: {original_data_len} -> {len(data)}")
    
    if len(normal_examples) > 0 or len(defect_examples) > 0:
        print(f"\nSelected {len(normal_examples)} normal examples:")
        for ex in normal_examples:
            print(f"  - {ex['image']}")
        print(f"\nSelected {len(defect_examples)} defect examples:")
        for ex in defect_examples:
            print(f"  - {ex['image']}")
    
    # 예시 이미지 로드 및 전처리
    example_images = []
    example_image_sizes = []
    for ex in normal_examples + defect_examples:
        # support_base_dir가 None이면 이미 절대 경로 (icl_example_dir 사용)
        if support_base_dir is None:
            ex_path = ex['image']
        else:
            ex_path = os.path.join(support_base_dir, ex['image'])
        ex_img = Image.open(ex_path).convert("RGB")
        
        # 이미지 리사이즈
        if max(ex_img.size) > 1024:
            if ex_img.width > ex_img.height:
                new_width = 1024
                new_height = int(1024 * ex_img.height / ex_img.width)
            else:
                new_height = 1024
                new_width = int(1024 * ex_img.width / ex_img.height)
            ex_img = ex_img.resize((new_width, new_height))
        
        example_images.append(ex_img)
        example_image_sizes.append(ex_img.size)
    
    # 예시 이미지를 한 번만 텐서로 변환 (캐싱)
    example_image_tensors = []
    if len(example_images) > 0:
        print("\n[Performance] Pre-processing example images (caching)...")
        example_image_tensors = process_images(example_images, image_processor, model.config)
        example_image_tensors = [_image.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device) 
                                 for _image in example_image_tensors]
        print(f"[Performance] Cached {len(example_image_tensors)} example image tensors")
    else:
        print("\n[Zero-shot] No example images to cache")

    responses = []
    results_with_labels = []  # Store both predictions and labels
    conv_template = "qwen_1_5"

    # ICL System Prompt 생성 (task 설명 + 예시 설명)
    icl_system_prompt = build_icl_system_prompt(normal_examples, defect_examples)
    
    # User Prompt 생성 (이미지 토큰들 + 질문)
    icl_user_prompt = build_icl_user_prompt()
    
    print("\n" + "="*80)
    print("ICL System Prompt:")
    print("="*80)
    print(icl_system_prompt)
    print("="*80)
    print("\nICL User Prompt Template:")
    print("="*80)
    print(icl_user_prompt)
    print("="*80 + "\n")

    # 각 테스트 이미지에 대해 평가
    for index, d in enumerate(data):
        # ICL 예시에 포함된 이미지는 테스트에서 제외 (optional)
        if args.exclude_examples:
            if d in normal_examples or d in defect_examples:
                print(f"Skipping {d['image']} (used as ICL example)")
                continue
        
        image_path = os.path.join(base_dir, d['image'])
        test_image = Image.open(image_path).convert("RGB")

        # 테스트 이미지 리사이즈
        if max(test_image.size) > 1024:
            if test_image.width > test_image.height:
                new_width = 1024
                new_height = int(1024 * test_image.height / test_image.width)
            else:
                new_height = 1024
                new_width = int(1024 * test_image.width / test_image.height)
            test_image = test_image.resize((new_width, new_height))

        # 테스트 이미지만 텐서 처리 (예시 이미지는 이미 캐시됨)
        test_image_tensor = process_images([test_image], image_processor, model.config)
        test_image_tensor = [_image.to(dtype=torch.bfloat16 if device == "cuda" else torch.float32, device=device) 
                            for _image in test_image_tensor]
        
        # 캐시된 예시 이미지 텐서와 결합
        all_image_tensors = example_image_tensors + test_image_tensor
        all_image_sizes = example_image_sizes + [test_image.size]

        # Conversation 구성 - System prompt 올바르게 설정
        conv = copy.deepcopy(conv_templates[conv_template])
        
        # Qwen의 system prompt를 ICL system prompt로 교체
        # conv_qwen의 system은 이미 "<|im_start|>system\nYou are a helpful assistant."로 시작
        # 우리는 기본 내용만 교체
        conv.system = f"<|im_start|>system\n{icl_system_prompt}"
        
        # User message 추가
        conv.append_message(conv.roles[0], icl_user_prompt)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Debug: 첫 번째 샘플에서 실제 프롬프트 출력
        if index == 0:
            print("\n" + "="*80)
            print("First Sample - Full Prompt:")
            print("="*80)
            print(prompt_question)
            print("="*80 + "\n")

        # 토크나이즈
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        # 생성
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=all_image_tensors,
                image_sizes=all_image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
            )
        
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(f"[{index}] {d['image']} (label={d['label']}): {text_outputs[0]}")
        responses.append(text_outputs[0])
        results_with_labels.append({
            "image": d['image'],
            "label": d['label'],
            "prediction": text_outputs[0]
        })

    # 결과 저장
    results_path = args.save_path if args.save_path else f'./detection_results_icl_{args.size}.json'
    with open(results_path, 'w') as f:
        json.dump(responses, f, indent=4)
    print(f"\n[Info] Saved responses to {results_path}")
    
    # 상세 결과도 저장 (label 포함)
    detailed_results_path = results_path.replace('.json', '_detailed.json')
    with open(detailed_results_path, 'w') as f:
        json.dump(results_with_labels, f, indent=4)
    print(f"[Info] Saved detailed results with labels to {detailed_results_path}")

    # 메트릭 계산 및 저장
    metrics_results = {
        "config": {
            "model_checkpoint": args.model_checkpoint,
            "model_size": args.size,
            "num_normal_examples": len(normal_examples),
            "num_defect_examples": len(defect_examples),
            "icl_mode": "zero-shot" if len(normal_examples) == 0 and len(defect_examples) == 0 else "few-shot",
            "eval_ratio": args.eval_ratio,
            "total_samples": len(data),
            "seed": args.seed
        },
        "metrics": {}
    }
    
    try:
        from calc_accuracy import cal_metrics_miic
        print("\n" + "="*80)
        print("Evaluation Metrics:")
        print("="*80)
        acc, precision, recall, f1 = cal_metrics_miic(data, responses)
        metrics_results["metrics"] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    except ImportError:
        print("\n[Warning] calc_accuracy module not found. Calculating basic metrics...")
        # 간단한 정확도 계산
        correct = 0
        tp, tn, fp, fn = 0, 0, 0, 0
        total = len(responses)
        for i, (d, resp) in enumerate(zip(data, responses)):
            pred = 1 if 'yes' in resp.lower() else 0
            label = d['label']
            if pred == label:
                correct += 1
                if pred == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    fn += 1
        
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")
        
        metrics_results["metrics"] = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn
        }
    
    # 메트릭 결과 저장
    metrics_path = results_path.replace('.json', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_results, f, indent=4)
    print(f"[Info] Saved evaluation metrics to {metrics_path}")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test detection with In-Context Learning on MIIC dataset")
    
    # 데이터 경로
    parser.add_argument("--data_dir", type=str, default='/home/jhjeong/sait_data', 
                       help="Path to your data directory")
    parser.add_argument("--bench_json", type=str, default='VisA/test_data.json', 
                       help="Path to your benchmark json file")
    parser.add_argument("--csv_path", type=str, default="/home/jhjeong/miic/test_dataset/label.csv", 
                       help="CSV file with columns: new_filename,label")
    parser.add_argument("--image_root", type=str, default="/home/jhjeong/miic/test_dataset/image", 
                       help="Root directory for images (used with csv_path)")
    
    # 모델 설정
    parser.add_argument("--model_checkpoint", type=str, required=True,
                       help="Path to your pretrained model")
    parser.add_argument("--size", type=str, default='7b', 
                       choices=['7b', '0.5b'],
                       help="Model size")
    
    # ICL 예시 소스 설정 (4가지 옵션)
    parser.add_argument("--icl_example_dir", type=str, default=None,
                       help="[Option 1] Directory containing normal/ and abnormal/ folders with ICL examples (recommended)")
    parser.add_argument("--support_csv", type=str, default=None,
                       help="[Option 2] CSV file for support/train set to sample ICL examples from")
    parser.add_argument("--support_image_root", type=str, default=None,
                       help="Root directory for support set images (used with support_csv)")
    parser.add_argument("--normal_examples", type=str, nargs='+', default=None,
                       help="[Option 3] List of normal example image paths (space-separated)")
    parser.add_argument("--defect_examples", type=str, nargs='+', default=None,
                       help="[Option 3] List of defect example image paths (space-separated)")
    
    # ICL 설정
    parser.add_argument("--num_normal", type=int, default=0,
                       help="Number of normal examples for ICL. Set to 0 for zero-shot. (default: 0)")
    parser.add_argument("--num_defect", type=int, default=0,
                       help="Number of defect examples for ICL. Set to 0 for zero-shot. (default: 0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for example selection")
    parser.add_argument("--exclude_examples", action='store_true',
                       help="Exclude ICL example images from test set (only works when sampling from test set)")
    parser.add_argument("--eval_ratio", type=float, default=1.0,
                       help="Ratio of test data to evaluate (0.0-1.0). E.g., 0.1 for 10%% of data")
    
    # 생성 설정
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    
    # 출력 설정
    parser.add_argument("--save_path", type=str, default=None, 
                       help="Path to save the results")
    
    args = parser.parse_args()

    eval_model_with_icl(args)
