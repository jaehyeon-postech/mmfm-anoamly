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

warnings.filterwarnings("ignore")

def eval_model(args):

    pretrained = args.model_checkpoint
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    overwrite_config = {'vocab_size': 152064}
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, cache_dir='/cache', torch_dtype="bfloat16", overwrite_config=overwrite_config)
    
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
    anomaly_encoder.to(dtype=torch.bfloat16, device=model.device)
    # freeze the anomaly encoder
    anomaly_encoder.requires_grad_(False)
    anomaly_encoder.eval()

    model.set_anomaly_encoder(anomaly_encoder)
    model.eval()

    def create_conv(history_questions, history_responses, question):
        conv = copy.deepcopy(conv_templates[conv_template])
        for q, r in zip(history_questions, history_responses):
            conv.append_message(conv.roles[0], q)
            conv.append_message(conv.roles[1], r)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv

    image_path = 'image3.png'

    question_set = []
    history_questions = []
    history_responses = []
    temp_answers = []

    image = Image.open(image_path).convert("RGB")
    # if the longest side of the image is greater than 1024, resize it to 1024 while keeping the aspect ratio
    if max(image.size) > 1024:
        if image.width > image.height:
            new_width = 1024
            new_height = int(1024 * image.height / image.width)
        else:
            new_height = 1024
            new_width = int(1024 * image.width / image.height)
        image = image.resize((new_width, new_height))

    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    q1 = input("Enter your first question: ")
    question_set.append(q1)
    question = DEFAULT_IMAGE_TOKEN + "\n" + question_set[0]
    history_questions.append(question)
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
    print(text_outputs[0])
    temp_answers.append(text_outputs[0])
    history_responses.append(text_outputs[0])

    while True:
        q = input("Enter your question (enter 'exit' to quit): ")
        if q == 'exit':
            break
        question_set.append(q)
        question = question_set[-1]
        conv = create_conv(history_questions, history_responses, question)
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
        print(text_outputs[0])
        temp_answers.append(text_outputs[0])
        history_responses.append(text_outputs[0])
        history_questions.append(question)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with the model on single image")
    parser.add_argument("-i", "--image", type=str, default='image.png', help="Image path")
    parser.add_argument("--model_checkpoint", type=str, default=None, help="Path to your pretrained model")
    parser.add_argument("--size", type=str, default='7b', help="Model size")
    args = parser.parse_args()

    eval_model(args)