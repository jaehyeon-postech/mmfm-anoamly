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
base_dir = '/data/02/jiacong/data/'
data_json = 'VisA/test_data.json'

with open(os.path.join(base_dir, data_json), 'r') as f:
    data = json.load(f)

import sys
import warnings

warnings.filterwarnings("ignore")

#pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov" #"checkpoints/llava_ov_05B_only_llm_only_anomaly_data/checkpoint-100"
pretrained = 'checkpoints/anomaly_ov3_05B_only_llm_and_projector_only_anomaly_data'
# pretrained = "checkpoints/llava_ov_7B_only_llm_only_anomaly_data/checkpoint-20"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
#overwrite_config = {'tie_word_embeddings': False, 'use_cache': True, 'vocab_size': 152064}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, cache_dir='./cache', torch_dtype="bfloat16") #, overwrite_config=overwrite_config)
model.lm_head.weight = model.model.embed_tokens.weight
# print(torch.all(model.lm_head.weight == model.model.embed_tokens.weight))
print(model.lm_head.weight[0, :10])


anomaly_encoder = AnomalyOV()
anomaly_encoder.load_zero_shot_weights()
# raise NotImplementedError("AnomalyOV is not implemented yet.")
anomaly_encoder.freeze_layers()
anomaly_encoder.to(dtype=torch.bfloat16, device=model.device)
# freeze the anomaly encoder
anomaly_encoder.requires_grad_(False)
anomaly_encoder.eval()

model.set_anomaly_encoder(anomaly_encoder)
model.eval()

responses = []

for d in data:
    image_path = os.path.join(base_dir, d['image'])
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
    print(text_outputs[0])
    responses.append(text_outputs[0])
    #break

with open('anomalyov3_05b_finetune_llm_projector_total_data_yes_no_responses.json', 'w') as f:
    json.dump(responses, f, indent=4)





