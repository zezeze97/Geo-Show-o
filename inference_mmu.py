# coding=utf-8
# Copyright 2024 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import wandb
from models import MAGVITv2, VQModel, GeoUniForCausalLM
from training.prompting_utils import UniversalPrompting
from training.utils import get_config, flatten_omega_conf
from training.geo_data_aug import crop
from training.custom_data import image_transform
from transformers import AutoTokenizer
import json

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    elif model_type == "geo":
        return VQModel
    else:
        raise ValueError(f"model_type {model_type} not supported.")

def load_geo_vqgan(vq_model, config, ckpt_path=None, use_ema=True):
    model = vq_model(**config)
    
    if ckpt_path is not None:
        # 加载检查点文件中的 state_dict
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
         # 提取出普通模型权重和 EMA 权重
        if use_ema:
            key_map = {k.replace('.', ''): k for k in sd.keys() if not k.startswith('model_ema.') and 'loss' not in k} 
            weights = {key_map[k.replace('model_ema.', '')]: v for k, v in sd.items() if k.startswith('model_ema.') and 'loss' not in k and 'model_ema.decay' not in k and 'model_ema.num_updates' not in k}
            print("Load from EMA!")
            
        else:
            weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.') and 'loss' not in k}
        
    
        model.load_state_dict(weights, strict=True)
            
  
    return model.eval()

if __name__ == '__main__':

    config = get_config()
    save_path = config.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_name = config.save_file_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.geouni.llm_model_path)

    uni_prompting = UniversalPrompting(tokenizer, max_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                            "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>", "<formalization>", "</formalization>", "<answer>", "</answer>"
                                       ),
                                       ignore_id=-100)

    vq_model = get_vq_model_class(config.model.vq_model.type)
   
    if config.model.vq_model.type == "geo": 
        vq_model = load_geo_vqgan(vq_model, config.model.vq_model.vq_model_config, ckpt_path=config.model.vq_model.pretrained_model_path).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        
        print('Load from pretrained vq_model')
    elif config.model.vq_model.type == "magvitv2":
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()

    # model = GeoUniForCausalLM.from_pretrained(config.model.geouni.pretrained_model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16).to(device)    
    model = GeoUniForCausalLM.from_pretrained(config.model.geouni.pretrained_model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map={'': device})    
    model.eval()
    
    
   # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    
    overfit_train_mode = config.get("overfit_train_mode", False)
    if overfit_train_mode:
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_info = json.load(f)
    else:
        validation_info = []
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            for line in f:
                validation_info.append(json.loads(line))
    
    temperature = 1.0
    outputs = []
    
    for item in tqdm(validation_info):
        image_path = os.path.join(config.mmu_image_root, item['image'])
        image_id = item['image'].split('/')[-1].replace('.png', '')
        image_ori = Image.open(image_path).convert("RGB")
        image_ori = crop(image_ori)
        image_ori = expand2square(image_ori, (255, 255, 255))
        image = image_transform(image_ori, resolution=config.dataset.preprocessing.resolution).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        if overfit_train_mode:
            assert item['conversations'][0]['from'] == 'human'
            assert item['conversations'][1]['from'] == 'gpt'
            question = item['conversations'][0]['value']
            gt = item['conversations'][1]['value']
        else:
            question = item['text']
        prompt = question
        input_ids, _ = uni_prompting([image_tokens, prompt], 'mmu_gen')
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids,
                                        max_new_tokens=config.max_new_tokens,
                                        temperature=temperature,
                                        pad_token_id=uni_prompting.text_tokenizer.convert_tokens_to_ids('[PAD]'),
                                        eos_token_id = uni_prompting.text_tokenizer.eos_token_id,
                                        do_sample=False,
                                        top_p=None,
                                        # eot_token=uni_prompting.sptids_dict['<|eot|>'],
                                        use_cache=True)

        respone = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        if overfit_train_mode:
            print(f'gt: {gt}')
            print(f'generate: {respone}')
        outputs.append({'question_id': image_id,
                        'prompt': prompt,
                        'response': respone})

with open(os.path.join(save_path, f'{save_file_name}.jsonl'), 'w') as f:
    for line in outputs:
        f.write(json.dumps(line) + '\n')

