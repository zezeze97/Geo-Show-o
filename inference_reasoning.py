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
from models import MAGVITv2, VQModel, GeoUniForCausalLM
from src.open_r1.trainer.prompting_utils import UniversalPrompting
from src.open_r1.trainer.geo_data_aug import crop
from src.open_r1.trainer.custom_data import image_transform
from transformers import AutoTokenizer
import json
from omegaconf import OmegaConf

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
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
        
         # 提取出普通模型权重和 EMA 权重
        if use_ema:
            key_map = {k.replace('.', ''): k for k in sd.keys() if not k.startswith('model_ema.') and 'loss' not in k} 
            weights = {key_map[k.replace('model_ema.', '')]: v for k, v in sd.items() if k.startswith('model_ema.') and 'loss' not in k and 'model_ema.decay' not in k and 'model_ema.num_updates' not in k}
            print("Load from EMA!")
            
        else:
            weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.') and 'loss' not in k}
        
    
        model.load_state_dict(weights, strict=True)
            
  
    return model.eval()


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf

if __name__ == '__main__':

    config = get_config()
    save_path = config.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file_name = config.save_file_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.geouni.llm_model_path)

    uni_prompting = UniversalPrompting(tokenizer,
                                       special_tokens=(
                                            "<|soi|>", "<|eoi|>", "<|t2i|>", "<|mmu|>", "<|mix|>", "<formalization>", "</formalization>", "<answer>", "</answer>"
                                       ),
                                       ignore_id=-100)

    vq_model = get_vq_model_class(config.vq_model.type)
   
    if config.vq_model.type == "geo": 
        vq_model = load_geo_vqgan(vq_model, config.vq_model.vq_model_config, ckpt_path=config.vq_model.pretrained_model_path).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        print(f'Load from pretrained vq_model: {config.vq_model.pretrained_model_path}')


    # model = GeoUniForCausalLM.from_pretrained(config.model.geouni.pretrained_model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16).to(device)    
    model = GeoUniForCausalLM.from_pretrained(config.geouni.pretrained_model_path, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16, device_map={'': device})    
    model.eval()
    
    
    # load from users passed arguments
    validation_info = []
    with open(config.validation_prompts_file, "r") as f:
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
        image = image_transform(image_ori, resolution=config.vq_model.vq_model_config.resolution).to(device)
        image = image.unsqueeze(0)
        image_tokens = vq_model.get_code(image) + len(uni_prompting.text_tokenizer)
        question = item['text']
        if config.language == 'en':
            if config.formalization:
                prompt = f"Analyze the input geometry image to extract consCDL and imgCDL, then answer the question.\nQuestion: {question}"
            else:
                prompt = f"Answer the question based on the provided geometry image.\nQuestion: {question}"
        elif config.language == 'cn':
            if config.formalization:
                prompt = f"根据输入的几何图像和问题，首先分析图像提取 consCDL 和 imgCDL，然后给出答案。\n问题：{question}"
            else:
                prompt = f"请根据提供的几何图像回答问题。\n问题：{question}"
        input_ids, attention_mask = uni_prompting([image_tokens, prompt], 'mmu_gen')
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        padding_size = 10
        padding = torch.full((1, padding_size), uni_prompting.text_tokenizer.pad_token_id, dtype=torch.long, device=input_ids.device)
        padded_input_ids = torch.cat([padding, input_ids], dim=1)
        
        padding_mask = torch.full((1, padding_size), 0, dtype=torch.long, device=attention_mask.device)  # pad_token_id is 0
        padding_mask = padding_mask.to(device)
        padded_attention_mask = torch.cat([padding_mask, attention_mask], dim=1)
        
        
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        max_new_tokens=config.max_new_tokens,
                                        temperature=temperature,
                                        pad_token_id=uni_prompting.text_tokenizer.pad_token_id,
                                        eos_token_id = uni_prompting.text_tokenizer.eos_token_id,
                                        do_sample=False,
                                        top_p=None,
                                        use_cache=False)
            output_ids_padded = model.generate(input_ids=padded_input_ids,
                                        attention_mask=padded_attention_mask,
                                        max_new_tokens=config.max_new_tokens,
                                        temperature=temperature,
                                        pad_token_id=uni_prompting.text_tokenizer.pad_token_id,
                                        eos_token_id = uni_prompting.text_tokenizer.eos_token_id,
                                        do_sample=False,
                                        top_p=None,
                                        use_cache=False)

        response = uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        response_padded = uni_prompting.text_tokenizer.batch_decode(output_ids_padded[:, padded_input_ids.shape[1]:], skip_special_tokens=True)[0]
        print(f'generate: {response}')
        print(f'padded generate: {response_padded}')
        
        
        
        
        outputs.append({'question_id': image_id,
                        'prompt': prompt,
                        'response': response})

with open(os.path.join(save_path, f'{save_file_name}.jsonl'), 'w') as f:
    for line in outputs:
        f.write(json.dumps(line) + '\n')

