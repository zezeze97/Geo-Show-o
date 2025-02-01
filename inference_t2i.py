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

os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["WANDB_MODE"]="offline"
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import torch
# import wandb
from models import VQModel, MAGVITv2, GeoUniForCausalLM
from omegaconf import OmegaConf
from training.prompting_utils import UniversalPrompting
from training.utils import get_config
from transformers import AutoTokenizer
import torch.nn.functional as F
import json
from training.geo_data_aug import crop
from training.custom_data import expand2square

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
    # 原本图片的目录
    ori_image_root_path = 'data/formalgeo7k/formalgeo7k_v2'
    


    config = get_config()
    save_path = config.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.geouni.llm_model_path)

    uni_prompting = UniversalPrompting(tokenizer, max_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|sot|>", "<|eot|>", "<|soi|>", "<|eoi|>", "<|t2i|>", "<|formalization|>", "<|reasoning|>", "<|step|>", "<|conclusion|>"
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

    model = GeoUniForCausalLM.from_pretrained(config.model.geouni.pretrained_model_path, attn_implementation='sdpa', torch_dtype=torch.bfloat16).to(device)    
    model.eval()


    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_info = json.load(f)

    for item in tqdm(validation_info):
        assert item['conversations'][0]['from'] == 'human'
        prompt = item['conversations'][0]['value']
        input_ids, attention_masks = uni_prompting(prompt, 't2i_gen')
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        with torch.no_grad():
            gen_token_ids = model.t2i_generate(
                input_ids=input_ids,
                pad_token_id=uni_prompting.text_tokenizer.pad_token_id,
                attention_masks=attention_masks,
                temperature=1.0,
            )

        image = vq_model.decode_code(gen_token_ids)

        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
        image *= 255.0
        #images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        image = image.detach().cpu().permute(0, 2, 3, 1).squeeze(0).numpy().astype(np.uint8)
        gen_image = Image.fromarray(image)
        
        caption = prompt
        ori_image = Image.open(os.path.join(ori_image_root_path, item['image'])).convert('RGB')
        ori_image = expand2square(crop(ori_image), (255, 255, 255))
        # 调整 ori_image 的大小，使其与 gen_image 保持一致
        ori_image = ori_image.resize(gen_image.size, Image.Resampling.LANCZOS)
            # 创建一个新的图像，宽度为两个图像宽度之和，高度为两个图像高度加上文本高度
        new_width = gen_image.width + ori_image.width
        # text_height = font.getsize(caption)[1]  # 获取文本高度
        new_height = gen_image.height # + text_height
        new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))  # 白色背景

        # 将 gen_image 和 ori_image 拼接到新图像中
        new_image.paste(gen_image, (0, 0))
        new_image.paste(ori_image, (gen_image.width, 0))

        # 在新图像下方添加文本
        # draw = ImageDraw.Draw(new_image)
        # text_width = font.getsize(caption)[0]
        # text_x = (new_width - text_width) // 2  # 文本居中
        # text_y = gen_image.height
        # draw.text((text_x, text_y), caption, font=font, fill=(0, 0, 0))  # 黑色文本
        
        image_id = item['image'].split('/')[-1].replace('.png', '')
        new_image.save(os.path.join(save_path, f'combined_{image_id}.png'))  # 保存图像为 PNG 格式


        
