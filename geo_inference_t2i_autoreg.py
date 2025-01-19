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
from models import Showo, VQModel, get_mask_chedule, MAGVITv2, Showo_t2i_autoreg
from omegaconf import OmegaConf
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_casual_attention_mask
from training.utils import get_config, flatten_omega_conf, image_transform
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

def load_vqgan_new(vq_model, config, ckpt_path=None, use_ema=True):
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


def load_config(config_path, display=True):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_yaml(config))
    return config

if __name__ == '__main__':
    # 原本图片的目录
    ori_image_root_path = 'data/formalgeo7k/formalgeo7k_v2'
    


    config = get_config()
    save_path = config.output_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")

    uni_prompting = UniversalPrompting(tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=("<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"),
                                       ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type)

    
    if config.model.vq_model.type == "geo": 
        vq_model = load_vqgan_new(vq_model, config.model.vq_model.vq_model_config, ckpt_path=config.model.vq_model.pretrained_model_path).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()
        
        print('Load from pretrained vq_model')
    elif config.model.vq_model.type == "magvitv2":
        vq_model = vq_model.from_pretrained(config.model.vq_model.vq_model_name).to(device)
        vq_model.requires_grad_(False)
        vq_model.eval()

    model = Showo_t2i_autoreg.from_pretrained(config.model.showo.pretrained_model_path).to(device)
    model.eval()


    # load from users passed arguments
    if config.get("validation_prompts_file", None) is not None:
        config.dataset.params.validation_prompts_file = config.validation_prompts_file
    config.training.batch_size = config.batch_size
    # load from users passed arguments
    if config.mode == 't2i':
        validation_info = []
        with open(config.dataset.params.validation_prompts_file, "r") as f:
            validation_info = json.load(f)

        for step in tqdm(range(0, len(validation_info), config.training.batch_size)):
            temp_info = validation_info[step:step + config.training.batch_size]
            prompts = []
            for item in temp_info:
                assert item['conversations'][0]['from'] == 'human'
                prompts.append(item['conversations'][0]['value'])


            input_ids, _ = uni_prompting(prompts, 't2i_autoreg_gen')
            input_ids = input_ids.to(device)

            
            # attention_mask = create_casual_attention_mask(input_ids,
            #                                      return_inverse_mask=True)
            # attention_mask = attention_mask.to(device)
            

            with torch.no_grad():
                gen_token_ids = model.t2i_generate(
                    input_ids=input_ids,
                    # attention_mask=attention_mask,
                    temperature=config.training.get("generation_temperature", 1.0),
                    config=config,
                )

            gen_token_ids = torch.clamp(gen_token_ids, max=config.model.showo.codebook_size - 1, min=0)
            images = vq_model.decode_code(gen_token_ids)

            images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
            images *= 255.0
            #images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            images = images.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            gen_images = [Image.fromarray(image) for image in images]
            for i, gen_image in enumerate(gen_images):
                caption = prompts[i]
                ori_image = Image.open(os.path.join(ori_image_root_path, temp_info[i]['image'])).convert('RGB')
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
                
                image_id = temp_info[i]['image'].split('/')[-1].replace('.png', '')
                new_image.save(os.path.join(save_path, f'combined_{image_id}.png'))  # 保存图像为 PNG 格式


            
