import torch
from models import VQModel
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
from training.geo_data_aug import crop
from omegaconf import OmegaConf

def load_config(config_path, display=True):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_yaml(config))
    return config

def load_vqgan_new(config, ckpt_path=None):
    model = MAGVIT2(**config.model.init_args)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()


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
    
    

def image_transform(image, resolution=256):
    # 定义预处理步骤：调整大小、中心裁剪、转为张量、归一化
    preprocess = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # 对图像应用预处理
    transformed_image = preprocess(image)

    return transformed_image

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = "/lustre/home/2201210053/GEOMETERY/results1031/vqgan/test/config.yaml"
    ckpt_path = "/lustre/home/2201210053/GEOMETERY/checkpoints1031/vqgan/test/epoch=102-step=30694.ckpt"
    
    # 加载模型
    config_model = load_config(config_path=config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=ckpt_path).to(device)
    
    # 图片文件夹路径
    img_folder = '/lustre/home/2001110054/Geo-Multi-Modal/data_backup/formalgeo7k/formalgeo7k_v2/diagrams'  # 记得修改文件夹路径
    
    # 获取文件夹中的所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]  # 你可以根据需要更改扩展名
    
    # 随机选择8张图片
    selected_imgs = random.sample(img_files, 8)

    
    for resolution in [256, 384, 512]:
        # 用于保存原始图像的列表
        original_images = []
        img_tensors = []  # 用于保存预处理后的图像张量
        for img_file in selected_imgs:
            # 图像路径
            img_path = os.path.join(img_folder, img_file)
            
            # 打开图像, 图像预处理
            img = crop(Image.open(img_path).convert("RGB"))  # 确保图像是RGB格式
            img = expand2square(img, (255, 255, 255))
            img_tensor = image_transform(img, resolution=resolution).unsqueeze(0)  # 增加batch维度但不移动到device
            img_tensors.append(img_tensor)  # 收集张量
            original_images.append(img.resize((resolution, resolution)))  # 存储调整大小后的原始图片

        # 将所有图像张量组合成一个batch
        img_batch = torch.cat(img_tensors, dim=0).to(device)  # 组合为 (8, 3, 512, 512)

        # 模型推理
        with torch.no_grad():
            img_tokens = vq_model.get_code(img_batch)  # 获取图像tokens
            print(f'img_tokens shape is {img_tokens.shape}')
            print(f'img_tokens is {img_tokens}')
            recovered_imgs = vq_model.decode_code(img_tokens)  # 解码重建图像

        # 后处理：将重建后的图片从 [-1, 1] 转换回 [0, 255]
        recovered_imgs = torch.clamp((recovered_imgs + 1.0) / 2.0, min=0.0, max=1.0)
        recovered_imgs *= 255.0
        recovered_imgs = recovered_imgs.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # 转换为numpy

        # 保存重建图像的 PIL 列表
        recovered_images = [Image.fromarray(recovered_img) for recovered_img in recovered_imgs]

        # 拼接成2x8的图像网格
        combined_width = original_images[0].width * 8  # 8张图片并排
        combined_height = original_images[0].height * 2  # 两行，第一行原图，第二行重建图

        # 创建一张空白的图片用于拼接
        combined_img = Image.new('RGB', (combined_width, combined_height))

        # 将原始图片和恢复图片粘贴到空白图上
        for i in range(8):
            # 第一行粘贴原始图像
            combined_img.paste(original_images[i], (i * resolution, 0))  # 原始图片放在第一行
            # 第二行粘贴恢复的图像
            combined_img.paste(recovered_images[i], (i * resolution, resolution))  # 重建图片放在第二行

        # 保存拼接后的图像
        combined_img.save(f'combined_image_grid_{resolution}.jpg')
