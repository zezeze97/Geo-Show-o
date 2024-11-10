import torch
from models import MAGVITv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
from training.geo_data_aug import crop
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    
    
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    vq_model = MAGVITv2.from_pretrained('showlab/magvitv2').to(device)
    vq_model.requires_grad_(False)
    vq_model.eval()
    
    # 图片文件夹路径
    img_folder = '/lustre/home/2001110054/Geo-Multi-Modal/data_backup/formalgeo7k/formalgeo7k_v2/diagrams'  # 记得修改文件夹路径
    
    # 获取文件夹中的所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]  # 你可以根据需要更改扩展名
    
    num_of_sample = 100
    
    # 随机选择8张图片
    selected_imgs = random.sample(img_files, num_of_sample)
    resolution = 512

    
    
    img_tokens = []
    for img_file in tqdm(selected_imgs):
        # 图像路径
        img_path = os.path.join(img_folder, img_file)
        
        # 打开图像, 图像预处理
        img = crop(Image.open(img_path).convert("RGB"))  # 确保图像是RGB格式
        img = expand2square(img, (255, 255, 255))
        img_tensor = image_transform(img, resolution=resolution).unsqueeze(0).to(device)

        # 模型推理
        with torch.no_grad():
            img_token = vq_model.get_code(img_tensor)  # 获取图像tokens
        img_tokens.append(img_token.detach().cpu())
    img_tokens = torch.cat(img_tokens, dim=0).detach().cpu().numpy()
    print(f'img_tokens shape: {img_tokens.shape}')
    
    # 将所有 token 拉平为一个一维数组
    tokens = img_tokens.flatten()
    
    # 绘制 token 分布的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(tokens, bins=100, range=(0, 8095), color='skyblue', edgecolor='black')
    plt.title('Token Distribution')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.savefig("showlab_magvitv2_token_distribution.png", format="png", dpi=300)  # 以 300 dpi 的分辨率保存为 PNG 文件

   
