import torch
from models import MAGVITv2, VQModel 
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
from training.geo_data_aug import crop
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

def load_config(config_path, display=True):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_yaml(config))
    return config

def load_vqgan_new(config, ckpt_path=None):
    model = VQModel(**config.model.init_args)
    
    if ckpt_path is not None:
        # 加载检查点文件中的 state_dict
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
        # 提取出普通模型权重和 EMA 权重
        model_weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.')}
        ema_weights = {k.replace('model_ema.', ''): v for k, v in sd.items() if k.startswith('model_ema.')}
        
        # 加载普通模型的权重
        model.load_state_dict(model_weights, strict=False)
        
        # 加载 EMA 模型的权重
        if ema_weights:
            model.load_state_dict(ema_weights, strict=False)
            print("Load from EMA!")
    
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

def compare_vq_models(model1, model2, img_folder, num_of_sample=100, resolution=512):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model1.to(device).eval()
    model1.requires_grad_(False)
    
    model2.to(device).eval()
    model2.requires_grad_(False)

    # 获取文件夹中的所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]
    selected_imgs = random.sample(img_files, num_of_sample)

    img_tokens_model1 = []
    img_tokens_model2 = []

    for img_file in tqdm(selected_imgs):
        # 图像路径
        img_path = os.path.join(img_folder, img_file)
        
        # 打开图像, 图像预处理
        img = crop(Image.open(img_path).convert("RGB"))
        img = expand2square(img, (255, 255, 255))
        img_tensor = image_transform(img, resolution=resolution).unsqueeze(0).to(device)

        # 模型1推理
        with torch.no_grad():
            img_token_model1 = model1.get_code(img_tensor)
            print("img_token_model1: ", img_token_model1.shape)
            
        img_tokens_model1.append(img_token_model1.detach().cpu())
        
        
        # 模型2推理
        with torch.no_grad():
            img_token_model2 = model2.get_code(img_tensor)
            print("img_token_model2: ", img_token_model2.shape)

        img_tokens_model2.append(img_token_model2.detach().cpu())
        
    img_tokens_model1 = torch.cat(img_tokens_model1, dim=0).detach().cpu().numpy()
    img_tokens_model2 = torch.cat(img_tokens_model2, dim=0).detach().cpu().numpy()

    # 将所有 token 拉平为一个一维数组
    tokens_model1 = img_tokens_model1.flatten()
    tokens_model2 = img_tokens_model2.flatten()

    # 绘制两个模型的token分布对比图
    plt.figure(figsize=(10, 6))
    plt.hist(tokens_model1, bins=100, range=(0, 8192), color='skyblue', edgecolor='black', alpha=0.7, label='Model 1')
    plt.hist(tokens_model2, bins=100, range=(0, 8192), color='orange', edgecolor='black', alpha=0.7, label='Model 2')
    plt.title('Token Distribution Comparison')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig("vq_models_token_distribution_comparison.png", format="png", dpi=300)  # 保存为PNG文件

if __name__ == '__main__':
    
    model1 = MAGVITv2.from_pretrained('/lustre/home/2201210053/GEOMETERY/others/show_base/')
    
    # 加载模型二
    config_file_2 = "/lustre/home/2201210053/GEOMETERY/results1031/vqgan/test/config.yaml"
    ckpt_path_2 = "/lustre/home/2201210053/GEOMETERY/checkpoints1031/vqgan/test/epoch=149-step=44700.ckpt"
    
    config_model_2 = load_config(config_path=config_file_2, display=False)
    model2 = load_vqgan_new(config_model_2, ckpt_path=ckpt_path_2)

    img_folder = '/lustre/home/2201210053/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/diagrams'  # 图片文件夹路径
    compare_vq_models(model1, model2, img_folder)
