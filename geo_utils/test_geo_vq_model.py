import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random
from omegaconf import OmegaConf
import sys
sys.path.append('/lustre/home/2001110054/Show-o')  # 项目根目录
from training.geo_data_aug import crop
from models import VQModel, MAGVITv2


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    return Image.fromarray(x).convert("RGB")

def load_config(config_path, display=True):
    config = OmegaConf.load(config_path)
    if display:
        print(OmegaConf.to_yaml(config))
    return config

def load_vqgan_new(config, ckpt_path=None, use_ema=True):
    model = VQModel(**config.model.init_args.ddconfig)
    if ckpt_path is not None:
        # 加载检查点文件中的 state_dict
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        
        # 提取出普通模型权重和 EMA 权重
        if use_ema:
            key_map = {k.replace('.', ''): k for k in sd.keys() if not k.startswith('model_ema.') and 'loss' not in k} 
            weights = {key_map[k.replace('model_ema.', '')]: v for k, v in sd.items() if k.startswith('model_ema.') and 'loss' not in k and 'model_ema.decay' not in k and 'model_ema.num_updates' not in k}
            print("Load from EMA!")
            # ! Todo: fix keys error in ema!!!!
        else:
            weights = {k: v for k, v in sd.items() if not k.startswith('model_ema.') and 'loss' not in k}
        model.load_state_dict(weights, strict=True)
            
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--ckpt')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_1 = MAGVITv2.from_pretrained('showlab/magvitv2').to(device)
    
    # config_file = "/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1110_mask/show/config.yaml"
    # ckpt_path = "/lustre/home/2001110054/GEO-Open-MAGVIT2/outputs/expr_1110_mask/ckpt/epoch=135-step=50592.ckpt"
    config_file = args.config
    ckpt_path = args.ckpt
    
    # 加载模型
    config_model = load_config(config_path=config_file, display=False)
    model_2 = load_vqgan_new(config_model, ckpt_path=ckpt_path).to(device)
    
    # 图片文件夹路径
    img_folder = '/lustre/home/2001110054/GEO-Open-MAGVIT2/geo_data/val'  # 记得修改文件夹路径
    
    # 获取文件夹中的所有图片文件
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.png')]  # 你可以根据需要更改扩展名
    
    # 随机选择8张图片
    selected_imgs = random.sample(img_files, 8)
    
    resolution = 512
    original_images = []
    img_tensors = [] 
    
    for img_file in selected_imgs:
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
        img_tokens = model_1.get_code(img_batch)
        reconstructed_images_1 = model_1.decode_code(img_tokens)
        
            
        quant, indices = model_2.encode(img_batch)
        reconstructed_images_2 = model_2.decode(quant)
            
    recovered_images_1 = [custom_to_pil(recovered_img)for recovered_img in reconstructed_images_1]
    recovered_images_2 = [custom_to_pil(recovered_img)for recovered_img in reconstructed_images_2]

        
    # 拼接成2x8的图像网格
    combined_width = original_images[0].width * 8  # 8张图片并排
    combined_height = original_images[0].height * 3  # 3行，第一行原图，第二行原始show/magvit, 第三行我们自己训练的magvit

    # 创建一张空白的图片用于拼接
    combined_img = Image.new('RGB', (combined_width, combined_height))
    for i in range(8):
        combined_img.paste(original_images[i], (i * resolution, 0))  # 原始图片放在第一行
        combined_img.paste(recovered_images_1[i], (i * resolution, resolution))  # 原始show/magvit重建图片放在第二行
        combined_img.paste(recovered_images_2[i], (i * resolution, resolution * 2)) # 自己训练的magvit重建图片放在第三行

    # 保存拼接后的图像
    combined_img.save(f'combined_image_grid_512.jpg')
