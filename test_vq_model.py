import torch
from models import MAGVITv2
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import random



def find_bounds(image):
    """找到图像中非白色区域的边界"""
    # 将图像转换为numpy数组
    np_image = np.array(image)
    
    # 找出所有非白色的像素点
    non_white_pixels = np.any(np_image < [250, 250, 250], axis=-1)
    rows, cols = np.where(non_white_pixels)
    
    # 获取非白色像素点的最小和最大坐标
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    return min_row, max_row, min_col, max_col

def crop(image, buffer=20):
    """在确保不丢失关键信息的前提下，剪裁图像"""
    min_row, max_row, min_col, max_col = find_bounds(image)
    
    # 添加缓冲区以确保不剪掉边界附近的重要信息
    min_row = max(0, min_row - buffer)
    max_row = min(image.height, max_row + buffer)
    min_col = max(0, min_col - buffer)
    max_col = min(image.width, max_col + buffer)
    
    # 剪裁图像
    return image.crop((min_col, min_row, max_col, max_row))



def random_rotate(image, angle=30):
    """对图像进行随机小角度旋转"""
    angle = random.uniform(-angle, angle)  # 随机旋转角度范围为-10度到10度
    return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))

def add_gaussian_noise(image, mean=0, std=15):
    """向图像添加高斯噪声"""
    np_image = np.array(image)
    gaussian = np.random.normal(mean, std, np_image.shape)
    noisy_image = np_image + gaussian
    noisy_image = np.clip(noisy_image, 0, 255)  # 限制值在有效范围内
    return Image.fromarray(noisy_image.astype('uint8'))



def enhance_image(image, enhancements=None, probabilities=None):
    """根据给定的增强列表和相应的概率随机应用增强方法"""
    if enhancements is None:
        enhancements = [random_rotate, crop]
    if probabilities is None:
        probabilities = [0.5, 1.0]  # 每种方法被选中的概率
        
    for enhance, probability in zip(enhancements, probabilities):
        if random.random() < probability:
            image = enhance(image)
    return image

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
    
    # 随机选择8张图片
    selected_imgs = random.sample(img_files, 8)

    # 图像预处理（调整大小、转换为张量、归一化）
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # 可根据需要调整图像大小
        transforms.ToTensor(),  # 转换为tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1, 1]
    ])
    
    # 用于保存原始图像的列表
    original_images = []
    img_tensors = []  # 用于保存预处理后的图像张量

    for img_file in selected_imgs:
        # 图像路径
        img_path = os.path.join(img_folder, img_file)
        
        # 打开图像
        img = crop(Image.open(img_path).convert("RGB"))  # 确保图像是RGB格式
        
        # 图像预处理
        img_tensor = preprocess(img).unsqueeze(0)  # 增加batch维度但不移动到device
        img_tensors.append(img_tensor)  # 收集张量
        original_images.append(img.resize((512, 512)))  # 存储调整大小后的原始图片

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
        combined_img.paste(original_images[i], (i * 512, 0))  # 原始图片放在第一行
        # 第二行粘贴恢复的图像
        combined_img.paste(recovered_images[i], (i * 512, 512))  # 重建图片放在第二行

    # 保存拼接后的图像
    combined_img.save('combined_image_grid.jpg')
