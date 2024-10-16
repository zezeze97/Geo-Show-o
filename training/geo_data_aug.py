import random
import numpy as np
from PIL import Image

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