# coding=utf-8
import json
import os
from torch.utils.data import Dataset
from .geo_data_aug import enhance_image
from PIL import Image
import torch
from torchvision import transforms


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



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, image_folder: str=None,
                 json_path: str=None, 
                 geo_customized_aug: bool=True,
                 image_aspect_ratio: str='pad',
                 resolution: int = 256,
                 is_t2i: bool = False,
                 is_formalization: bool = False,
                 is_reasoning: bool = False
                 ):
        super(LazySupervisedDataset, self).__init__()
        
        with open(json_path, "r") as f:
            list_data_dict = json.load(f)
        self.image_folder = image_folder
        self.geo_customized_aug = geo_customized_aug
        self.image_aspect_ratio = image_aspect_ratio
        self.resolution = resolution
        self.list_data_dict = list_data_dict
        self.is_t2i = is_t2i
        self.is_formalization = is_formalization
        self.is_reasoning = is_reasoning
    

    def __len__(self):
        return len(self.list_data_dict)


    def __getitem__(self, i):
        sources = self.list_data_dict[i]
        
        assert len(sources['conversations']) == 2
        instruction = ''
        response = ''
        for item in sources['conversations']:
            if item['from'] == 'human':
                instruction = item['value']
            elif item['from'] == 'gpt':
                response = item['value']
        
        if 'image' in sources:
            image_file = sources['image']
            image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
            if self.geo_customized_aug:
                image = enhance_image(image)
            if self.image_aspect_ratio == 'pad':
                
                image = expand2square(image, (255, 255, 255))
            image = image_transform(image, self.resolution)
        else:
            image = torch.zeros(3, self.resolution, self.resolution)
    
        if self.is_t2i:
            data_dict = {
                "images": image,
                "instructions": instruction
            }
        elif self.is_formalization or self.is_reasoning:
            # text = 'USER: \n' + instruction + ' ASSISTANT:' + response
            
            if instruction.startswith('<image>\n'):
                instruction = instruction.replace('<image>\n', '')
            
            data_dict = {
                "images": image,
                "instructions": instruction,
                "responses": response
            }
               
        return data_dict





if __name__ == '__main__':
    dataset = LazySupervisedDataset(image_folder='/lustre/home/2201210053/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2',
                                json_path='/lustre/home/2201210053/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/custom_json/qa_resoning/formalgeov2_aug_train.json',
                                )
    for item in dataset:
        print(item)
        break