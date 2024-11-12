import os
import sys
sys.path.append(os.getcwd())
import torch
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from training.geo_data_aug import crop
from models import VQModel, MAGVITv2


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    return Image.fromarray(x).convert("RGB")

def load_config(config_path, display=False):
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

class MyCustomDataset(Dataset):
    def __init__(self, data_path, resolution=512):
        self.data = [os.path.join(data_path, file) for file in os.listdir(data_path) 
                     if file.endswith(('.png', '.jpg', 'jpeg'))]
        self.augmentations = crop
        self.transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        example = dict()
        img_path = self.data[idx]
        img_name = img_path.split('/')[-1].replace('.png', '')
        image = Image.open(img_path).convert("RGB")
        image = self.augmentations(image)
        image = expand2square(image, (255, 255, 255))
        image = self.transform(image)
        example['image'] = image
        example['name'] = img_name
        return example

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--use_show_magvit", action='store_true')
    parser.add_argument("--image_size", default=512, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--data_path", required=True, type=str)

    return parser.parse_args()

def main(args):
    if args.use_show_magvit:
        model = MAGVITv2.from_pretrained('showlab/magvitv2').to(DEVICE)
    else:
        # Load model configuration
        config_model = load_config(args.config_file, display=False)
        model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)


    geo_dataset = MyCustomDataset(data_path=args.data_path, resolution=args.image_size)
    
    dataloader = DataLoader(geo_dataset, batch_size=args.batch_size, num_workers=4)

    os.makedirs(args.output_dir, exist_ok=True)

    
    for examples in tqdm(dataloader):
        images = examples['image'].to(DEVICE)
        names = examples['name']
        with torch.no_grad():
            img_tokens = model.get_code(images)
            reconstructed_images = model.decode_code(img_tokens)

        # Save reconstructed images to output directory
        for i, recon_img in enumerate(reconstructed_images):
            pil_image = custom_to_pil(recon_img)
            name = names[i]
            image_path = os.path.join(args.output_dir, f"reconstructed_{name}.png")
            pil_image.save(image_path)


if __name__ == "__main__":
    args = get_args()
    main(args)
