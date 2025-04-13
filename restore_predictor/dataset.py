import os
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torch

class CompressedImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        
        # Common transform: random crop and tensor conversion
        self.crop_size = (128, 128)
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load original image
        original = Image.open(img_path).convert("RGB")

        # Random crop params
        i, j, h, w = T.RandomCrop.get_params(original, output_size=self.crop_size)
        i = int(i/self.crop_size[0]) * self.crop_size[0]
        j = int(j/self.crop_size[1]) * self.crop_size[1]

        # print(f"Image: {img_name}, Crop: ({i}, {j}, {h}, {w})")
        # Crop original (this will be the mask)
        target = T.functional.crop(original, i, j, h, w)

        quality_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # quality_list = [30]
        quality = random.choice(quality_list)
        # Compress cropped image to JPEG with quality=10 (simulate compression artifacts)
        buffer = BytesIO()
        target.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")

        # Convert both to tensors
        input_tensor = self.to_tensor(compressed)
        target_tensor = self.to_tensor(target)
        
        # Add quality channel
        quality_channel = torch.full((1, self.crop_size[0], self.crop_size[1]), quality / 100.0)
        input_tensor = torch.cat([input_tensor, quality_channel], dim=0)

        return input_tensor, target_tensor
