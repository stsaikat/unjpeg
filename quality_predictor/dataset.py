import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

class JPEGCompressionDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with JPEG images like img_q10.jpg, img_q50.jpg, etc.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Extract quality value from filename (e.g., "img_q30.jpg" → 30)
        try:
            quality = int(img_name.split('_q')[-1].split('.')[0])
        except ValueError:
            raise ValueError(f"Could not parse quality from filename: {img_name}")

        quality = torch.tensor(quality, dtype=torch.float32)

        return image, quality
