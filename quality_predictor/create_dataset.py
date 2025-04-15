from PIL import Image
import os
from tqdm import tqdm
import random
import shutil

def create_dataset(img_name, original_path, out_dir, qualities=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
    os.makedirs(out_dir, exist_ok=True)
    img = Image.open(original_path).convert("RGB")
    # img = img.resize((128, 128), resample=Image.Resampling.NEAREST)
    
    for q in qualities:
        save_path = os.path.join(out_dir, f"{img_name}_q{q}.jpg")
        img.save(save_path, quality=q)    
        
def create_train_dataset():
    for file in tqdm(os.listdir("dataset/images")):
        img_name = file.split(".")[0]
        original_path = os.path.join('dataset/images', file)
        out_dir = "dataset/train"
        create_dataset(img_name, original_path, out_dir)
        
def create_test_dataset():
    tarin_dir = "dataset/train"
    test_dir = "dataset/test"
    os.makedirs(test_dir, exist_ok=True)
    train_files = os.listdir(tarin_dir)
    random.shuffle(train_files)
    
    test_files = train_files[:int(len(train_files) * 0.1)]
    for file in tqdm(test_files):
        original_path = os.path.join(tarin_dir, file)
        out_dir = os.path.join(test_dir, file)
        shutil.move(original_path, out_dir)

create_train_dataset()
create_test_dataset()