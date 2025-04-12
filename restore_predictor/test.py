import os
import random
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.utils import save_image
from math import log10
from model import UNet  # Ensure your UNet is imported properly

# -------- Config -------- #
image_dir = "dataset/test"
model_path = "saved_models/v4_model_latest_0.0007.pth"
save_vis_dir = "test_results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
crop_size = 128

# -------- Make save directory -------- #
os.makedirs(save_vis_dir, exist_ok=True)

# -------- PSNR Function -------- #
def calculate_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    psnr = 10 * log10(1.0 / mse.item())
    return psnr

# -------- Transforms -------- #
to_tensor = T.ToTensor()

# -------- Load Model -------- #
model = UNet(in_channels=4, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------- Inference Loop -------- #
psnr_values = []

for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    image = Image.open(img_path).convert("RGB")

    # Random crop
    w, h = image.size
    if w < crop_size or h < crop_size:
        print(f"Skipping {img_name} — smaller than crop size")
        continue

    i = random.randint(0, h - crop_size)
    i = int(i/crop_size) * crop_size
    j = random.randint(0, w - crop_size)
    j = int(j/crop_size) * crop_size
    original_crop = image.crop((j, i, j + crop_size, i + crop_size))

    quality_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    quality_list = [10]
    quality = random.choice(quality_list)
    # Compress the crop (JPEG quality=10)
    buffer = BytesIO()
    original_crop.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_crop = Image.open(buffer).convert("RGB")

    # To tensor and move to device
    input_tensor = to_tensor(compressed_crop).unsqueeze(0).to(device)
    target_tensor = to_tensor(original_crop).unsqueeze(0).to(device)

    quality_channel = torch.full((1, crop_size, crop_size), quality / 100.0)
    quality_channel = quality_channel.unsqueeze(0).to(device)
    input_tensor = torch.cat([input_tensor, quality_channel], dim=1)

    # Inference
    with torch.no_grad():
        output = model(input_tensor).clamp(0, 1)

    # PSNR
    psnr = calculate_psnr(output, target_tensor)
    psnr_values.append(psnr)
    print(f"🖼️ {img_name} - PSNR: {psnr:.2f} dB")

    # Save visual comparison
    input_img = input_tensor[0].cpu()
    output_img = output[0].cpu()
    target_img = target_tensor[0].cpu()
    # vis_grid = torch.cat([input_img, output_img, target_img], dim=2)  # width-wise
    # save_path = os.path.join(save_vis_dir, f"{os.path.splitext(img_name)[0]}_vis.png")
    # save_image(vis_grid, save_path)

# -------- Average PSNR -------- #
if psnr_values:
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"\n📊 Average PSNR over {len(psnr_values)} images: {avg_psnr:.2f} dB")
else:
    print("No valid images found for testing.")
