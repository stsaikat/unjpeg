import os
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from model import UNet  # Make sure UNet is correctly imported
from io import BytesIO

# -------- Config -------- #
image_path = "0001.png"

# from PIL import Image

# def crop_to_multiple_of_128(pil_img):
#     width, height = pil_img.size
#     new_width = (width // 128) * 128
#     new_height = (height // 128) * 128
#     cropped = pil_img.crop((0, 0, new_width, new_height))
#     return cropped

# image = Image.open(image_path).convert("RGB")
# image = crop_to_multiple_of_128(image)
# image.save(image_path)

model_path = "saved_models/v4_model_latest_0.0007.pth"
save_path = "output/result.png"
tile_size = 128
quality = 10  # JPEG quality
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Load Model -------- #
model = UNet(in_channels=4, out_channels=3).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# -------- Transforms -------- #
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()

# -------- Load Image -------- #
image = Image.open(image_path).convert("RGB")
image.save("output/original.jpg", format="JPEG", quality=quality)  # Save original image for reference
width, height = image.size

# Padding if necessary
pad_w = (tile_size - width % tile_size) % tile_size
pad_h = (tile_size - height % tile_size) % tile_size
if pad_w != 0 or pad_h != 0:
    image = F.pad(to_tensor(image), (0, pad_w, 0, pad_h), mode="reflect")
    image = to_pil(image)
    width += pad_w
    height += pad_h

# -------- Tile Processing -------- #
output_image = Image.new("RGB", (width, height))

for top in range(0, height, tile_size):
    for left in range(0, width, tile_size):
        # Crop tile
        patch = image.crop((left, top, left + tile_size, top + tile_size))

        # JPEG compress
        buffer = BytesIO()
        patch.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_patch = Image.open(buffer).convert("RGB")

        # Add quality channel
        input_tensor = to_tensor(compressed_patch).unsqueeze(0).to(device)
        quality_channel = torch.full((1, 1, tile_size, tile_size), quality / 100.0).to(device)
        input_tensor = torch.cat([input_tensor, quality_channel], dim=1)

        # Inference
        with torch.no_grad():
            output = model(input_tensor).clamp(0, 1)

        output_patch = to_pil(output.squeeze(0).cpu())
        output_image.paste(output_patch, (left, top))

# -------- Save Result -------- #
output_image.save(save_path)
print(f"✅ Inference complete! Saved to {save_path}")

import torch
import torch.nn.functional as F
from torchvision import transforms as T
from math import log10

def calculate_psnr_pil(img1_pil, img2_pil):
    # Convert to tensors (normalized to [0, 1])
    to_tensor = T.ToTensor()
    img1 = to_tensor(img1_pil).unsqueeze(0)  # shape: [1, 3, H, W]
    img2 = to_tensor(img2_pil).unsqueeze(0)

    # Ensure same size
    assert img1.shape == img2.shape, "Images must be the same size"

    # Calculate MSE
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")

    psnr = 10 * log10(1.0 / mse.item())
    return psnr

from PIL import Image

img = Image.open(image_path).convert("RGB")
input_org_img = Image.open("output/original.jpg").convert("RGB")
output = Image.open("output/result.png").convert("RGB").resize(img.size)

print(f"input size: {img.size}")
print(f"output size: {output.size}")
print(f"input org size: {input_org_img.size}")

psnr = calculate_psnr_pil(img, input_org_img)
print(f"input PSNR: {psnr:.2f} dB")
psnr = calculate_psnr_pil(img, output)
print(f"output PSNR: {psnr:.2f} dB")
