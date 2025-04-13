import os
import sys
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from io import BytesIO
from math import log10

from restore_predictor.model import UNet
from quality_predictor.model import JPEGCompressionPredictor
import shutil

# -------- Config -------- #
PREDICTOR_PATH = "quality_predictor/model.pth"
RESTORE_MODEL_PATH = "restore_predictor/saved_models/v8_model_latest_0.0005_qr.pth"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE = 128

# -------- Transforms -------- #
transform_predictor = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor()
])
to_tensor = T.ToTensor()
to_pil = T.ToPILImage()


# -------- Load JPEG Quality Predictor -------- #
predictor = JPEGCompressionPredictor().to(DEVICE)
predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
predictor.eval()


# -------- Load Restorer Model -------- #
restorer = UNet(in_channels=4, out_channels=3).to(DEVICE)
restorer.load_state_dict(torch.load(RESTORE_MODEL_PATH, map_location=DEVICE))
restorer.eval()


# -------- Quality Prediction -------- #
def predict_jpeg_quality(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128), resample=Image.Resampling.NEAREST)
    image.save("tmp.jpg", format="JPEG", quality=10)  # Save as JPEG to simulate compression
    image = Image.open("tmp.jpg").convert("RGB")
    image_tensor = transform_predictor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_quality_float = predictor(image_tensor).item()
        predicted_quality = round(predicted_quality_float / 10) * 10
    return predicted_quality, predicted_quality_float


# -------- PSNR Calculation -------- #
def calculate_psnr_pil(img1_pil, img2_pil):
    img1 = to_tensor(img1_pil).unsqueeze(0)
    img2 = to_tensor(img2_pil).unsqueeze(0)
    assert img1.shape == img2.shape, "Images must be the same size"
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 10 * log10(1.0 / mse.item())
    return psnr


# -------- Restoration Pipeline -------- #
def restore_image(image_path):
    # Load and predict quality
    predicted_quality, precise = predict_jpeg_quality(image_path)
    print(f"Predicted JPEG Quality: {predicted_quality} (Precise: {precise:.2f})")

    image = Image.open(image_path).convert("RGB")
    # image.save(f"{OUTPUT_DIR}/original.jpg", format="JPEG", quality=predicted_quality)
    width, height = image.size

    # Pad if needed
    pad_w = (TILE_SIZE - width % TILE_SIZE) % TILE_SIZE
    pad_h = (TILE_SIZE - height % TILE_SIZE) % TILE_SIZE
    if pad_w or pad_h:
        image = F.pad(to_tensor(image), (0, pad_w, 0, pad_h), mode="reflect")
        image = to_pil(image)
        width += pad_w
        height += pad_h

    output_image = Image.new("RGB", (width, height))

    for top in range(0, height, TILE_SIZE):
        for left in range(0, width, TILE_SIZE):
            patch = image.crop((left, top, left + TILE_SIZE, top + TILE_SIZE))
            input_tensor = to_tensor(patch).unsqueeze(0).to(DEVICE)
            quality_channel = torch.full((1, 1, TILE_SIZE, TILE_SIZE), predicted_quality / 100.0).to(DEVICE)
            input_tensor = torch.cat([input_tensor, quality_channel], dim=1)

            with torch.no_grad():
                output = restorer(input_tensor).clamp(0, 1)

            output_patch = to_pil(output.squeeze(0).cpu())
            output_image.paste(output_patch, (left, top))

    output_image.save(f"{OUTPUT_DIR}/restored.png")

    # -------- PSNR Metrics -------- #
    original_img = Image.open('0001.png').convert("RGB")
    jpeg_img = Image.open(image_path).convert("RGB")
    restored_img = Image.open(f"{OUTPUT_DIR}/restored.png").convert('RGB')

    psnr_input = calculate_psnr_pil(original_img, jpeg_img)
    psnr_output = calculate_psnr_pil(original_img, restored_img)

    print(f"Input PSNR: {psnr_input:.2f} dB")
    print(f"Restored PSNR: {psnr_output:.2f} dB")
    if psnr_output < psnr_input:
        print("⚠️  Warning: Restored PSNR is worse than JPEG input PSNR")


# -------- Main Entry -------- #
if __name__ == "__main__":
    png_img_path = "0001.png"
    input_img_path = "test.jpg"
    img = Image.open(png_img_path).convert("RGB")
    img.save(input_img_path, format="JPEG", quality=10)
    
    # predicted_quality, precise = predict_jpeg_quality(input_img_path)
    # print(f"Predicted JPEG Quality: {predicted_quality} (Precise: {precise:.2f})")
    restore_image(input_img_path)
    
    # for _ in range(10):
    #     shutil.copy("output/restored.png", f"original.jpg")
    #     img_path = "original.jpg"
    #     restore_image(img_path)
