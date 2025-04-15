import os
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
from math import log10
import argparse

from restore_predictor.model import UNet
from quality_predictor.model import JPEGCompressionPredictor

# -------- Config -------- #
PREDICTOR_PATH = "quality_predictor/saved_models/v6_model_epoch_981_14.5304.pth"
RESTORE_MODEL_PATH = "restore_predictor/saved_models/v9_model_epoch_169_0.0005.pth"
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

def load_models():
    # Load JPEG Quality Predictor
    predictor = JPEGCompressionPredictor().to(DEVICE)
    predictor.load_state_dict(torch.load(PREDICTOR_PATH, map_location=DEVICE))
    predictor.eval()

    # Load Restorer Model
    restorer = UNet(in_channels=4, out_channels=3).to(DEVICE)
    restorer.load_state_dict(torch.load(RESTORE_MODEL_PATH, map_location=DEVICE))
    restorer.eval()
    
    return predictor, restorer

def predict_jpeg_quality(image_path, predictor):
    image = Image.open(image_path).convert("RGB")
    i, j, h, w = T.RandomCrop.get_params(image, output_size=(128, 128))
    i = int(i/128) * 128
    j = int(j/128) * 128
    image = T.functional.crop(image, i, j, h, w)
    
    image_tensor = transform_predictor(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        predicted_quality_float = predictor(image_tensor).item()
        predicted_quality = round(predicted_quality_float / 10) * 10
    return predicted_quality, predicted_quality_float

def calculate_psnr_pil(img1_pil, img2_pil):
    img1 = to_tensor(img1_pil).unsqueeze(0)
    img2 = to_tensor(img2_pil).unsqueeze(0)
    assert img1.shape == img2.shape, "Images must be the same size"
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float("inf")
    psnr = 10 * log10(1.0 / mse.item())
    return psnr

def restore_image(image_path, predictor, restorer):
    # Load and predict quality
    predicted_quality, precise = predict_jpeg_quality(image_path, predictor)
    print(f"Predicted JPEG Quality: {predicted_quality} (Precise: {precise:.2f})")

    image = Image.open(image_path).convert("RGB")
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

    output_path = os.path.join(OUTPUT_DIR, "restored.png")
    output_image.save(output_path)
    print(f"Restored image saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Process a single image through the JPEG restoration pipeline')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    args = parser.parse_args()

    # Load models
    predictor, restorer = load_models()
    
    # Process the image
    restore_image(args.image_path, predictor, restorer)

if __name__ == "__main__":
    main() 