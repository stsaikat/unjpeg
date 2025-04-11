import torch
from torchvision import transforms
from PIL import Image
import sys
from model import JPEGCompressionPredictor

# ---------- CONFIG ----------
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- TRANSFORM ----------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------- LOAD MODEL ----------
model = JPEGCompressionPredictor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------- PREDICT FUNCTION ----------
def predict_jpeg_quality(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        predicted_quality_float = model(image_tensor).item()
        predicted_quality = round(predicted_quality_float / 10) * 10

    return predicted_quality, predicted_quality_float

# ---------- MAIN ----------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_single.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    try:
        predicted_quality, precise = predict_jpeg_quality(image_path)
        print(f"Predicted JPEG Quality: {predicted_quality} (Precise: {precise:.2f})")
    except Exception as e:
        print(f"Error: {e}")
