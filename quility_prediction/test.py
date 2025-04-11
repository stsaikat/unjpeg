import os
import torch
from torchvision import transforms
from PIL import Image
from model import JPEGCompressionPredictor  # Update path if needed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- CONFIG ----------
MODEL_PATH = "model.pth"
TEST_DIR = "dataset/test"
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

# ---------- INFERENCE ----------
predictions = []
actuals = []

with torch.no_grad():
    for filename in os.listdir(TEST_DIR):
        if filename.endswith(".jpg"):
            path = os.path.join(TEST_DIR, filename)
            image = Image.open(path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)

            predicted_quality_float = model(image_tensor).item()
            predicted_quality = round(predicted_quality_float / 10) * 10

            # Extract actual quality from filename
            try:
                actual_quality = int(filename.split("_q")[-1].split(".")[0])
                predictions.append(predicted_quality)
                actuals.append(actual_quality)
                print(f"{filename} => Predicted: {predicted_quality:.2f}, Actual: {actual_quality}")
                if int(predicted_quality) != int(actual_quality):
                    print(f"Mismatch for {filename}: Predicted {predicted_quality}, Actual {actual_quality} Precisely {predicted_quality_float}")
            except ValueError:
                print(f"Skipping {filename} — cannot extract quality from name.")

# ---------- ACCURACY ----------
def regression_accuracy(actuals, predictions):
    correct = sum(int(a) == int(p) for a, p in zip(actuals, predictions))
    return 100.0 * correct / len(actuals)

if predictions and actuals:
    acc = regression_accuracy(actuals, predictions)

    print("\n--- Evaluation Metrics ---")
    print(f"tested on {len(actuals)} images")
    print(f"Accuracy : {acc:.2f}%")

