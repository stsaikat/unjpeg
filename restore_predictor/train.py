import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import CompressedImageDataset

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

# Model, dataset, etc.
model = UNet(in_channels=4, out_channels=3).to(device)
model.load_state_dict(torch.load("saved_models/v8_model_latest_0.0005_qr.pth", map_location=device))
dataset = CompressedImageDataset("dataset/train")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training
epochs = 200
last_best_saved_avg_loss = float('inf')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(dataloader)
    print(f"✅ Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

    if avg_loss < last_best_saved_avg_loss:
        last_best_saved_avg_loss = avg_loss
        model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}_{avg_loss:.4f}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved at: {model_path}")

# (Optional) Always save latest checkpoint
torch.save(model.state_dict(), os.path.join(save_dir, f"model_latest_{avg_loss:.4f}.pth"))