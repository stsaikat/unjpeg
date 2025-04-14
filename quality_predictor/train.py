import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset import JPEGCompressionDataset
from model import JPEGCompressionPredictor

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JPEGCompressionPredictor().to(device)
model.load_state_dict(torch.load("saved_models/v5_model_epoch_300_29.81.pth", map_location=device))
dataset = JPEGCompressionDataset("dataset/train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,  num_workers=8)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1000
last_best_saved_avg_loss = 1000.0

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, qualities in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        qualities = qualities.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, qualities)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    
    if avg_loss < last_best_saved_avg_loss:
        last_best_saved_avg_loss = avg_loss
        torch.save(model.state_dict(), f"checkpoints/model_epoch_{epoch+1}_{avg_loss:.4f}.pth")