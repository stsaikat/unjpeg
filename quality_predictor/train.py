import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset import JPEGCompressionDataset
from model import JPEGCompressionPredictor

# Initialize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = JPEGCompressionPredictor().to(device)
model.load_state_dict(torch.load("saved_models/v3_model_epoch_1000_7.54.pth", map_location=device))
dataset = JPEGCompressionDataset("dataset/train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 1000

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
    
    if (epoch+1) % 100 == 0:
        torch.save(model.state_dict(), f"saved_models/model_epoch_{epoch+1}_{avg_loss:.2f}.pth")