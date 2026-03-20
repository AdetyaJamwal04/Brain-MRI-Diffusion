import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.dataset import MRIDataset
from diffusion.model import SimpleUNet
import numpy as np
from tqdm import tqdm

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Dataset
dataset = MRIDataset("data/training")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = SimpleUNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Diffusion params
timesteps = 300

beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Training loop
epochs = 10

for epoch in range(epochs):
    pbar = tqdm(loader)
    total_loss = 0

    for imgs, _ in pbar:
        imgs = imgs.to(device)

        t = torch.randint(0, timesteps, (imgs.size(0),), device=device)

        noise = torch.randn_like(imgs)

        alpha_hat_t = alpha_hat[t].view(-1, 1, 1, 1)

        noisy_imgs = torch.sqrt(alpha_hat_t) * imgs + torch.sqrt(1 - alpha_hat_t) * noise

        pred_noise = model(noisy_imgs)

        loss = criterion(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_description(f"Loss: {loss.item():.4f}")

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f"diffusion\diffusion_model_epoch_{epoch+1}.pth")
    