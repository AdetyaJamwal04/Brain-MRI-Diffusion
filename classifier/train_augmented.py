import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights

from utils.combined_dataset import CombinedDataset

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# Dataset
# =========================
train_dataset = CombinedDataset(
    real_dir="data/training",
    synthetic_dir="synthetic",
    limit_per_class=30   # controlled synthetic usage
)

test_dataset = CombinedDataset(
    real_dir="data/testing",
    synthetic_dir="synthetic",
    limit_per_class=0   # IMPORTANT: no synthetic in test
)

# =========================
# DataLoaders
# =========================
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# =========================
# Model
# =========================
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Adjust for grayscale input
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

# Add dropout for regularization
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(512, 4)
)

model = model.to(device)

# =========================
# Loss + Optimizer
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# =========================
# Training Loop
# =========================
epochs = 10
best_acc = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)

    # =========================
    # Evaluation
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")

    # =========================
    # Save Best Model
    # =========================
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_classifier_controlled_Augmentation.pth")
        print(f"✅ New best model saved with accuracy: {best_acc:.2f}%")

print("\n🔥 Training Complete")
print(f"🏆 Best Accuracy Achieved: {best_acc:.2f}%")