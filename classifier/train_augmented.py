import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.combined_dataset import CombinedDataset

from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

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
    limit_per_class=30
)

test_dataset = CombinedDataset(
    real_dir="data/testing",
    synthetic_dir="synthetic",
    limit_per_class=0
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

# Adjust for grayscale
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)

# Add dropout
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
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Accuracy
    accuracy = np.mean(all_preds == all_labels) * 100

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")

    # =========================
    # Classification Report
    # =========================
    print("\n📊 Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["glioma", "meningioma", "pituitary", "no_tumor"]
    ))

    # =========================
    # AUC-ROC
    # =========================
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        print(f"🔥 AUC-ROC: {auc:.4f}")
    except Exception as e:
        print("AUC calculation skipped:", e)

    # =========================
    # Save Best Model
    # =========================
    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "best_classifier_augmented.pth")
        print(f"✅ Best model saved (Accuracy: {best_acc:.2f}%)")

print("\n🏆 Training Complete")
print(f"Best Accuracy Achieved: {best_acc:.2f}%")