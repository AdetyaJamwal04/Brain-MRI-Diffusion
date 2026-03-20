import cv2
import os
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, root_dir, size=128):
        self.data = []
        self.labels = []
        self.size = size

        self.classes = sorted(os.listdir(root_dir))

        for label, cls in enumerate(self.classes):
            path = os.path.join(root_dir, cls)

            for img in os.listdir(path):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    self.data.append(os.path.join(path, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Corrupted image: {img_path}")

        img = cv2.resize(img, (self.size, self.size))
        img = img / 127.5 - 1.0

        img = torch.tensor(img).float().unsqueeze(0)
        label = torch.tensor(label)

        return img, label