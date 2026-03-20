import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import torch
from torch.utils.data import Dataset

class CombinedDataset(Dataset):
    def __init__(self, real_dir, synthetic_dir, size=128, limit_per_class=100):
        self.data = []
        self.labels = []
        self.size = size

        classes = sorted(os.listdir(real_dir))

        for label, cls in enumerate(classes):
            # Real data
            real_path = os.path.join(real_dir, cls)
            for img in os.listdir(real_path):
                if img.endswith(('.jpg', '.png', '.jpeg')):
                    self.data.append(os.path.join(real_path, img))
                    self.labels.append(label)

            # Synthetic data (LIMITED)
            name_map = {
                  "notumor": "no_tumor",
                  "no_tumor": "no_tumor"
                  }
            synth_cls = name_map.get(cls, cls)
            synth_path = os.path.join(synthetic_dir, synth_cls)
            synth_imgs = os.listdir(synth_path)[:limit_per_class]

            for img in synth_imgs:
                if img.endswith('.png'):
                    self.data.append(os.path.join(synth_path, img))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (self.size, self.size))
        img = img / 127.5 - 1.0

        img = torch.tensor(img).float().unsqueeze(0)
        label = torch.tensor(label)

        return img, label