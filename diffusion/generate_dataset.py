import torch
import os
import cv2
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.model import SimpleUNet

# Fix cuDNN issue
torch.backends.cudnn.enabled = False

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU:", torch.cuda.get_device_name(0))

# Load model
model = SimpleUNet().to(device)
model.load_state_dict(torch.load(r"diffusion\diffusion_model_epoch_10.pth", map_location=device))
model.eval()

# Diffusion params
timesteps = 200
beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Sampling function
def sample():
    x = torch.randn(1, 1, 128, 128).to(device)

    for t in reversed(range(timesteps)):
        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]
        beta_t = beta[t]

        pred_noise = model(x)

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

    return x


# Create folders
base_path = "synthetic"
classes = ["glioma", "meningioma", "pituitary", "no_tumor"]

for cls in classes:
    os.makedirs(os.path.join(base_path, cls), exist_ok=True)

# Generate images
num_images = 400  # total images

print("Generating synthetic images...")

with torch.no_grad():
    for i in range(num_images):
        img = sample()

        img = img.squeeze().cpu().numpy()
        img = (img + 1) / 2
        img = (img * 255).astype("uint8")

        # Assign class (round-robin)
        cls = classes[i % 4]

        save_path = os.path.join(base_path, cls, f"img_{i}.png")
        cv2.imwrite(save_path, img)

        if i % 50 == 0:
            print(f"Generated {i}/{num_images}")

print("Done!")