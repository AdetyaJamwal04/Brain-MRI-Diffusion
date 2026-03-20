import torch
import matplotlib.pyplot as plt
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.model import SimpleUNet

# 🔴 IMPORTANT: Fix cuDNN crash
torch.backends.cudnn.enabled = False

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load model
model = SimpleUNet().to(device)
model.load_state_dict(torch.load(r"diffusion\diffusion_model_epoch_10.pth", map_location=device))
model.eval()

# Diffusion parameters (reduced for your GPU)
timesteps = 200

beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
alpha = 1. - beta
alpha_hat = torch.cumprod(alpha, dim=0)

# Sampling function
def sample(model):
    # Start from random noise
    x = torch.randn(1, 1, 128, 128).to(device)

    for t in reversed(range(timesteps)):
        alpha_t = alpha[t]
        alpha_hat_t = alpha_hat[t]
        beta_t = beta[t]

        # Predict noise
        pred_noise = model(x)

        # Add noise (except last step)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # Reverse diffusion step
        x = (1 / torch.sqrt(alpha_t)) * (
            x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
        ) + torch.sqrt(beta_t) * noise

    return x


# 🟢 Clear memory (important for small GPU)
torch.cuda.empty_cache()

# 🟢 Disable gradients (important)
with torch.no_grad():
    sampled_img = sample(model)

# Convert to displayable image
img = sampled_img.squeeze().cpu().numpy()

# Convert from [-1,1] → [0,1]
img = (img + 1) / 2
img = img.clip(0, 1)

# Display
plt.imshow(img, cmap='gray')
plt.title("Generated MRI (Diffusion)")
plt.axis('off')
plt.show()