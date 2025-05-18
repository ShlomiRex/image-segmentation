import torch
from unet import UNet  # Import your U-Net model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "unet_oxford_pet.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)
NUM_CLASSES = 2

print(f"Using device: {DEVICE}")

# -------------------------------
# Load model
# -------------------------------
model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# -------------------------------
# Load and preprocess image
# -------------------------------
image_to_segment = "data/custom_images/car.png"
image = Image.open(image_to_segment).convert("RGB").resize(IMG_SIZE)
image_np = np.array(image) / 255.0  # Normalize to [0, 1]
image_tensor = torch.tensor(image_np).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

# -------------------------------
# Run model and postprocess
# -------------------------------
with torch.no_grad():
    output = model(image_tensor)  # [1, 2, H, W]
    mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # [H, W]
    binary_mask = (mask > 0.5).astype(np.uint8)  # Binary mask [H, W]

# -------------------------------
# Create masked image
# -------------------------------
masked_image_np = image_np * (binary_mask[:, :, np.newaxis] == 0)

# -------------------------------
# Plot images
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(binary_mask, cmap="gray")
axes[1].set_title("Predicted Mask")
axes[1].axis("off")

axes[2].imshow(masked_image_np)
axes[2].set_title("Masked Image")
axes[2].axis("off")

plt.tight_layout()
plt.show()
