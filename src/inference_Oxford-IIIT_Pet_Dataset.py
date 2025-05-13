import torch
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from unet import UNet  # Import your U-Net model

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "unet_oxford_pet1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (128, 128)
NUM_CLASSES = 2
NUM_SAMPLES = 5

# -------------------------------
# Load model
# -------------------------------
model = UNet(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded successfully!")

# Print model number of parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters in the model: {:,}".format(num_params))

# -------------------------------
# Dataset and Transform
# -------------------------------
class PetSegTransform:
    def __call__(self, image, mask):
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=Image.NEAREST)

        image_tensor = TF.to_tensor(image)
        mask_tensor = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask_tensor = (mask_tensor > 1).long()  # Pet = 1, background = 0

        return image_tensor, mask_tensor, image, mask  # Return PILs for plotting

dataset = OxfordIIITPet(
    root="./data",
    download=True,
    target_types="segmentation"
)

transform = PetSegTransform()

# -------------------------------
# Inference and Visualization
# -------------------------------
def predict_and_plot(idx):
    image, mask = dataset[idx]
    input_tensor, mask_tensor, pil_image, pil_mask = transform(image, mask)
    
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]
    with torch.no_grad():
        output = model(input_tensor)  # [1, 2, H, W]
        prediction = torch.argmax(F.softmax(output, dim=1), dim=1).squeeze(0).cpu().numpy()

    # Convert label to binary mask (optional color mapping)
    mask_tensor = mask_tensor.numpy()

    # Apply prediction mask to image
    image_np = np.array(pil_image.resize(IMG_SIZE))  # shape: (H, W, 3)
    prediction_mask = prediction.astype(bool)        # shape: (H, W)

    masked_image = image_np.copy()
    masked_image[prediction_mask] = 0  # Zero out background

    # Plot original image, ground truth, prediction, masked
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(pil_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(mask_tensor, cmap="gray")
    axs[1].set_title("Ground Truth")

    axs[2].imshow(prediction, cmap="gray")
    axs[2].set_title("Prediction")

    axs[3].imshow(masked_image)
    axs[3].set_title("Masked Image (Prediction)")

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# -------------------------------
# Run Inference on Random Images
# -------------------------------
if __name__ == "__main__":
    indices = np.random.choice(len(dataset), size=NUM_SAMPLES, replace=False)
    for i in indices:
        predict_and_plot(i)
        print(f"Processed image {i}")