import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from cityscapes_dataset import CityscapesDataset  # your custom dataset
from unet import UNet                              # your UNet model
from tqdm import tqdm
import os
from matplotlib import pyplot as plt

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 19
IN_CHANNELS = 3
BATCH_SIZE = 4
NUM_EPOCHS = 25
LR = 1e-4
IMG_SIZE = (96, 256) # (Height, Width)
DATASET_PATH = r"C:\Users\Shlomi\.cache\kagglehub\datasets\shuvoalok\cityscapes\versions\2"
SAVE_MODEL = True
MODEL_PATH = "unet_cityscapes.pth"

# -------------------------------
# Transformations
# -------------------------------
image_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

# Labels will be processed inside CityscapesDataset with label_map
label_map = {
    (128, 64,128): 0,  # road
    (244, 35,232): 1,  # sidewalk
    (70, 70, 70): 2,   # building
    (102,102,156): 3,  # wall
    (190,153,153): 4,  # fence
    (153,153,153): 5,  # pole
    (250,170, 30): 6,  # traffic light
    (220,220,  0): 7,  # traffic sign
    (107,142, 35): 8,  # vegetation
    (152,251,152): 9,  # terrain
    (70,130,180): 10,  # sky
    (220, 20, 60): 11, # person
    (255,  0,  0): 12, # rider
    (0,  0,142): 13,   # car
    (0,  0, 70): 14,   # truck
    (0, 60,100): 15,   # bus
    (0, 80,100): 16,   # train
    (0,  0,230): 17,   # motorcycle
    (119,11, 32): 18,  # bicycle
}

# -------------------------------
# Dataset & Dataloader
# -------------------------------
train_dataset = CityscapesDataset(
    root_dir=DATASET_PATH,
    split='train',
    transform=image_transform,
    label_map=label_map
)

val_dataset = CityscapesDataset(
    root_dir=DATASET_PATH,
    split='val',
    transform=image_transform,
    label_map=label_map
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# Training Loop
# -------------------------------
def train():
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Plot the images and masks
            plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Image")
            plt.axis('off')
            plt.show()
            plt.imshow(masks[0].cpu().numpy())
            plt.title("Mask")
            plt.axis('off')
            plt.show()

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)  # [B, C, H, W]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    if SAVE_MODEL:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train()
