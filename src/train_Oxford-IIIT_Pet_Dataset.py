import mlflow.data.dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torchvision.transforms import functional as TF
from unet import UNet  # Your U-Net model
from tqdm import tqdm
import os
import random
import numpy as np
from PIL import Image
import mlflow
import mlflow.pytorch
import psutil

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_EPOCHS = 20
LR = 1e-4
IMG_SIZE = (128, 128)
SAVE_MODEL = True
MODEL_PATH = "unet_oxford_pet.pth"

print(f"Using device: {DEVICE}")

mlflow.set_experiment("UNet-Oxford-Pet")  # Creates one if doesn't exist
# mlflow.pytorch.autolog()
mlflow.enable_system_metrics_logging()

# -------------------------------
# Transformations
# -------------------------------

# Joint transform for image + mask
class PetSegTransform:
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=Image.NEAREST)

        # # Random horizontal flip
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # To tensor
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Convert to binary mask: pet = 1, background = 0
        mask = (mask > 1).long()

        return image, mask

# -------------------------------
# Dataset Wrapper
# -------------------------------
class OxfordPetSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None):
        self.dataset = OxfordIIITPet(
            root=root,
            download=True,
            target_types="segmentation"
        )
        self.transform = transform
        self.indices = list(range(len(self.dataset)))
        split_point = int(0.9 * len(self.indices))
        if split == "train":
            self.indices = self.indices[:split_point]
        else:
            self.indices = self.indices[split_point:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.dataset[self.indices[idx]]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = UNet(in_channels=3, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# DataLoader
# -------------------------------
train_dataset = OxfordPetSegDataset(root="./data", split="train", transform=PetSegTransform())
val_dataset   = OxfordPetSegDataset(root="./data", split="val", transform=PetSegTransform())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -------------------------------
# Mlflow metrics
# -------------------------------
def compute_iou(preds, labels, num_classes=2):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        if union == 0:
            ious.append(float('nan'))  # Avoid division by zero
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)

def log_model_params(model):
    mlflow.log_param("model_type", "UNet")
    mlflow.log_param("num_epochs", NUM_EPOCHS)
    mlflow.log_param("input_shape", str(IMG_SIZE))
    mlflow.log_param("num_classes", 2)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("learning_rate", LR)
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("loss_function", "CrossEntropyLoss")

    # Calculate and log number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_param("num_params", num_params)
    print(f"Number of parameters in the model: {num_params:,}")
    mlflow.log_param("device", str(DEVICE))

def log_epoch_metrics(epoch, train_loss, val_loss, val_iou, avg_loss):
    mlflow.log_metric("train_loss", train_loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
    mlflow.log_metric("val_iou", val_iou, step=epoch)
    mlflow.log_metric("avg_loss", (train_loss + val_loss) / 2, step=epoch)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")

def log_system_metrics(epoch):
    # Get CPU usage as a percentage
    cpu_percent = psutil.cpu_percent(interval=1)
    
    # Get available memory in GB
    memory = psutil.virtual_memory()
    memory_used = memory.used / (1024 ** 3)  # in GB
    memory_total = memory.total / (1024 ** 3)  # in GB
    memory_percent = memory.percent

    # Get GPU metrics (if using NVIDIA GPU with `nvidia-smi`)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.free,memory.total', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        gpu_usage, gpu_free_memory, gpu_total_memory = result.stdout.strip().split("\n")[0].split(", ")
        gpu_usage = float(gpu_usage)
        gpu_free_memory = float(gpu_free_memory) / 1024  # in GB
        gpu_total_memory = float(gpu_total_memory) / 1024  # in GB
    except Exception as e:
        gpu_usage = gpu_free_memory = gpu_total_memory = None  # If GPU metrics can't be retrieved
    
    # Log system metrics to MLflow
    mlflow.log_metric("cpu_percent", cpu_percent, step=epoch)
    mlflow.log_metric("memory_used_gb", memory_used, step=epoch)
    mlflow.log_metric("memory_total_gb", memory_total, step=epoch)
    mlflow.log_metric("memory_percent", memory_percent, step=epoch)
    if gpu_usage is not None:
        mlflow.log_metric("gpu_usage_percent", gpu_usage, step=epoch)
        mlflow.log_metric("gpu_free_memory_gb", gpu_free_memory, step=epoch)
        mlflow.log_metric("gpu_total_memory_gb", gpu_total_memory, step=epoch)

def log_images_masks_predictions(model, epoch):
    # Log 3 images and masks to mlflow
    images, masks = next(iter(train_loader))  # Get first batch
    images = images.to(DEVICE)
    predictions = model(images)
    for i in range(3):
        image = images[i].cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        image = (image * 255).astype(np.uint8)  # Convert to uint8
        
        mask = masks[i].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask

        prediction = torch.argmax(predictions[i], dim=0).cpu().numpy()
        prediction = (prediction > 0).astype(np.uint8) * 255  # Convert to binary mask

        mlflow.log_image(image, f"images/{epoch}/image_{i}.png")
        mlflow.log_image(mask, f"images/{epoch}/mask_{i}.png")
        mlflow.log_image(prediction, f"images/{epoch}/prediction_{i}.png")

# -------------------------------
# Training Loop
# -------------------------------
def train():
    with mlflow.start_run(log_system_metrics=True):
        log_model_params(model)
        for epoch in range(NUM_EPOCHS):
            log_images_masks_predictions(model, epoch)

            model.train()
            epoch_loss = 0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            train_loss = epoch_loss / len(train_loader)

            # Validation loop
            model.eval()
            val_loss = 0.0
            val_iou = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(DEVICE)
                    masks = masks.to(DEVICE)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    val_iou += compute_iou(outputs, masks)
            
            val_loss /= len(val_loader)
            val_iou /= len(val_loader)
            avg_loss = epoch_loss / len(train_loader)

            log_epoch_metrics(epoch, train_loss, val_loss, val_iou, avg_loss) # Log train epoch metrics (loss, iou)
            # log_system_metrics(epoch)  # Log system metrics

            if epoch % 5 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch}.pth"
                torch.save(model.state_dict(), checkpoint_path)
                mlflow.log_artifact(checkpoint_path)  # Uploads to MLflow run
            
        if SAVE_MODEL:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")
            mlflow.pytorch.log_model(model, "model")

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    train()
