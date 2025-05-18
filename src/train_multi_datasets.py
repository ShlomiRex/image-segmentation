import mlflow.data.dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from torchvision.datasets import VOCSegmentation
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
import time
import matplotlib.pyplot as plt
from io import BytesIO
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import ConcatDataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask


# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 1e-4
IMG_SIZE = (128, 128)
SAVE_MODEL = True
MODEL_PATH = "final_unet_model.pth"

print(f"Using device: {DEVICE}")

mlflow.set_experiment("UNet-Multi-Datasets")  # Creates one if doesn't exist
# mlflow.pytorch.autolog()
mlflow.enable_system_metrics_logging()

# -------------------------------
# Transformations
# -------------------------------

# Joint transform for image + mask
class CommonTransform:
    """
    invert: invert the mask? black to white and vice versa
    """
    def __init__(self, invert=False):
        self.invert = invert
    
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=Image.NEAREST)

        # To tensor
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Convert to binary mask: pet = 1, background = 0
        if not self.invert:
            mask = (mask > 1).long()
        else:
            mask = (mask == 1).long()

        return image, mask

# -------------------------------
# Dataset Wrapper
# -------------------------------
class OxfordPetSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, split="train", transform=None):
        self.dataset = OxfordIIITPet(
            root=root,
            download=True,
            target_types="segmentation",
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

class VOCSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, year="2012", image_set="train", transform=None):
        self.dataset = VOCSegmentation(
            root=root,
            year=year,
            image_set=image_set,
            download=True,
            transforms=None  # We'll use our own
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if self.transform:
            img, mask = self.transform(img, mask)
        return img, mask

class CocoSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.coco = COCO(ann_file)
        self.root = root
        self.transform = transform
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)

        # Load image
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        # Create segmentation mask
        mask = np.zeros((img.height, img.width), dtype=np.uint8)
        for ann in anns:
            if ann.get("iscrowd", 0):
                rle = coco_mask.frPyObjects(ann['segmentation'], img.height, img.width)
                m = coco_mask.decode(rle)
                m = m if len(m.shape) == 2 else m.sum(axis=2)
            else:
                m = coco.annToMask(ann)
            mask[m > 0] = ann['category_id']

        mask = Image.fromarray(mask)

        if self.transform:
            img = self.transform(img)
            mask = torch.from_numpy(np.array(mask)).long()

        return img, mask

    def __len__(self):
        return len(self.ids)

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = UNet(in_channels=3, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------------
# Datasets, DataLoader
# -------------------------------

# Plot 5 masks from each dataset
def plot_masks(dataset, title):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i in range(5):
        img, mask = dataset[i]
        axes[i].imshow(mask.squeeze(0).cpu().numpy(), cmap="gray")
        axes[i].axis("off")
        axes[i].set_title(f"{title} Mask {i+1}")
    plt.show()

# Oxford IIT Pet Dataset
print("Loading Oxford IIT Pet Dataset...")
dataset_train_oxford_iit_pet = OxfordPetSegDataset(root="./data", split="train", transform=CommonTransform(invert=True))
dataset_val_oxford_iit_pet   = OxfordPetSegDataset(root="./data", split="val", transform=CommonTransform(invert=True))
# plot_masks(dataset_train_oxford_iit_pet, "Oxford IIT Pet")

# VOC Segmentation Dataset
print("Loading VOC Segmentation Dataset...")
dataset_train_voc = VOCSegDataset(root="./data", year="2012", image_set="train", transform=CommonTransform())
dataset_val_voc = VOCSegDataset(root="./data", year="2012", image_set="val", transform=CommonTransform())

# Combine datasets
train_dataset = ConcatDataset([dataset_train_oxford_iit_pet, dataset_train_voc])
val_dataset = ConcatDataset([dataset_val_oxford_iit_pet, dataset_val_voc])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

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

def log_images_masks_predictions(model, train_loader, epoch, device):
    model.eval()
    images, masks = next(iter(train_loader))  # Get first batch
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    column_titles = ["Image", "Mask", "Prediction"]

    for i in range(3):
        image = images[i].cpu().numpy().transpose(1, 2, 0)
        image = (image * 255).astype(np.uint8)

        mask = masks[i].cpu().numpy()
        mask = (mask > 0).astype(np.uint8) * 255

        prediction = torch.argmax(predictions[i], dim=0).cpu().numpy()
        prediction = (prediction > 0).astype(np.uint8) * 255

        axes[i, 0].imshow(image)
        axes[i, 0].axis("off")
        if i == 0:
            axes[i, 0].set_title(column_titles[0])

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].axis("off")
        if i == 0:
            axes[i, 1].set_title(column_titles[1])

        axes[i, 2].imshow(prediction, cmap="gray")
        axes[i, 2].axis("off")
        if i == 0:
            axes[i, 2].set_title(column_titles[2])

    plt.tight_layout()

    path = f"combined_grid.png"
    fig.savefig(path)
    # Load the image as numpy array
    image = Image.open(path)
    image = np.array(image)
    # Convert to RGB
    image = image[:, :, :3]
    # Convert to uint8
    image = image.astype(np.uint8)

    mlflow.log_image(image, f"images/{epoch}_prediction.png")
    
    plt.close(fig)

# -------------------------------
# Training Loop
# -------------------------------
def train():
    with mlflow.start_run(log_system_metrics=True):
        log_model_params(model)
        for epoch in range(NUM_EPOCHS):
            log_images_masks_predictions(model, train_loader, epoch, DEVICE)

            epoch_time_start = time.time()

            model.train()
            epoch_loss = 0

            total_backward_pass_time = 0
            total_optimizer_step_time = 0
            
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                # Plot masks
                fig, axes = plt.subplots(1, 2, figsize=(8, 4))
                axes[0].imshow(images[0].cpu().numpy().transpose(1, 2, 0))
                axes[0].axis("off")
                axes[0].set_title("Image")
                axes[1].imshow(masks[0].cpu().numpy(), cmap="gray")
                axes[1].axis("off")
                axes[1].set_title("Mask")
                plt.tight_layout()
                plt.show()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)

                start = time.time()
                loss.backward()
                total_backward_pass_time += time.time() - start

                start = time.time()
                optimizer.step()
                epoch_loss += loss.item()
                total_optimizer_step_time += time.time() - start


            # Log time it took for a single epoch
            epoch_time_end = time.time()
            mlflow.log_metric("time/epoch", epoch_time_end - epoch_time_start, step=epoch)

            # Log time metrics for backward pass and optimizer step
            mlflow.log_metric("time/avg_backward_pass", total_backward_pass_time / len(train_loader), step=epoch)
            mlflow.log_metric("time/avg_optimizer_step", total_optimizer_step_time / len(train_loader), step=epoch)

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
