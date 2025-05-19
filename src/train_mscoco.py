import mlflow.data.dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
import torchvision
import PIL
import matplotlib
from matplotlib.patches import Patch

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 1e-4
IMG_SIZE = (128, 128)
SAVE_MODEL = True
MODEL_PATH = "final_unet_model_mscoco.pth"

print(f"Using device: {DEVICE}")

mlflow.set_experiment("UNet-MSCOCO")  # Creates one if doesn't exist
# mlflow.pytorch.autolog()
mlflow.enable_system_metrics_logging()

# -------------------------------
# Transformations
# -------------------------------

# Joint transform for image + mask
class MscocoTransform:
    def __call__(self, image, mask):
        # Resize
        image = TF.resize(image, IMG_SIZE)
        mask = TF.resize(mask, IMG_SIZE, interpolation=Image.NEAREST)

        # To tensor
        image = TF.to_tensor(image)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        # Convert to binary mask: pet = 1, background = 0
        mask = (mask > 1).long()

        return image, mask

# -------------------------------
# Dataset Wrapper
# -------------------------------
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
# Datasets, DataLoader
# -------------------------------
coco = COCO('./data/coco/annotations/instances_train2017.json')
num_classes = len(coco.getCatIds())

# Print category names, ids, and supercategories
cat_ids = coco.getCatIds()
cat_names = coco.loadCats(cat_ids)
for category in cat_names:
    print(category)
print(f"Number of classes in COCO dataset: {num_classes}")
img_ids = coco.getImgIds()

# # Plot image, multiple mask categories, and their segmentation (mask applied to the image)
# def plot_image_with_annotations(image_id):
#     # Load image info
#     x = coco.loadImgs(image_id)
#     width, height = x[0]['width'], x[0]['height']
#     image_path = os.path.join('./data/coco/train2017', x[0]['file_name'])
#     image = Image.open(image_path).convert('RGB')
#     image_np = np.array(image)

#     # Load annotations for image
#     annotations = coco.getAnnIds(imgIds=image_id)
#     loaded_annotations = coco.loadAnns(annotations)

#     # Plot individual masks
#     for i, ann in enumerate(loaded_annotations):
#         category_name = coco.loadCats([ann['category_id']])[0]['name']
#         print(f"Category: {category_name}, Segmentation: {ann['segmentation']}")

#         # Convert segmentation to binary mask
#         if isinstance(ann['segmentation'], list):
#             rles = coco_mask.frPyObjects(ann['segmentation'], height, width)
#             rle = coco_mask.merge(rles)
#         elif isinstance(ann['segmentation'], dict):
#             rle = ann['segmentation']
#         else:
#             continue

#         mask = coco_mask.decode(rle)

#         # Apply mask to image
#         masked_image = image_np.copy()
#         masked_image[mask == 0] = 0  # Black out everything except the mask

#         # Plot original image, mask, and masked image
#         fig, axs = plt.subplots(1, 3, figsize=(15, 5))
#         axs[0].imshow(image_np)
#         axs[0].set_title("Original Image")
#         axs[0].axis('off')

#         axs[1].imshow(mask, cmap='gray')
#         axs[1].set_title(f"Mask ({category_name})")
#         axs[1].axis('off')

#         axs[2].imshow(masked_image)
#         axs[2].set_title("Image with Mask Applied")
#         axs[2].axis('off')

#         plt.tight_layout()
#         plt.show()

# # plot_image_with_annotations(9)
# # plot_image_with_annotations(25)
# # plot_image_with_annotations(30)
# # plot_image_with_annotations(34)
# # plot_image_with_annotations(36)

# class CustomMSCOCODataset(torch.utils.data.Dataset):
#     def __init__(self, ann_file, image_dir='./data/coco/train2017', img_size=(224, 224)):
#         self.coco = COCO(ann_file)
#         self.img_ids = self.coco.getImgIds()
#         self.image_dir = image_dir
#         self.img_size = img_size
#         self.resize = transforms.Resize(self.img_size)
#         self.to_tensor = transforms.ToTensor()
#         print(f"Number of annotations in coco: {len(self.coco.anns)}")

#     def __len__(self):
#         return len(self.img_ids)

#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
#         img_info = self.coco.loadImgs(img_id)[0]
#         img_path = os.path.join(self.image_dir, img_info['file_name'])
#         image = Image.open(img_path).convert('RGB')
#         image = self.resize(image)
#         image_tensor = self.to_tensor(image)

#         # Load annotations
#         ann_ids = self.coco.getAnnIds(imgIds=img_id)
#         anns = self.coco.loadAnns(ann_ids)

#         masks = []
#         category_ids = []
#         category_names = []

#         for ann in anns:
#             if 'segmentation' in ann:
#                 mask = self.coco.annToMask(ann)
#                 mask = torch.tensor(mask, dtype=torch.uint8)
#                 masks.append(mask)

#                 category_ids.append(ann['category_id'])

#                 category_name = self.coco.loadCats([ann['category_id']])[0]['name']
#                 category_names.append(category_name)

#         return image_tensor, masks, category_ids, category_names

# def custom_collate_fn(batch):
#     images = []
#     all_masks = []
#     all_category_ids = []
#     all_category_names = []

#     for image, masks, category_ids, category_names in batch:
#         images.append(image)
#         all_masks.append(masks)         # list of [H, W] tensors (variable per image)
#         all_category_ids.append(category_ids)
#         all_category_names.append(category_names)

#     return images, all_masks, all_category_ids, all_category_names

# colormap = matplotlib.cm.get_cmap('tab20b', num_classes)
# # Create a dict {category_name: RGB tuple}
# category_colors = {
#     cat_names[i]: tuple((np.array(colormap(i)[:3]) * 255).astype(int))
#     for i in range(num_classes)
# }

# # Example output
# for name, color in category_colors.items():
#     print(f"{name}: {color}")

class CustomMSCOCODataset(torch.utils.data.Dataset):
    def __init__(self, ann_file):
        self.coco = COCO(ann_file)
        self.image_dir = './data/coco/train2017'
         # Define transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),  # Converts to [0, 1] float tensor
        ])

    def __len__(self):
        return len(self.coco.imgs)

    def __getitem__(self, idx):
        img_id = img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info['file_name'])

        # Load image
        image = Image.open(img_path).convert('RGB')
        width, height = img_info['width'], img_info['height']

        # Initialize mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Load annotations and draw masks
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        for ann in anns:
            if 'segmentation' not in ann:
                continue

            cat_id = ann['category_id']  # 1 to 90, COCO IDs (may skip some numbers)
            coco_cat_index = self.coco.getCatIds().index(cat_id) + 1  # Index from 1 to num_classes

            # Handle RLE or polygon
            if isinstance(ann['segmentation'], list):
                rles = coco_mask.frPyObjects(ann['segmentation'], height, width)
                rle = coco_mask.merge(rles)
            elif isinstance(ann['segmentation'], dict):
                rle = ann['segmentation']
            else:
                continue

            binary_mask = coco_mask.decode(rle)
            mask[binary_mask > 0] = coco_cat_index  # Label the pixels with class index

        # Resize image and mask
        image = self.image_transform(image)
        mask = Image.fromarray(mask)
        mask = transforms.Resize(IMG_SIZE, interpolation=Image.NEAREST)(mask)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask

def visualize_image_and_mask(image, mask, coco):
    # Convert image from tensor to NumPy
    image_np = image.permute(1, 2, 0).numpy()

    # Get unique labels in mask
    unique_classes = torch.unique(mask).tolist()

    # Get COCO category ID to name mapping
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    id_to_name = {cat['id']: cat['name'] for cat in cats}

    # Assign random colors to each class
    colors = {cls: [random.randint(0, 255) for _ in range(3)] for cls in unique_classes if cls != 0}
    colors[0] = [0, 0, 0]  # Background is black

    # Create color mask
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls in unique_classes:
        color = colors[cls]
        mask_color[mask == cls] = color

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(image_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(mask_color)
    axs[1].set_title("Segmentation Mask")
    axs[1].axis('off')

    # Create legend
    legend_elements = []
    for cls in unique_classes:
        if cls == 0:
            name = 'Background'
        else:
            name = id_to_name.get(cls, f"Class {cls}")
        legend_elements.append(Patch(facecolor=np.array(colors[cls])/255.0, label=name))

    fig.legend(handles=legend_elements, loc='lower center', ncol=5)
    plt.tight_layout()
    plt.show()

train_dataset = CustomMSCOCODataset(ann_file="./data/coco/annotations/instances_train2017.json")
val_dataset = CustomMSCOCODataset(ann_file="./data/coco/annotations/instances_val2017.json")

for image, mask in train_dataset:
    visualize_image_and_mask(image, mask, coco)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, collate_fn=custom_collate_fn)

# Print dataset sizes
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# -------------------------------
# Model, Loss, Optimizer
# -------------------------------
model = UNet(in_channels=3, num_classes=num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

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
