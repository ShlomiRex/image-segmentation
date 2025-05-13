from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from dataset_cityscapes import CityscapesDataset

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
    (220, 20, 60):11,  # person
    (255,  0,  0):12,  # rider
    (0,  0,142):13,    # car
    (0,  0, 70):14,    # truck
    (0, 60,100):15,    # bus
    (0, 80,100):16,    # train
    (0,  0,230):17,    # motorcycle
    (119,11, 32):18,   # bicycle
}


transform = transforms.Compose([
    transforms.Resize((256, 512)),
    transforms.ToTensor(),
])

dataset = CityscapesDataset(
    root_dir=r"C:\Users\Shlomi\.cache\kagglehub\datasets\shuvoalok\cityscapes\versions\2",
    split='train',
    transform=transform,
    label_map=label_map
)

img, label = dataset[0]
print(img.shape)      # [3, 256, 512]
print(label.shape)    # [256, 512] (class index per pixel)

def plot_raw_images(dataset, n=5):
    plt.figure(figsize=(6, n * 3))
    
    for i in range(n):
        # Load original PIL images (not tensors)
        img_path = dataset.image_dir + "/" + dataset.image_names[i]
        label_path = dataset.label_dir + "/" + dataset.label_names[i]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        plt.subplot(n, 2, 2 * i + 1)
        plt.imshow(image)
        plt.title(f"Image {i}")
        plt.axis('off')

        plt.subplot(n, 2, 2 * i + 2)
        plt.imshow(label)
        plt.title(f"Label {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

plot_raw_images(dataset, n=3)
