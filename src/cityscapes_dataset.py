import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CityscapesDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None, label_map=None):
        """
        Args:
            root_dir (str): Path to the dataset root.
            split (str): 'train' or 'val'.
            transform (callable, optional): Transform to apply to the input images.
            target_transform (callable, optional): Transform to apply to the label masks.
            label_map (dict, optional): Maps RGB color to integer class index. If None, you must convert externally.
        """
        self.image_dir = os.path.join(root_dir, split, 'img')
        self.label_dir = os.path.join(root_dir, split, 'label')
        self.image_names = sorted(os.listdir(self.image_dir))
        self.label_names = sorted(os.listdir(self.label_dir))
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = label_map

        assert len(self.image_names) == len(self.label_names), "Mismatch between image and label count!"

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        label_path = os.path.join(self.label_dir, self.label_names[idx])

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.label_map:
            label = self.rgb_to_class(label)
        elif self.target_transform:
            label = self.target_transform(label)

        return image, label

    def rgb_to_class(self, label_img):
        """Convert RGB label image to a 2D class index tensor."""
        label_np = torch.ByteTensor(torch.ByteStorage.from_buffer(label_img.tobytes()))
        label_np = label_np.view(label_img.size[1], label_img.size[0], 3)
        label_tensor = torch.zeros((label_img.size[1], label_img.size[0]), dtype=torch.long)

        # Vectorized mapping (slow if done naively; here's a simple loop)
        for i, (color, class_idx) in enumerate(self.label_map.items()):
            mask = (label_np == torch.tensor(color, dtype=torch.uint8).view(1, 1, 3)).all(dim=2)
            label_tensor[mask] = class_idx

        return label_tensor
