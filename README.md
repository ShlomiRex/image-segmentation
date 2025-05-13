# Image Segmentation with PyTorch

My implementation of the [U-Net paper](https://arxiv.org/abs/1505.04597) using PyTorch with explanation.

Task: segment images.

I used a [YouTube tutorial](https://www.youtube.com/watch?v=HS3Q_90hnDg) for guidance.

## Dataset

Download [Cityscapes dataset](https://www.kaggle.com/datasets/shuvoalok/cityscapes):

```
import kagglehub
path = kagglehub.dataset_download("shuvoalok/cityscapes")
print("Path to dataset files:", path)
```

Use this path to load the dataset in your code.
