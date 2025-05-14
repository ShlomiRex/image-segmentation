import torch
import torch.nn as nn

from unet import DownSampling, UpSampling, DoubleConvolution

class DynamicUNet(nn.Module):
    def __init__(self, enc_dec_num_layers, in_channels, num_classes):
        super().__init__()

        if enc_dec_num_layers < 1:
            raise ValueError("The number of encoder and decoder layers in the UNet must be greater than 1.")

        self.dconv1 = DownSampling(in_channels, 64)
        self.dconv2 = DownSampling(64, 128)
        self.dconv3 = DownSampling(128, 256)
        self.dconv4 = DownSampling(256, 512)

        self.bottle_neck = DoubleConvolution(512, 1024)

        self.uconv1 = UpSampling(1024, 512)
        self.uconv2 = UpSampling(512, 256)
        self.uconv3 = UpSampling(256, 128)
        self.uconv4 = UpSampling(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        d1,p1 = self.dconv1(x)
        d2,p2 = self.dconv2(p1)
        d3,p3 = self.dconv3(p2)
        d4,p4 = self.dconv4(p3)

        b = self.bottle_neck(p4)
        
        u1 = self.uconv1(b, d4)
        u2 = self.uconv2(u1, d3)
        u3 = self.uconv3(u2, d2)
        u4 = self.uconv4(u3, d1)

        out = self.out(u4)
        return out