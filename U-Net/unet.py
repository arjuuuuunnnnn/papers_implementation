import torch
import torch.nn as nn

from downsample import DownSampleBlock
from upsample import UpSampleBlock

class UNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.down1 = DownSampleBlock(1, 64)
        self.down2 = DownSampleBlock(64, 128)
        self.down3 = DownSampleBlock(128, 256)
        self.down4 = DownSampleBlock(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

        self.up4 = UpSampleBlock(1024, 512)
        self.up3 = UpSampleBlock(512, 256)
        self.up2 = UpSampleBlock(256, 128)
        self.up1 = UpSampleBlock(128, 64)

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1, skip1 = self.down1(x)
        x2, skip2 = self.down2(x1)
        x3, skip3 = self.down3(x2)
        x4, skip4 = self.down4(x3)

        x = self.bottleneck(x4)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)

        x = self.final_conv(x)

        return x
