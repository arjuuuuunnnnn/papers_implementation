import torch
import torch.nn as nn

class DownSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        output = self.conv(x)
        skip_connection = output
        output = self.pool(output)
        return output, skip_connection
