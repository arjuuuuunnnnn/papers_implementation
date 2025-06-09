import torch
import torch.nn as nn

class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels=input_channels, out_channels=output_channels, kernel_size=2, stride=2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2*output_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channels, out_channels=output_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_connection):
        output = self.up_conv(x)

        #calculate cropping for skip connection (skip connections are larger as they fgot no padding in encoder)
        diff_h = skip_connection.size()[2] - output.size()[2]
        diff_w = skip_connection.size()[3] - output.size()[3]

        #center crop for skip connection(symmetrically crop the height and width on both the sides)
        skip_connection = skip_connection[
            :,
            :,
            diff_h // 2 : skip_connection.size()[2] - diff_h // 2,
            diff_w // 2 : skip_connection.size()[3] - diff_h // 2
        ]

        output = torch.cat([x, skip_connection], dim=1)
        output = self.conv(output)
        return output

