import os
import torch.nn as nn
import torch.nn.functional as Functional


# Frequency decomposition function
def laplacian_decompose(x, pool_size=2):
    """
    Decomposes the input image into low- and high-frequency components.

    Steps:
        - Downsample x using average pooling to get a low-frequency approximation.
        - Upsample the low-frequency approximation back to the original size.
        - The high-frequency component is the difference betweem x and the upsampled low-frequency.
    :param x: Input image (tensor)
    :param pool_size: 2
    :return: low and high frequency components.
    """
    # Downsample to capture low frequency.
    low = Functional.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)

    # Upsample low frequency to match the original size.
    low_up = Functional.interpolate(low, size=x.size()[2:], mode="bilinear", align_corners=True)

    # High-frequency details are what remains after subtracting the low-frequency part.
    high = x - low_up
    return low, high


# Low Frequency Processing Module
class LowFrequencyModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=3):
        """
        Lightweight module to process low-frequency components (global illumination.
        Inspired by the Low-Resolution Shadow Removal Architecture (LSRNet) from SHARDS.
        """
        super(LowFrequencyModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Process through a few conv layers with ReLU activation.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# High Frequency Processing Module
class HighFrequencyModule(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=3):
        """
        Module to restore high-resolution details (e.g. text edges, fine textures).
        Uses dilated convolutions to increase receptive field without extra cost.
        Inspired by DRNet (SHARDS) and the High-Frequency restoration block in FSNet (HRDSR).
        """
        super(HighFrequencyModule, self).__init__()
        # Two dilated convolutional layers to capture context
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

