import torch
import torch.nn as nn
import torch.nn.functional as Functional

from Building_blocks import LowFrequencyModule, HighFrequencyModule, laplacian_decompose


# Hybrid shadow removal network
class HybridShadowRemoval(nn.Module):
    def __init__(self):
        """
        Hybrid network that combines frequency decomposition with separate branches
        for low-frequency deshading (global corrections) and high-frequency detail
        restoration. The outputs of the two branches are fused to produce a final
        high-resolution, shadow-free document image.
        """
        super(HybridShadowRemoval, self).__init__()

        # process low-frequency branch (Global corrections)
        self.low_freq_module = LowFrequencyModule(in_channels=3, hidden_channels=32, out_channels=3)

        # Process high-frequency branch (fine detail restoration)
        self.high_freq_module = HighFrequencyModule(in_channels=3, hidden_channels=32, out_channels=3)

        # Fusion layer to combine the two branches
        # We concatenate along the channel dimension (3+3=6 channels) and reduce to 3.
        self.fusion_conv = nn.Conv2d(6, 3, kernel_size=3, padding=1)

    def forward(self, x):
        """
        :param x: high-resolution input document image (3-channel RGB)
        :return: Output of shadow free document image
        """
        # Decompose the image into low and high frequency components.
        low, high = laplacian_decompose(x, pool_size=2)

        # Process the upsampled low-frequency branch.
        # Note: low is originally at a lower resolution; we upsample it back.
        low_up = Functional.interpolate(low, size=x.size()[2:], mode="bilinear", align_corners=True)
        low_processed = self.low_freq_module(low_up)

        # Process the high-frequency details.
        high_processed = self.high_freq_module(high)

        # Fuse the two processed branches.
        fused = torch.cat([low_processed, high_processed], dim=1)
        output = self.fusion_conv(fused)

        # Using TanH activation to bring output values into [-1, 1] range (if desired)
        output = torch.tanh(output)
        return output
