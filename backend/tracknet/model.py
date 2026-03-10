"""
TrackNetV3 Model Architecture

U-Net based encoder-decoder for shuttlecock heatmap prediction,
plus InpaintNet for trajectory gap filling.

Layer names match the original checkpoint exactly.
Ported from https://github.com/qaz812345/TrackNetV3 (MIT License)
"""

import torch
import torch.nn as nn


# =============================================================================
# 2D BUILDING BLOCKS (TrackNet)
# =============================================================================

class ConvBnRelu(nn.Module):
    """Single Conv2d(3x3) + BatchNorm + ReLU. Named as .conv and .bn."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Double2DConv(nn.Module):
    """Two consecutive ConvBnRelu blocks, named conv_1 and conv_2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = ConvBnRelu(in_channels, out_channels)
        self.conv_2 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


class Triple2DConv(nn.Module):
    """Three consecutive ConvBnRelu blocks, named conv_1, conv_2, conv_3."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = ConvBnRelu(in_channels, out_channels)
        self.conv_2 = ConvBnRelu(out_channels, out_channels)
        self.conv_3 = ConvBnRelu(out_channels, out_channels)

    def forward(self, x):
        return self.conv_3(self.conv_2(self.conv_1(x)))


# =============================================================================
# TRACKNET (2D U-Net for heatmap prediction)
# =============================================================================

class TrackNet(nn.Module):
    """
    TrackNetV3 shuttlecock trajectory predictor.

    U-Net encoder-decoder that takes N consecutive frames (optionally with
    background) and predicts heatmaps indicating shuttlecock position.

    Args:
        in_dim: Number of input channels.
                For bg_mode="concat", seq_len=8: (8+1)*3 = 27
        out_dim: Number of output heatmaps (default 8 = seq_len).
    """

    def __init__(self, in_dim: int = 27, out_dim: int = 8):
        super().__init__()

        # Encoder
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder — upsample layers have no parameters (not in state dict)
        self.up_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_block_1 = Triple2DConv(512 + 256, 256)

        self.up_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_block_2 = Double2DConv(256 + 128, 128)

        self.up_3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_block_3 = Double2DConv(128 + 64, 64)

        # Output head — no sigmoid here, applied in forward()
        self.predictor = nn.Conv2d(64, out_dim, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.down_block_1(x)              # (B, 64, H, W)
        e2 = self.down_block_2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.down_block_3(self.pool(e2))  # (B, 256, H/4, W/4)
        bn = self.bottleneck(self.pool(e3))    # (B, 512, H/8, W/8)

        # Decoder with skip connections
        d3 = self.up_1(bn)
        d3 = torch.cat([d3, e3], dim=1)        # (B, 768, H/4, W/4)
        d3 = self.up_block_1(d3)               # (B, 256, H/4, W/4)

        d2 = self.up_2(d3)
        d2 = torch.cat([d2, e2], dim=1)        # (B, 384, H/2, W/2)
        d2 = self.up_block_2(d2)               # (B, 128, H/2, W/2)

        d1 = self.up_3(d2)
        d1 = torch.cat([d1, e1], dim=1)        # (B, 192, H, W)
        d1 = self.up_block_3(d1)               # (B, 64, H, W)

        return torch.sigmoid(self.predictor(d1))  # (B, out_dim, H, W)


# =============================================================================
# 1D BUILDING BLOCKS (InpaintNet)
# =============================================================================

class Conv1DLeaky(nn.Module):
    """Conv1d(3) + LeakyReLU. Named as .conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Double1DConv(nn.Module):
    """Two consecutive Conv1DLeaky blocks, named conv_1 and conv_2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_1 = Conv1DLeaky(in_channels, out_channels)
        self.conv_2 = Conv1DLeaky(out_channels, out_channels)

    def forward(self, x):
        return self.conv_2(self.conv_1(x))


# =============================================================================
# INPAINTNET (1D U-Net for trajectory gap filling)
# =============================================================================

class InpaintNet(nn.Module):
    """
    Trajectory inpainting network.

    1D U-Net that takes shuttlecock coordinate sequences (x, y, visibility)
    and fills in gaps where the shuttle was not detected.

    Input:  (B, 3, L) — x, y coordinates + visibility mask
    Output: (B, 2, L) — predicted x, y coordinates (normalized 0-1)

    Note: The original code has a typo "buttleneck" — we match it exactly
    so the checkpoint loads correctly.
    """

    def __init__(self):
        super().__init__()

        # Encoder — single conv per level
        self.down_1 = Conv1DLeaky(3, 32)
        self.down_2 = Conv1DLeaky(32, 64)
        self.down_3 = Conv1DLeaky(64, 128)

        # Note: typo "buttleneck" matches the original checkpoint
        self.buttleneck = Double1DConv(128, 256)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Decoder — single conv per level (after skip concatenation)
        # up_1 takes concat(256, 128) = 384 → 128
        self.up_1 = Conv1DLeaky(256 + 128, 128)
        # up_2 takes concat(128, 64) = 192 → 64
        self.up_2 = Conv1DLeaky(128 + 64, 64)
        # up_3 takes concat(64, 32) = 96 → 32
        self.up_3 = Conv1DLeaky(64 + 32, 32)

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        # Output head
        self.predictor = nn.Conv1d(32, 2, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        e1 = self.down_1(x)                    # (B, 32, L)
        e2 = self.down_2(self.pool(e1))        # (B, 64, L/2)
        e3 = self.down_3(self.pool(e2))        # (B, 128, L/4)
        bn = self.buttleneck(self.pool(e3))    # (B, 256, L/8)

        # Decoder with skip connections
        d3 = self.upsample(bn)
        if d3.shape[2] != e3.shape[2]:
            d3 = d3[:, :, :e3.shape[2]]
        d3 = torch.cat([d3, e3], dim=1)        # (B, 384, L/4)
        d3 = self.up_1(d3)                     # (B, 128, L/4)

        d2 = self.upsample(d3)
        if d2.shape[2] != e2.shape[2]:
            d2 = d2[:, :, :e2.shape[2]]
        d2 = torch.cat([d2, e2], dim=1)        # (B, 192, L/2)
        d2 = self.up_2(d2)                     # (B, 64, L/2)

        d1 = self.upsample(d2)
        if d1.shape[2] != e1.shape[2]:
            d1 = d1[:, :, :e1.shape[2]]
        d1 = torch.cat([d1, e1], dim=1)        # (B, 96, L)
        d1 = self.up_3(d1)                     # (B, 32, L)

        return torch.sigmoid(self.predictor(d1))  # (B, 2, L)
