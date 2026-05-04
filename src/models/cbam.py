"""
CBAM — Convolutional Block Attention Module
=============================================
Modulo di attenzione che raffina le feature tra encoder e decoder nella U-Net.

REF: Paper Section 3.3.2:
     "The attention module employs the Convolutional Block Attention Module
      (CBAM) to connect the encoder and decoder, refining features effectively."

CBAM originale: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
     Applica attenzione sui canali + attenzione spaziale in sequenza.

In Radar-Mamba: usato come "bottleneck" tra encoder e decoder nella U-Net
per raffinare le feature prima di passarle al decoder.

Input/Output: [B, C, H, W] (stessa shape).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Attenzione sui canali: impara QUALE canale è importante.

    Usa average pooling + max pooling → shared MLP → sigmoid.
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        mid = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Average-pool e max-pool globali
        avg = F.adaptive_avg_pool2d(x, 1).view(B, C)  # (B, C)
        mx  = F.adaptive_max_pool2d(x, 1).view(B, C)  # (B, C)
        # Shared MLP su entrambi
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C)
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Attenzione spaziale: impara DOVE le feature sono importanti.

    Usa average e max pooling sui canali → conv 7×7 → sigmoid.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel_size deve essere 3 o 7"
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        mx  = x.max(dim=1, keepdim=True).values   # (B, 1, H, W)
        cat = torch.cat([avg, mx], dim=1)   # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(cat))  # (B, 1, H, W)
        return x * attn


class CBAM(nn.Module):
    """
    CBAM: ChannelAttention → SpatialAttention in sequenza.

    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_attn  = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attn  = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applica prima channel attention poi spatial attention.
        """
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x
