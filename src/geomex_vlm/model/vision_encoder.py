from __future__ import annotations

from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class ConvPatchEncoder(nn.Module):
    """Lightweight high-resolution patch encoder placeholder.

    Replace this module with a pretrained remote-sensing vision encoder for full experiments.
    """

    def __init__(self, in_channels: int = 3, dim: int = 768, patch_size: int = 16, num_scales: int = 4):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feat = self.blocks(self.patch(x))
        features = [feat]
        for i in range(1, self.num_scales):
            features.append(F.avg_pool2d(feat, kernel_size=2 ** i, stride=2 ** i, ceil_mode=True))
        return features
