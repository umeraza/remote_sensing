from __future__ import annotations

from typing import List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class ClassScaleGeoMemory(nn.Module):
    """Class-Scale Geo-Memory module.

    Stores memory as [class, scale, slot, dim, H_g, W_g], retrieves by class/scale,
    combines it with a mask-conditioned prior, and cross-attends image features to
    the retrieved geospatial context.
    """

    def __init__(self, num_classes: int, num_scales: int, memory_slots: int, dim: int, memory_hw: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = num_scales
        self.dim = dim
        self.memory_hw = memory_hw
        self.bank = nn.Parameter(torch.randn(num_classes, num_scales, memory_slots, dim, memory_hw, memory_hw) * 0.02)
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, dim, 3, padding=1),
        )
        self.cross_attn = nn.ModuleList([nn.MultiheadAttention(dim, num_heads, batch_first=True) for _ in range(num_scales)])
        self.self_proj = nn.ModuleList([nn.Conv2d(dim, dim, 1) for _ in range(num_scales)])
        self.alpha = nn.Parameter(torch.zeros(num_scales))
        self.pool_proj = nn.Linear(dim, dim)
        self.context_proj = nn.Linear(dim, dim)

    def forward(
        self,
        features: List[torch.Tensor],
        class_ids: torch.Tensor,
        scale_ids: Optional[torch.Tensor] = None,
        init_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature scales, got {len(features)}")
        bsz = features[0].shape[0]
        class_ids = class_ids.clamp(0, self.num_classes - 1).long()
        enhanced: List[torch.Tensor] = []
        tokens = []

        for s, feat in enumerate(features):
            _, c, h, w = feat.shape
            if c != self.dim:
                raise ValueError(f"Feature scale {s} has dim {c}, expected {self.dim}")
            memory = self.bank[class_ids, s]  # [B, slots, D, Hg, Wg]
            memory = memory.mean(dim=1)       # [B, D, Hg, Wg]
            if init_masks is not None and init_masks[s] is not None:
                mask = F.interpolate(init_masks[s].float(), size=(self.memory_hw, self.memory_hw), mode="bilinear", align_corners=False)
            else:
                mask = torch.zeros(bsz, 1, self.memory_hw, self.memory_hw, device=feat.device, dtype=feat.dtype)
            prior = self.mask_encoder(mask)
            geo = memory + prior

            q = self.self_proj[s](feat).flatten(2).transpose(1, 2)       # [B, HW, D]
            kv = geo.flatten(2).transpose(1, 2)                         # [B, HgWg, D]
            attended, _ = self.cross_attn[s](q, kv, kv, need_weights=False)
            attended = attended.transpose(1, 2).reshape(bsz, self.dim, h, w)
            out = feat + self.alpha[s].tanh() * attended
            enhanced.append(out)
            tokens.append(out.mean(dim=(2, 3)))

        gmem_tokens = self.pool_proj(torch.stack(tokens, dim=1))         # [B, num_scales, D]
        geo_context = self.context_proj(gmem_tokens.mean(dim=1))         # [B, D]
        return enhanced, gmem_tokens, geo_context
