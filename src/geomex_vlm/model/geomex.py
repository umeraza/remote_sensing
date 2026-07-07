from __future__ import annotations

from typing import Dict, Optional
import torch
from torch import nn

from .vision_encoder import ConvPatchEncoder
from .csgm import ClassScaleGeoMemory
from .moe import SparseMoE


class GeoMExVLM(nn.Module):
    """Minimal GeoMEx-VLM scaffold.

    This module implements the paper-specific interfaces. For full-scale experiments,
    replace `decoder` with a pretrained decoder-only LLM and connect tokenizer logits.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vision = ConvPatchEncoder(dim=cfg.hidden_dim, patch_size=cfg.patch_size, num_scales=cfg.num_scales)
        self.csgm = ClassScaleGeoMemory(cfg.num_classes, cfg.num_scales, cfg.memory_slots, cfg.hidden_dim, cfg.memory_hw, cfg.num_heads)
        self.task_embed = nn.Embedding(16, cfg.hidden_dim)
        self.moe = SparseMoE(cfg.hidden_dim, cfg.num_experts, cfg.top_k)
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg.hidden_dim, nhead=cfg.num_heads, batch_first=True)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)

    def forward(
        self,
        images: torch.Tensor,
        task_ids: torch.Tensor,
        class_ids: torch.Tensor,
        init_masks: Optional[list[torch.Tensor]] = None,
        enable_csgm: bool = True,
        enable_moe: bool = True,
    ) -> Dict[str, torch.Tensor]:
        features = self.vision(images)
        if enable_csgm:
            features, gmem_tokens, geo_context = self.csgm(features, class_ids, init_masks=init_masks)
        else:
            pooled = torch.stack([f.mean(dim=(2, 3)) for f in features], dim=1)
            gmem_tokens = pooled
            geo_context = pooled.mean(dim=1)
        visual_tokens = features[0].flatten(2).transpose(1, 2)
        x = torch.cat([visual_tokens, gmem_tokens], dim=1)
        task = self.task_embed(task_ids.long())
        if enable_moe:
            moe_out, aux = self.moe(x, task, geo_context)
            x = x + moe_out
        else:
            aux = {}
        hidden = self.decoder(x)
        logits = self.lm_head(hidden)
        return {"hidden": hidden, "logits": logits, "geo_context": geo_context, **aux}
