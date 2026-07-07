from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn.functional as F


def autoregressive_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=ignore_index)


def expert_load_balancing_loss(router_probs: torch.Tensor, top_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    one_hot = F.one_hot(top_idx, num_classes=num_experts).float().sum(dim=-2)
    fraction = one_hot.mean(dim=(0, 1))
    gate = router_probs.mean(dim=(0, 1))
    return num_experts * torch.sum(fraction * gate)


def geo_memory_alignment_loss(geo_context: torch.Tensor, class_ids: torch.Tensor, prototypes: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    geo_context = F.normalize(geo_context, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    logits = geo_context @ prototypes.t() / temperature
    return F.cross_entropy(logits, class_ids.long())


def cross_format_grounding_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return autoregressive_loss(logits, labels, ignore_index=ignore_index)


def total_loss(losses: Dict[str, torch.Tensor], lambda_elb: float = 0.01, lambda_gma: float = 0.10, lambda_cfg: float = 0.20) -> torch.Tensor:
    total = losses.get("ar", torch.tensor(0.0, device=next(iter(losses.values())).device))
    if "elb" in losses:
        total = total + lambda_elb * losses["elb"]
    if "gma" in losses:
        total = total + lambda_gma * losses["gma"]
    if "cfg" in losses:
        total = total + lambda_cfg * losses["cfg"]
    return total
