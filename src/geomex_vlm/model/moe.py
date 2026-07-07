from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardExpert(nn.Module):
    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GeoMemoryRouter(nn.Module):
    """Router conditioned on token features, task embeddings, and CSGM context."""

    def __init__(self, dim: int, num_experts: int):
        super().__init__()
        self.proj = nn.Linear(dim * 3, num_experts)

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor, geo_context: torch.Tensor) -> torch.Tensor:
        if task_emb.dim() == 2:
            task_emb = task_emb.unsqueeze(1).expand_as(x)
        if geo_context.dim() == 2:
            geo_context = geo_context.unsqueeze(1).expand_as(x)
        return F.softmax(self.proj(torch.cat([x, task_emb, geo_context], dim=-1)), dim=-1)


class SparseMoE(nn.Module):
    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = GeoMemoryRouter(dim, num_experts)
        self.experts = nn.ModuleList([FeedForwardExpert(dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor, task_emb: torch.Tensor, geo_context: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        probs = self.router(x, task_emb, geo_context)
        top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, x.shape[-1])
        flat_out = out.reshape(-1, out.shape[-1])
        flat_idx = top_idx.reshape(-1, self.top_k)
        flat_w = top_probs.reshape(-1, self.top_k)
        for k in range(self.top_k):
            idx_k = flat_idx[:, k]
            w_k = flat_w[:, k]
            for expert_id, expert in enumerate(self.experts):
                mask = idx_k == expert_id
                if mask.any():
                    flat_out[mask] += expert(flat_x[mask]) * w_k[mask].unsqueeze(-1)
        aux = {"router_probs": probs, "top_idx": top_idx, "top_probs": top_probs}
        return out, aux
