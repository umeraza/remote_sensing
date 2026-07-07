import torch
from geomex_vlm.model.losses import expert_load_balancing_loss, geo_memory_alignment_loss


def test_expert_load_balancing_loss():
    probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    top_idx = torch.topk(probs, 2, dim=-1).indices
    loss = expert_load_balancing_loss(probs, top_idx, 4)
    assert loss.ndim == 0


def test_geo_memory_alignment_loss():
    geo = torch.randn(3, 8)
    prototypes = torch.randn(5, 8)
    cls = torch.tensor([0, 1, 2])
    loss = geo_memory_alignment_loss(geo, cls, prototypes)
    assert loss.ndim == 0
