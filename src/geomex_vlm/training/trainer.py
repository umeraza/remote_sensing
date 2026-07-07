from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0


class GeoMExTrainer:
    """Minimal trainer interface for Stage-I/Stage-II experiments."""

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.state = TrainState()

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total = 0.0
        steps = 0
        for batch in tqdm(loader, desc=f"epoch {self.state.epoch}"):
            if "pixel_values" not in batch:
                continue
            images = batch["pixel_values"].to(self.device)
            bsz = images.shape[0]
            task_ids = torch.zeros(bsz, dtype=torch.long, device=self.device)
            class_ids = torch.zeros(bsz, dtype=torch.long, device=self.device)
            out = self.model(images, task_ids, class_ids)
            loss = out["hidden"].pow(2).mean() * 0.0  # replace with AR + auxiliary losses
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach().cpu())
            steps += 1
            self.state.global_step += 1
        self.state.epoch += 1
        return {"loss": total / max(steps, 1)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "state": self.state.__dict__}, path)
