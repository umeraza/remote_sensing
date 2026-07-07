from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml


@dataclass
class GeoMExConfig:
    image_size: int = 512
    patch_size: int = 16
    vision_dim: int = 768
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_classes: int = 64
    num_scales: int = 4
    memory_slots: int = 8
    memory_hw: int = 8
    num_experts: int = 8
    top_k: int = 2
    aab_resolution: int = 1000
    rab_resolution: int = 1000
    mask_resolution: int = 64


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model_config(path: str | Path) -> GeoMExConfig:
    cfg = load_yaml(path)
    model_cfg = cfg.get("model", cfg)
    return GeoMExConfig(**{k: v for k, v in model_cfg.items() if k in GeoMExConfig.__dataclass_fields__})
