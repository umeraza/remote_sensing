from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.config import load_yaml, load_model_config
from geomex_vlm.model.geomex import GeoMExVLM
import torch


def main():
    parser = argparse.ArgumentParser(description="Stage-I dense remote-sensing alignment.")
    parser.add_argument("--config", default="configs/train_stage1.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model_cfg = load_model_config(cfg["model_config"])
    model = GeoMExVLM(model_cfg)
    print("Initialized Stage-I model scaffold.")
    print("Replace placeholder trainer loss with decoder autoregressive loss for full training.")


if __name__ == "__main__":
    main()
