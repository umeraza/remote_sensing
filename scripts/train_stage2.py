from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.config import load_yaml, load_model_config
from geomex_vlm.model.geomex import GeoMExVLM


def main():
    parser = argparse.ArgumentParser(description="Stage-II geo-memory-guided sparse expert specialization.")
    parser.add_argument("--config", default="configs/train_stage2.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    model_cfg = load_model_config(cfg["model_config"])
    model = GeoMExVLM(model_cfg)
    print("Initialized Stage-II model scaffold with CSGM and sparse MoE.")
    if args.resume:
        print(f"Resume checkpoint requested: {args.resume}")


if __name__ == "__main__":
    main()
