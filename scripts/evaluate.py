from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate GeoMEx-VLM outputs.")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    print("Evaluation metrics configured:")
    for group, metrics in cfg.get("metrics", {}).items():
        print(f"  {group}: {', '.join(metrics)}")


if __name__ == "__main__":
    main()
