from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description="Run a GeoMEx-VLM ablation configuration.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    print(f"Ablation: {cfg.get('name')}")
    print("Variants:")
    for v in cfg.get("variants", []):
        print(f"  - {v}")


if __name__ == "__main__":
    main()
