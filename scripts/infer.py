from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json


def main():
    parser = argparse.ArgumentParser(description="Single-image inference placeholder.")
    parser.add_argument("--input", default="examples/inference_request.json")
    parser.add_argument("--checkpoint", default="checkpoints/geomex_vlm.pt")
    args = parser.parse_args()
    with open(args.input, "r", encoding="utf-8") as f:
        request = json.load(f)
    print("Loaded request:")
    print(json.dumps(request, indent=2))
    print("Connect this CLI to the trained decoder for full inference.")


if __name__ == "__main__":
    main()
