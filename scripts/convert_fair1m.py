from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.data.corpus_builder import write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Convert FAIR1M annotations to GeoMEx-VLM JSONL schema.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Add FAIR1M-specific annotation parsing here. Output records must follow GeoInstructionSample.")


if __name__ == "__main__":
    main()
