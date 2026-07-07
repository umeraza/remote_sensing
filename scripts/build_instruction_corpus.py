from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geomex_vlm.data.corpus_builder import build_mock_corpus, write_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build unified GeoMEx-VLM instruction corpus.")
    parser.add_argument("--datasets", default="configs/datasets.yaml", help="Dataset path configuration.")
    parser.add_argument("--output", default="data/geomex_instructions.jsonl")
    parser.add_argument("--mock", action="store_true", help="Write a small mock corpus for smoke testing.")
    args = parser.parse_args()
    # Dataset-specific parsers should be implemented in scripts/convert_*.py.
    samples = build_mock_corpus()
    write_jsonl(samples, args.output)
    print(f"Wrote {len(samples)} example records to {args.output}")


if __name__ == "__main__":
    main()
