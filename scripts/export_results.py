from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Export evaluation JSON to a Markdown table.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="results/summary.md")
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text())
    lines = ["| Metric | Value |", "|---|---:|"]
    for k, v in sorted(data.items()):
        lines.append(f"| {k} | {v} |")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
