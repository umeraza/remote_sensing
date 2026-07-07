# GeoMEx-VLM: Geo-Memory-Guided Sparse Expert Vision-Language Model

This repository is a research-code scaffold for **GeoMEx-VLM**, a geo-memory-guided sparse expert vision-language framework for strategic asset grounding in remote sensing. It is organized around the paper components: Class-Scale Geo-Memory (CSGM), geo-memory-guided sparse Mixture-of-Experts (MoE) routing, unified AAB/RAB/mask spatial tokenization, instruction-corpus construction, geo-context augmentation, two-stage dense-to-sparse training, evaluation metrics, ablation studies, and result reporting.

> **Status.** This release contains a clean, runnable implementation scaffold and dataset-conversion interfaces. It does not include pretrained weights or third-party datasets. Download each dataset from its official source and set paths in `configs/datasets.yaml`.

## Main features

- Unified instruction-response schema for scene classification, VQA, captioning, counting, AAB grounding, RAB localization, and compressed mask prediction.
- Spatial-token utilities for `<box>`, `<rab>`, and `<seg>` outputs.
- CSGM module for class- and scale-conditioned geospatial memory retrieval.
- Geo-memory-guided sparse MoE layer using token features, task embeddings, and CSGM context.
- Progressive Stage-I dense alignment and Stage-II sparse expert specialization scripts.
- Evaluation utilities for AAB IoU, RAB-to-AAB projection, mask-to-AAB projection, SAS, Count MAE, and count accuracy.
- Ablation configuration templates for CSGM, routing, spatial tokenization, progressive training, and geo-context augmentation.

## Repository layout

```text
GeoMEx-VLM/
├── configs/                  # dataset, model, training, evaluation, ablation configs
├── docs/                     # paper-aligned method, dataset, training, evaluation docs
├── examples/                 # small JSONL examples and inference input
├── scripts/                  # corpus conversion, training, evaluation, ablation CLIs
├── src/geomex_vlm/           # Python package
│   ├── data/                 # schema, dataset loader, augmentation, corpus builder
│   ├── evaluation/           # metrics and evaluator
│   ├── model/                # CSGM, MoE, spatial tokens, losses, model scaffold
│   └── training/             # trainer utilities
├── tests/                    # smoke tests for tokens, metrics, losses, dataset schema
├── results/                  # expected result tables and placeholders
├── CITATION.cff
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/<user>/GeoMEx-VLM.git
cd GeoMEx-VLM
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Prepare datasets

1. Download DIOR/DIOR-RSVG, DOTA-v2.0, FAIR1M, iSAID, GeoChat-Instruct, RefSegRS, and xView from their official sources.
2. Edit `configs/datasets.yaml` with local paths.
3. Convert native annotations into the unified JSONL schema:

```bash
python scripts/build_instruction_corpus.py \
  --datasets configs/datasets.yaml \
  --output data/geomex_instructions.jsonl
```

The expected JSONL format is documented in `docs/DATASETS.md` and illustrated in `examples/sample_instruction.jsonl`.

## Training

Stage I performs dense remote-sensing alignment:

```bash
python scripts/train_stage1.py --config configs/train_stage1.yaml
```

Stage II activates CSGM and geo-memory-guided sparse experts:

```bash
python scripts/train_stage2.py --config configs/train_stage2.yaml --resume checkpoints/stage1_last.pt
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/eval.yaml --checkpoint checkpoints/geomex_vlm.pt
```

Run a specific ablation:

```bash
python scripts/run_ablation.py --config configs/ablations/csgm.yaml
```

## Reported paper targets

The paper reports the following headline results for GeoMEx-VLM:

| Setting | Metric | Result |
|---|---:|---:|
| Overall multi-task | VQA Acc. | 80.92 |
| Overall multi-task | CIDEr | 95.7 |
| Overall multi-task | AAB-G | 78.36 |
| Overall multi-task | RAB-G | 58.72 |
| Overall multi-task | RES | 75.42 |
| Overall multi-task | Count MAE | 1.74 |
| DIOR-RSVG | Pr@0.5 / Pr@0.9 / cIoU | 82.36 / 39.84 / 83.76 |
| DOTA-v2.0 | r-mAP | 64.72 |
| FAIR1M | rAP50 / rAP75 | 79.91 / 57.42 |
| RefSegRS | Pr@0.5 / Pr@0.7 / oIoU | 85.04 / 71.28 / 82.16 |
| Instruction/counting | Count Acc. / SAS | 76.35 / 74.02 |

## Reproducibility notes

- Dataset licenses and redistribution rules differ; this repository stores conversion code but not dataset files.
- The model scaffold uses modular components so that the visual encoder and decoder-only LLM can be replaced by your selected pretrained backbones.
- Exact paper reproduction requires the same dataset splits, pretrained model initialization, preprocessing, and GPU training budget described in the manuscript.

## Citation

See `CITATION.cff`.
