# GeoMEx-VLM: Geo-Memory-Guided Sparse Expert Vision-Language Model

GeoMEx-VLM is a geo-memory-guided sparse expert vision-language framework for strategic asset grounding in remote sensing. It is organized around the paper components: Class-Scale Geo-Memory (CSGM), geo-memory-guided sparse Mixture-of-Experts (MoE) routing, unified AAB/RAB/mask spatial tokenization, instruction-corpus construction, geo-context augmentation, two-stage dense-to-sparse training, evaluation metrics, ablation studies, and result reporting.



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

## Reported Results

Following are the GeoMEx-VLM performance comparison with SoTA Models:

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


