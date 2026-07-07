# File manifest

## Core repository files

- `README.md`: project overview, installation, training, evaluation, reported metrics.
- `LICENSE`: MIT license placeholder for generated code.
- `CITATION.cff`: citation metadata for GitHub and Zenodo-style software citation.
- `CONTRIBUTING.md`: contribution workflow.
- `requirements.txt`, `environment.yml`, `pyproject.toml`: Python packaging and environment files.
- `.gitignore`: excludes datasets, checkpoints, logs, and build artifacts.

## Source files

- `src/geomex_vlm/model/csgm.py`: Class-Scale Geo-Memory.
- `src/geomex_vlm/model/moe.py`: geo-memory-guided sparse MoE routing.
- `src/geomex_vlm/model/spatial_tokens.py`: AAB/RAB/mask tokenization.
- `src/geomex_vlm/model/losses.py`: AR, ELB, GMA, CFG loss utilities.
- `src/geomex_vlm/data/*`: instruction schema, dataset loader, augmentation, corpus builder.
- `src/geomex_vlm/evaluation/*`: IoU, SAS, counting, and evaluator utilities.
- `src/geomex_vlm/training/trainer.py`: training interface.

## Experiment files

- `configs/*.yaml`: model, dataset, training, and evaluation configs.
- `configs/ablations/*.yaml`: ablation templates.
- `scripts/*.py`: conversion, training, evaluation, inference, and result export commands.
- `docs/*.md`: method and reproducibility documentation.
- `examples/*`: small JSONL and inference request examples.
- `tests/*`: smoke tests.
