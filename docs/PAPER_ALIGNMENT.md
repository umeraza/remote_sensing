# Paper-to-repository alignment

| Paper section | Repository files |
|---|---|
| Proposed GeoMEx-VLM | `src/geomex_vlm/model/geomex.py`, `vision_encoder.py`, `csgm.py`, `moe.py` |
| Class-Scale Geo-Memory | `src/geomex_vlm/model/csgm.py` |
| Geo-memory-guided routing | `src/geomex_vlm/model/moe.py` |
| Unified Spatial Tokenization | `src/geomex_vlm/model/spatial_tokens.py` |
| Training objectives | `src/geomex_vlm/model/losses.py` |
| Cross-Representation Spatial Agreement | `src/geomex_vlm/evaluation/metrics.py` |
| Datasets and instruction corpus | `configs/datasets.yaml`, `src/geomex_vlm/data/*`, `scripts/convert_*.py` |
| Geo-Context Augmentation | `src/geomex_vlm/data/augmentations.py` |
| Training Strategy | `configs/train_stage1.yaml`, `configs/train_stage2.yaml`, `scripts/train_stage1.py`, `scripts/train_stage2.py` |
| Evaluation Metrics | `configs/eval.yaml`, `src/geomex_vlm/evaluation/*` |
| Ablation Studies | `configs/ablations/*.yaml`, `scripts/run_ablation.py` |
| Performance Comparison | `results/README.md` |
