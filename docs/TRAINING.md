# Training protocol

GeoMEx-VLM follows progressive dense-to-sparse training.

## Stage I: dense remote-sensing alignment

- Optimize visual projection and LoRA/adaptation layers.
- Use instruction data for scene classification, VQA, captioning, counting, and basic grounding.
- Disable CSGM and sparse MoE to establish stable visual-language alignment.

## Stage II: geo-memory-guided expert specialization

- Convert selected FFN layers into sparse MoE layers.
- Initialize each expert from the corresponding dense FFN weights.
- Activate CSGM and route tokens with `[x_lj; e_tau; g_c]`.
- Train on AAB, RAB, mask, counting, scene, VQA, captioning, and strategic grounding samples.

## Objective

```text
L_total = L_AR + lambda_elb L_elb + lambda_gma L_gma + lambda_cfg L_cfg
```

Use `configs/train_stage1.yaml` and `configs/train_stage2.yaml` as starting points.
