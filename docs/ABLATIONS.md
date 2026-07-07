# Ablation studies

This repository includes configuration templates for the five central ablation groups:

1. **CSGM ablation:** remove CSGM, class memory, scale memory, geo-memory alignment, or replace memory with random tokens.
2. **Routing ablation:** compare full `[x; task; geo]` routing against token-only, task-aware, geo-only, dense FFN, and MoE without load balancing.
3. **Spatial tokenization ablation:** compare AAB-only, RAB-only, mask-only, partial-format, and full AAB+RAB+mask tokenization.
4. **Progressive training ablation:** compare full two-stage training against dense-only, MoE from scratch, no FFN cloning, no CSGM activation, and frozen router.
5. **Instruction/augmentation ablation:** evaluate class-scale sampling, instruction expansion, asset composition, geometric remapping, and full geo-context augmentation.
