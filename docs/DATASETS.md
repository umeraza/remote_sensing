# Datasets and unified instruction corpus

GeoMEx-VLM uses a heterogeneous dataset suite:

- **DIOR/DIOR-RSVG:** AAB grounding, object counting, language-conditioned localization.
- **DOTA-v2.0:** RAB token learning, rotation-aware grounding, class-scale memory construction.
- **FAIR1M:** fine-grained strategic asset recognition and RAB prediction.
- **iSAID:** instance masks for compressed `<seg>` token supervision.
- **GeoChat-Instruct:** Stage-I dense vision-language instruction alignment.
- **RefSegRS:** text-conditioned mask grounding.
- **xView:** dense small-object localization, counting, and robustness.

## Unified JSONL schema

Each line contains:

```json
{
  "image": "relative/path.png",
  "instruction": "Ground the docked aircraft carrier.",
  "answer": "<rab>[[...]]</rab>",
  "task": "rab_grounding",
  "class_name": "aircraft_carrier",
  "class_id": 2,
  "scale_id": 1,
  "spatial_format": "rab",
  "annotation_source": "native",
  "aab": [[x1, y1, x2, y2]],
  "rab": [[xc, yc, w, h, theta]],
  "mask_rle": "<seg>...</seg>",
  "metadata": {"dataset": "DOTA-v2.0", "split": "train"}
}
```

Native labels should be used for primary evaluation. Derived labels such as RAB-to-AAB or mask-to-AAB should be stored as auxiliary supervision for cross-format training and SAS analysis.
