# Model card: GeoMEx-VLM

## Intended use

GeoMEx-VLM is intended for research on remote sensing vision-language understanding, strategic asset grounding, spatial reasoning, and sparse expert routing.

## Inputs

- RGB remote sensing image.
- Natural-language instruction.
- Optional class/scale metadata for training.

## Outputs

- Natural-language response.
- Count prediction.
- AAB token: `<box>[[x1,y1,x2,y2]]</box>`.
- RAB token: `<rab>[[xc,yc,w,h,theta]]</rab>`.
- Compressed mask token: `<seg>...</seg>`.

## Limitations

The scaffold does not include pretrained weights. The paper notes remaining challenges in cluttered harbor scenes, dense small-object counting, elongated-structure segmentation, and orientation-sensitive localization.
