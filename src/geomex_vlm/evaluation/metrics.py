from __future__ import annotations

from typing import Sequence
import numpy as np

from geomex_vlm.model.spatial_tokens import rbox_to_aab, mask_to_aab


def aab_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def precision_at_iou(pred_boxes, gt_boxes, threshold: float = 0.5) -> float:
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    hits = 0
    used = set()
    for pred in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            if j in used:
                continue
            iou = aab_iou(pred, gt)
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= threshold:
            hits += 1
            used.add(best_j)
    return hits / max(len(gt_boxes), 1)


def count_mae(pred_counts, gt_counts) -> float:
    return float(np.mean(np.abs(np.asarray(pred_counts) - np.asarray(gt_counts))))


def count_accuracy(pred_counts, gt_counts) -> float:
    pred = np.asarray(pred_counts)
    gt = np.asarray(gt_counts)
    return float(np.mean(pred == gt))


def spatial_agreement_score(aab, rab, mask) -> float:
    rab_aab = rbox_to_aab(rab)
    mask_aab = mask_to_aab(mask)
    return (aab_iou(aab, rab_aab) + aab_iou(aab, mask_aab) + aab_iou(rab_aab, mask_aab)) / 3.0
