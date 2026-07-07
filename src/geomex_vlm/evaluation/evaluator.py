from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .metrics import count_accuracy, count_mae, precision_at_iou


@dataclass
class EvaluationAccumulator:
    pred_counts: List[int] = field(default_factory=list)
    gt_counts: List[int] = field(default_factory=list)
    pred_aab: List[list] = field(default_factory=list)
    gt_aab: List[list] = field(default_factory=list)

    def summarize(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.gt_counts:
            out["count_mae"] = count_mae(self.pred_counts, self.gt_counts)
            out["count_accuracy"] = count_accuracy(self.pred_counts, self.gt_counts)
        if self.gt_aab:
            out["pr_0_5"] = precision_at_iou(self.pred_aab, self.gt_aab, 0.5)
            out["pr_0_7"] = precision_at_iou(self.pred_aab, self.gt_aab, 0.7)
            out["pr_0_9"] = precision_at_iou(self.pred_aab, self.gt_aab, 0.9)
        return out
