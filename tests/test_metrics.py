import numpy as np
from geomex_vlm.evaluation.metrics import aab_iou, count_mae, count_accuracy, spatial_agreement_score


def test_aab_iou():
    assert aab_iou((0,0,10,10), (0,0,10,10)) == 1.0
    assert aab_iou((0,0,10,10), (20,20,30,30)) == 0.0


def test_count_metrics():
    assert count_mae([1, 2, 4], [1, 3, 3]) == 2/3
    assert count_accuracy([1, 2, 4], [1, 3, 4]) == 2/3


def test_sas():
    mask = np.zeros((20,20), dtype=np.uint8)
    mask[5:15,5:15] = 1
    score = spatial_agreement_score((5,5,15,15), (10,10,10,10,0), mask)
    assert 0 <= score <= 1
