"""AP/mAP por quadro e remoção de detecções que casam com distratores."""

import numpy as np
import pytest

from pa2.metrics.detection import (average_precision, drop_ignored_detections,
                                   evaluate_detections)


def _gt(n_frames=3, per_frame=2):
    rows = []
    for f in range(1, n_frames + 1):
        for i in range(per_frame):
            rows.append([f, i + 1, 10 + 50 * i, 10, 20, 40, 1.0])
    return np.array(rows, float)


def _dets_from_gt(gt, score=0.9):
    return np.column_stack([gt[:, 0], gt[:, 2:6], np.full(len(gt), score)])


def test_perfect_detections_give_ap_one():
    gt = _gt()
    r = evaluate_detections(_dets_from_gt(gt), gt)
    assert r["ap50"] == pytest.approx(1.0) and r["map"] == pytest.approx(1.0)
    assert r["precision"] == r["recall"] == r["f1"] == 1.0


def test_no_detections_give_zero():
    r = evaluate_detections(np.zeros((0, 6)), _gt())
    assert r["ap50"] == 0.0 and r["recall"] == 0.0


def test_half_recall_ap_is_about_half():
    gt = _gt()
    dets = _dets_from_gt(gt)[::2]          # detecta metade, sem falsos positivos
    r = evaluate_detections(dets, gt)
    assert r["recall"] == pytest.approx(0.5) and r["precision"] == 1.0
    assert r["ap50"] == pytest.approx(0.505, abs=0.01)   # 51 de 101 pontos de recall


def test_false_positives_with_higher_score_lower_ap():
    gt = _gt()
    good = _dets_from_gt(gt, 0.5)
    fps = np.array([[1, 200, 200, 20, 40, 0.9], [2, 200, 200, 20, 40, 0.9]])
    r = evaluate_detections(np.vstack([good, fps]), gt)
    assert r["ap50"] < 1.0 and r["recall"] == 1.0


def test_ap_independent_of_order_inside_a_score_tie():
    gt = _gt()
    good = _dets_from_gt(gt, 1.0)
    fps = np.array([[1, 200, 200, 20, 40, 1.0], [2, 200, 200, 20, 40, 1.0]])
    a = evaluate_detections(np.vstack([good, fps]), gt)["ap50"]
    b = evaluate_detections(np.vstack([fps, good]), gt)["ap50"]
    assert a == pytest.approx(b)


def test_average_precision_known_curve():
    # 4 detecções por score decrescente: TP, FP, TP, FP; 2 GT
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    tp = np.array([True, False, True, False])
    # precisão: 1, .5, .667, .5 | recall: .5, .5, 1, 1 -> envelope: 1 (rec<=.5), .667 (rec<=1)
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert average_precision(scores, tp, 2) == pytest.approx(expected, abs=1e-6)


def test_distractor_matches_are_dropped_but_pedestrian_matches_kept():
    ped = np.array([[1, 1, 10, 10, 20, 40, 1.0]])
    ign = np.array([[1, 9, 100, 10, 20, 40]])
    dets = np.array([[1, 10, 10, 20, 40, 0.9],      # pedestre
                     [1, 100, 10, 20, 40, 0.9],      # pessoa estática (distrator)
                     [1, 300, 10, 20, 40, 0.9]])     # falso positivo de verdade
    kept, mask = drop_ignored_detections(dets, ped, ign)
    assert mask.tolist() == [True, False, True]
    assert len(kept) == 2


def test_detection_on_pedestrian_overlapping_distractor_is_not_dropped():
    ped = np.array([[1, 1, 10, 10, 20, 40, 1.0]])
    ign = np.array([[1, 9, 10, 10, 20, 40]])       # mesma caixa do pedestre
    dets = np.array([[1, 10, 10, 20, 40, 0.9]])
    _, mask = drop_ignored_detections(dets, ped, ign)
    assert mask.tolist() == [True]                  # já casou com o pedestre
