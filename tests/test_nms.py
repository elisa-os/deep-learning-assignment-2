"""NMS próprio: casos à mão e comparação com uma versão ingênua escrita à parte."""

import numpy as np
import pytest
import torch

from pa2.detection.nms import batched_nms, box_iou_xyxy, nms


def _naive_nms(boxes, scores, thr):
    """Referência O(n²) com laços explícitos (independente de nms())."""
    def iou(a, b):
        iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = iw * ih
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    kept = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= thr for j in kept):
            kept.append(i)
    return kept


def test_hand_case():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10.5]], float)
    scores = np.array([0.9, 0.8, 0.7, 0.95])
    # a caixa 3 (maior score) suprime 0 e 1; a 2 está longe
    assert nms(boxes, scores, 0.5).tolist() == [3, 2]


def test_iou_values():
    iou = box_iou_xyxy(np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]]))
    assert iou[0].tolist() == pytest.approx([1.0, 1 / 3, 0.0])


def test_empty_and_single():
    assert len(nms(np.zeros((0, 4)), np.zeros(0))) == 0
    assert nms(np.array([[0, 0, 1, 1]]), np.array([0.3])).tolist() == [0]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("thr", [0.3, 0.5, 0.7])
def test_matches_naive_reference(seed, thr):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (60, 2))
    wh = rng.uniform(5, 40, (60, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1)
    scores = rng.random(60)
    assert nms(boxes, scores, thr).tolist() == _naive_nms(boxes, scores, thr)


def test_tied_scores_are_deterministic_by_index():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    assert nms(boxes, np.array([0.5, 0.5]), 0.5).tolist() == [0]


def test_accepts_torch_tensors():
    boxes = torch.tensor([[0.0, 0, 10, 10], [1, 1, 11, 11]])
    assert nms(boxes, torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]


def test_batched_does_not_suppress_across_classes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    assert sorted(batched_nms(boxes, np.array([0.9, 0.8]), np.array([1, 2]), 0.5).tolist()) == [0, 1]
    assert batched_nms(boxes, np.array([0.9, 0.8]), np.array([1, 1]), 0.5).tolist() == [0]
