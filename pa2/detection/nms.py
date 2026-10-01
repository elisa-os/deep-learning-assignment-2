"""Non-Maximum Suppression — implementação própria (``torchvision.ops.nms`` é proibido).

Greedy clássico: ordena por score, mantém a caixa de maior score e descarta as restantes
com IoU > ``iou_threshold`` em relação a ela; repete com o que sobrou.
"""

from __future__ import annotations

import numpy as np


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU entre (N, 4) e (M, 4) caixas ``x1, y1, x2, y2`` -> (N, M)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def nms(boxes, scores, iou_threshold: float = 0.5) -> np.ndarray:
    """NMS guloso. ``boxes`` (N, 4) em ``x1, y1, x2, y2``; devolve os índices mantidos,
    em ordem decrescente de score (empates: menor índice primeiro).

    Aceita ``np.ndarray`` ou ``torch.Tensor``.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        iou = box_iou_xyxy(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def batched_nms(boxes, scores, labels, iou_threshold: float = 0.5) -> np.ndarray:
    """NMS independente por classe (caixas de classes diferentes não se suprimem)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    keep: list[int] = []
    for c in np.unique(labels):
        idx = np.nonzero(labels == c)[0]
        keep.extend(idx[nms(boxes[idx], scores[idx], iou_threshold)].tolist())
    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(-scores[keep], kind="stable")]
