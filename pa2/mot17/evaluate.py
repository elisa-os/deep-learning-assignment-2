"""Avaliação de tracks e detecções contra o GT do MOT17 (convenções do benchmark)."""

from __future__ import annotations

import numpy as np

from pa2.metrics.detection import drop_ignored_detections, evaluate_detections
from pa2.metrics.tracking import evaluate_tracking_sequence
from pa2.mot17.loader import Sequence, to_mot_records


def evaluate_tracks(seq: Sequence, tracks: list[dict], iou_threshold: float = 0.5) -> dict:
    """IDF1, ID switches, fragmentações etc. de ``tracks`` (formato MOT) para a sequência.

    Antes de medir, remove as caixas previstas que casam com GT de classes distratoras
    (pessoa estática, reflexo...) e não com um pedestre.
    """
    gt = seq.gt_pedestrians()
    if tracks:
        arr = np.array([[t["frame"], t["id"], t["bb_left"], t["bb_top"], t["bb_width"],
                         t["bb_height"]] for t in tracks], dtype=np.float64)
        _, keep = drop_ignored_detections(arr, gt, seq.gt_distractors(),
                                          iou_threshold, boxes_cols=slice(2, 6))
        tracks = [t for t, k in zip(tracks, keep) if k]
    res = evaluate_tracking_sequence(tracks, to_mot_records(gt), iou_threshold)
    res["n_pred_boxes"] = len(tracks)
    return res


def evaluate_sequence_detections(
    seq: Sequence, dets: np.ndarray | None = None, score_threshold: float = -np.inf
) -> dict:
    """AP50 / mAP / P / R / F1 das detecções (default: as públicas da sequência)."""
    dets = seq.det if dets is None else dets
    return evaluate_detections(dets, seq.gt_pedestrians(), seq.gt_distractors(), score_threshold)
