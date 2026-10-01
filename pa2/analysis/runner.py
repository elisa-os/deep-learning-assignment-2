"""Roda o rastreador com movimento guardando o que a Parte 4 precisa inspecionar.

Além dos pares (id, detecção) de cada quadro, guarda a **caixa que a recorrência previu** para
cada track confirmada antes de ver o quadro (o "portão" da associação). É o mapa intermediário
das figuras de falha.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from pa2.association.motion_tracker import MotionTracker
from pa2.metrics.tracking import iou_matrix
from pa2.mot17.loader import Sequence


def sorted_dets(dets: np.ndarray, min_conf: float) -> np.ndarray:
    """Detecções com score >= min_conf, ordenadas por quadro (ordem estável)."""
    d = dets[dets[:, 5] >= min_conf]
    return d[np.argsort(d[:, 0], kind="stable")]


def run_trace(D: np.ndarray, num_frames: int, motion, keep: np.ndarray | None = None, **tracker_kw):
    """``D`` (M, 6) ordenado por quadro. ``keep`` (M,) bool remove detecções (oclusão injetada).

    Devolve ``assign`` (lista por quadro, índice f-1, de ``(track_id, linha_em_D)``) e
    ``gates`` (lista por quadro de ``{track_id: caixa prevista para esse quadro [l, t, w, h]}``).
    """
    rows = np.arange(len(D)) if keep is None else np.nonzero(keep)[0]
    d = D[rows]
    frames = d[:, 0].astype(int)
    tracker = MotionTracker(motion, **tracker_kw)
    assign, gates = [], []
    for f in range(1, num_frames + 1):
        sel = np.nonzero(frames == f)[0]
        gates.append({t.track_id: t.pred.copy() for t in tracker.tracks if t.track_id is not None})
        pairs = tracker.update(d[sel, 1:5], d[sel, 5])
        assign.append([(tid, int(rows[sel[k]])) for tid, k in pairs])
    return assign, gates


def match_gt_to_dets(seq: Sequence, D: np.ndarray, iou_threshold: float = 0.5) -> dict[tuple[int, int], int]:
    """(id_do_GT, quadro) -> linha de D da detecção que casa com ele (Hungarian por quadro)."""
    gt = seq.gt_pedestrians()
    out: dict[tuple[int, int], int] = {}
    lo = np.searchsorted(D[:, 0], np.arange(1, seq.info.seq_length + 2), side="left")
    for f in range(1, seq.info.seq_length + 1):
        g = gt[gt[:, 0] == f]
        a, b = lo[f - 1], lo[f]
        if len(g) == 0 or b == a:
            continue
        iou = iou_matrix(g[:, 2:6], D[a:b, 1:5])
        r, c = linear_sum_assignment(-iou)
        for i, j in zip(r, c):
            if iou[i, j] >= iou_threshold:
                out[(int(g[i, 1]), f)] = int(a + j)
    return out


def records_from_assign(D: np.ndarray, assign) -> list[dict]:
    """Converte ``assign`` de ``run_trace`` em tracks no formato MOT (para as métricas)."""
    out = []
    for f, pairs in enumerate(assign, start=1):
        for tid, r in pairs:
            l, t, w, h, s = D[r, 1:6]
            out.append({"frame": f, "id": tid, "bb_left": float(l), "bb_top": float(t),
                        "bb_width": float(w), "bb_height": float(h), "conf": float(s)})
    return out
