"""Métricas de detecção por quadro (AP / mAP) — implementação própria.

Usa caixas ``[left, top, width, height]``. O AP segue o protocolo COCO (interpolação em
101 pontos de recall); ``AP50`` usa IoU 0.5 e ``mAP`` a média sobre IoU 0.50:0.05:0.95.

Empates de score. Os detectores públicos do MOT17 (FRCNN, SDP) saturam o score em 1.0
para a maioria das caixas. Para o resultado não depender da ordem arbitrária dentro de
um empate, a curva precisão x recall só é avaliada nos limiares *distintos* de score
(todas as detecções com o mesmo score entram juntas).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from pa2.metrics.tracking import iou_matrix

IOU_THRESHOLDS = np.arange(0.5, 0.951, 0.05)


def drop_ignored_detections(
    dets: np.ndarray,
    gt_ped: np.ndarray,
    gt_ignore: np.ndarray,
    iou_threshold: float = 0.5,
    boxes_cols: slice = slice(1, 5),
) -> tuple[np.ndarray, np.ndarray]:
    """Remove detecções que casam com GT de classes distratoras.

    Em cada quadro, primeiro casa as detecções com os pedestres (Hungarian, IoU >=
    limiar); as que sobram e casam com um distrator (Hungarian) são descartadas, pois
    detectar uma pessoa estática ou um reflexo não é um erro. É o pré-processamento do
    benchmark MOT17. Devolve ``(dets_mantidas, máscara_mantidas)``.

    ``dets``: (M, >=5) com o quadro na coluna 0 e a caixa em ``boxes_cols``.
    ``gt_ped`` e ``gt_ignore``: linhas ``frame, id, l, t, w, h, ...``.
    """
    if len(dets) == 0 or len(gt_ignore) == 0:
        return dets, np.ones(len(dets), dtype=bool)

    keep = np.ones(len(dets), dtype=bool)
    ped_f = _by_frame_idx(gt_ped)
    ign_f = _by_frame_idx(gt_ignore)
    for f, d_idx in _by_frame_idx(dets).items():
        if f not in ign_f:
            continue
        d_boxes = dets[d_idx][:, boxes_cols]
        free = np.ones(len(d_idx), dtype=bool)
        if f in ped_f:
            iou = iou_matrix(d_boxes, gt_ped[ped_f[f]][:, 2:6])
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= iou_threshold:
                    free[i] = False
        if free.any():
            sub = np.nonzero(free)[0]
            iou = iou_matrix(d_boxes[sub], gt_ignore[ign_f[f]][:, 2:6])
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= iou_threshold:
                    keep[d_idx[sub[i]]] = False
    return dets[keep], keep


def _by_frame_idx(arr: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, list[int]] = {}
    for i, f in enumerate(arr[:, 0].astype(int)):
        out.setdefault(int(f), []).append(i)
    return {f: np.asarray(idx) for f, idx in out.items()}


def match_detections(
    dets: np.ndarray, gt_ped: np.ndarray, thresholds: np.ndarray = IOU_THRESHOLDS
) -> tuple[np.ndarray, np.ndarray, int]:
    """Casa detecções com GT em cada quadro, guloso por score (como no COCO).

    ``dets``: (M, 6) frame, l, t, w, h, score. ``gt_ped``: (N, >=6) frame, id, l, t, w, h.
    Retorna ``(scores (M,), tp (T, M) bool, n_gt)`` com ``T = len(thresholds)``.
    """
    n_gt = len(gt_ped)
    scores = dets[:, 5].copy()
    tp = np.zeros((len(thresholds), len(dets)), dtype=bool)
    gt_f = _by_frame_idx(gt_ped)
    for f, d_idx in _by_frame_idx(dets).items():
        if f not in gt_f:
            continue
        order = d_idx[np.argsort(-dets[d_idx, 5], kind="stable")]
        iou = iou_matrix(dets[order][:, 1:5], gt_ped[gt_f[f]][:, 2:6])
        for t_i, thr in enumerate(thresholds):
            used = np.zeros(iou.shape[1], dtype=bool)
            for k, d in enumerate(order):
                cand = np.where(used, -1.0, iou[k])
                j = int(np.argmax(cand))
                if cand[j] >= thr:
                    used[j] = True
                    tp[t_i, d] = True
    return scores, tp, n_gt


def average_precision(scores: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """AP (101 pontos de recall) para um limiar de IoU. ``tp`` é (M,) bool."""
    if n_gt == 0:
        return 0.0
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    s, t = scores[order], tp[order]
    ctp = np.cumsum(t)
    cfp = np.cumsum(~t)
    # só os fins de grupo de scores empatados
    last = np.r_[s[1:] != s[:-1], True]
    recall = ctp[last] / n_gt
    precision = ctp[last] / (ctp[last] + cfp[last])
    # envelope monotônico da precisão
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    vals = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(vals.mean())


def evaluate_detections(
    dets: np.ndarray,
    gt_ped: np.ndarray,
    gt_ignore: np.ndarray | None = None,
    score_threshold: float = -np.inf,
    full: bool = True,
) -> dict[str, float]:
    """AP50, mAP (0.5:0.95) e precisão/recall/F1 em IoU 0.5 para um conjunto de detecções.

    ``score_threshold`` filtra as detecções *antes* de tudo (precisão/recall/F1 são
    medidos nesse ponto de operação; AP50/mAP percorrem todos os scores acima dele).
    Com ``full=False`` só o IoU 0.5 é avaliado (mais rápido; ``map`` fica igual a ``ap50``).
    """
    dets = dets[dets[:, 5] >= score_threshold]
    if gt_ignore is not None and len(gt_ignore):
        dets, _ = drop_ignored_detections(dets, gt_ped, gt_ignore)
    thresholds = IOU_THRESHOLDS if full else IOU_THRESHOLDS[:1]
    scores, tp, n_gt = match_detections(dets, gt_ped, thresholds)
    aps = [average_precision(scores, tp[i], n_gt) for i in range(len(thresholds))]
    n_tp = int(tp[0].sum())
    precision = n_tp / len(dets) if len(dets) else 0.0
    recall = n_tp / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "ap50": aps[0], "map": float(np.mean(aps)),
        "precision": precision, "recall": recall, "f1": f1,
        "n_dets": len(dets), "n_gt": n_gt, "n_tp": n_tp,
    }
