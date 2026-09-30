"""Métricas de tracking para o PA2 — implementação própria.

Implementa:
- compute_idf1: IDF1 (Identity F1) como definido no MOTChallenge
- count_id_switches: número de trocas de identidade ao longo da sequência
- count_fragmentations: número de fragmentações de tracks
- match_global: atribuição global um-para-um entre tracks previstos e GT

Referência: Reis et al., "IDF1: Identity F1 Score for Multi-Object Tracking",
MOTChallenge benchmark evaluation metrics.

NOTA: ID switches e fragmentações são contados com base no matching frame a
frame (IoU entre caixa prevista e GT no frame), não no matching global.
O matching global serve apenas para o IDF1.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre duas caixas no formato MOT: [bb_left, bb_top, bb_width, bb_height]."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    x1 = max(xa, xb)
    y1 = max(ya, yb)
    x2 = min(xa + wa, xb + wb)
    y2 = min(ya + ha, yb + hb)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb
    union = area_a + area_b - inter

    return float(inter / union) if union > 0 else 0.0


def _parse_tracks_to_frames(
    tracks: list[dict[str, Any]],
) -> tuple[dict[int, dict[int, np.ndarray]], int]:
    """Converte lista de tracks MOT em dict {frame: {id: box}}."""
    frames: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    max_frame = 0
    for det in tracks:
        frame = int(det["frame"])
        obj_id = int(det["id"])
        box = np.array([
            float(det["bb_left"]),
            float(det["bb_top"]),
            float(det["bb_width"]),
            float(det["bb_height"]),
        ], dtype=np.float64)
        frames[frame][obj_id] = box
        max_frame = max(max_frame, frame)

    full_frames: dict[int, dict[int, np.ndarray]] = {}
    for f in range(1, max_frame + 1):
        full_frames[f] = frames.get(f, {})

    return full_frames, max_frame


def match_frame_by_iou(
    pred_frame: dict[int, np.ndarray],
    gt_frame: dict[int, np.ndarray],
    iou_threshold: float = 0.3,
) -> dict[int, int]:
    """Faz matching frame a frame entre predições e GT por IoU guloso decrescente.

    Retorna dict {gt_id → pred_id} para este frame.
    """
    gt_ids = sorted(gt_frame.keys())
    pred_ids = sorted(pred_frame.keys())

    # Calcula matriz de IoU
    iou_mat = np.zeros((len(pred_ids), len(gt_ids)))
    for i, pid in enumerate(pred_ids):
        for j, gid in enumerate(gt_ids):
            iou_mat[i, j] = compute_iou(pred_frame[pid], gt_frame[gid])

    # Ordena pares por IoU decrescente
    pairs = []
    for i in range(len(pred_ids)):
        for j in range(len(gt_ids)):
            if iou_mat[i, j] >= iou_threshold:
                pairs.append((iou_mat[i, j], i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_pred = set()
    matched_gt = set()
    gt_to_pred: dict[int, int] = {}

    for iou_val, i, j in pairs:
        pid = pred_ids[i]
        gid = gt_ids[j]
        if pid not in matched_pred and gid not in matched_gt:
            gt_to_pred[gid] = pid
            matched_pred.add(pid)
            matched_gt.add(gid)

    return gt_to_pred


def match_global(
    tracks_pred: dict[int, dict[int, np.ndarray]],
    tracks_gt: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.3,
    num_frames: int | None = None,
) -> tuple[
    dict[int, int],
    set[int],
    set[int],
]:
    """Atribuição global um-para-um entre identidades previstas e verdadeiras.

    Usa Hungarian (assignment ótimo) para maximizar a soma de IoUs ao longo
    de toda a sequência, respeitando o limiar de IoU.
    """
    if num_frames is None:
        all_frames = set(tracks_pred.keys()) | set(tracks_gt.keys())
        num_frames = max(all_frames) if all_frames else 0

    all_pred_ids: set[int] = set()
    all_gt_ids: set[int] = set()

    for f in range(1, num_frames + 1):
        all_pred_ids.update(tracks_pred.get(f, {}).keys())
        all_gt_ids.update(tracks_gt.get(f, {}).keys())

    pred_ids = sorted(all_pred_ids)
    gt_ids = sorted(all_gt_ids)

    if not pred_ids or not gt_ids:
        return {}, set(pred_ids), set(gt_ids)

    iou_matrix = np.zeros((len(pred_ids), len(gt_ids)))

    for f in range(1, num_frames + 1):
        pred_frame = tracks_pred.get(f, {})
        gt_frame = tracks_gt.get(f, {})
        for i, pid in enumerate(pred_ids):
            if pid not in pred_frame:
                continue
            box_p = pred_frame[pid]
            for j, gid in enumerate(gt_ids):
                if gid not in gt_frame:
                    continue
                box_g = gt_frame[gid]
                iou_matrix[i, j] += compute_iou(box_p, box_g)

    from scipy.optimize import linear_sum_assignment

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    pred_to_gt: dict[int, int] = {}
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_threshold * num_frames:
            pred_to_gt[pred_ids[i]] = gt_ids[j]

    matched_pred = set(pred_to_gt.keys())
    unmatched_pred = set(pred_ids) - matched_pred
    unmatched_gt = set(gt_ids) - set(pred_to_gt.values())

    return pred_to_gt, unmatched_pred, unmatched_gt


def count_id_switches(
    tracks_pred: dict[int, dict[int, np.ndarray]],
    tracks_gt: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.3,
    num_frames: int | None = None,
) -> int:
    """Conta o número de trocas de identidade (ID switches) na sequência.

    Um ID switch ocorre quando um ground truth ID é rastreado por um pred ID
    em um frame, e em um frame subsequente é rastreado por um pred ID diferente
    (mesmo que caixa confere por IoU).

    Mantém mapeamento persistente por GT identity: se um GT não aparece em
    um frame, o mapeamento anterior é preservado. Quando o GT reaparece com
    um pred_id diferente, conta-se um switch.
    """
    if num_frames is None:
        all_frames = set(tracks_pred.keys()) | set(tracks_gt.keys())
        num_frames = max(all_frames) if all_frames else 0

    last_pred_id_for_gt: dict[int, int] = {}
    switches = 0

    for f in range(1, num_frames + 1):
        pred_frame = tracks_pred.get(f, {})
        gt_frame = tracks_gt.get(f, {})

        gt_to_pred = match_frame_by_iou(pred_frame, gt_frame, iou_threshold)

        for gt_id, pred_id in gt_to_pred.items():
            if gt_id in last_pred_id_for_gt and last_pred_id_for_gt[gt_id] != pred_id:
                switches += 1
            last_pred_id_for_gt[gt_id] = pred_id

        # GTs que não aparecem neste frame mantêm o mapeamento anterior
        # (não fazemos nada — last_pred_id_for_gt preserva o valor antigo)

    return switches


def count_fragmentations(
    tracks_pred: dict[int, dict[int, np.ndarray]],
    tracks_gt: dict[int, dict[int, np.ndarray]],
    iou_threshold: float = 0.3,
    num_frames: int | None = None,
) -> int:
    """Conta o número de fragmentações.

    Uma fragmentação ocorre quando um ground truth ID deixa de ser rastreado
    por um pred_id (perde o matching IoU ou muda de pred_id) por pelo menos
    1 frame, e depois volta a ser rastreado (por qualquer pred_id).

    Usa o matching frame a frame entre predições e GT para determinar
    se um GT está sendo rastreado ou não em cada frame.
    """
    if num_frames is None:
        all_frames = set(tracks_pred.keys()) | set(tracks_gt.keys())
        num_frames = max(all_frames) if all_frames else 0

    gt_tracked_frames: dict[int, set[int]] = defaultdict(set)
    for f in range(1, num_frames + 1):
        pred_frame = tracks_pred.get(f, {})
        gt_frame = tracks_gt.get(f, {})

        gt_to_pred = match_frame_by_iou(pred_frame, gt_frame, iou_threshold)

        for gt_id in gt_frame.keys():
            if gt_id in gt_to_pred:
                gt_tracked_frames[gt_id].add(f)

    fragmentations = 0
    for gt_id, present_frames in gt_tracked_frames.items():
        sorted_frames = sorted(present_frames)
        if len(sorted_frames) < 2:
            continue
        for i in range(len(sorted_frames) - 1):
            if sorted_frames[i + 1] - sorted_frames[i] > 1:
                fragmentations += 1

    return fragmentations


def compute_idf1(
    tracks_pred: dict[int, dict[int, np.ndarray]],
    tracks_gt: dict[int, dict[int, np.ndarray]],
    pred_to_gt: dict[int, int] | None = None,
    iou_threshold: float = 0.3,
    num_frames: int | None = None,
) -> tuple[float, dict[str, int]]:
    """Calcula o IDF1 (Identity F1 Score).

    IDF1 = IDTP / (IDTP + 0.5 * (IDFP + IDFN))
    Usa matching global (Hungarian) para atribuição um-para-um.
    """
    if num_frames is None:
        all_frames = set(tracks_pred.keys()) | set(tracks_gt.keys())
        num_frames = max(all_frames) if all_frames else 0

    if pred_to_gt is None:
        pred_to_gt, unmatched_pred, unmatched_gt = match_global(
            tracks_pred, tracks_gt, iou_threshold, num_frames
        )
    else:
        unmatched_pred = set()
        unmatched_gt = set()

    all_gt_ids: set[int] = set()
    all_pred_ids: set[int] = set()
    for f in range(1, num_frames + 1):
        all_gt_ids.update(tracks_gt.get(f, {}).keys())
        all_pred_ids.update(tracks_pred.get(f, {}).keys())

    gt_frames_present: dict[int, set[int]] = defaultdict(set)
    for gt_id in all_gt_ids:
        for f in range(1, num_frames + 1):
            if gt_id in tracks_gt.get(f, {}):
                gt_frames_present[gt_id].add(f)

    pred_frames_present: dict[int, set[int]] = defaultdict(set)
    for pid in all_pred_ids:
        for f in range(1, num_frames + 1):
            if pid in tracks_pred.get(f, {}):
                pred_frames_present[pid].add(f)

    gt_to_pred_frames: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for f in range(1, num_frames + 1):
        pred_frame = tracks_pred.get(f, {})
        gt_frame = tracks_gt.get(f, {})
        for pid, box_p in pred_frame.items():
            gt_id = pred_to_gt.get(pid)
            if gt_id is not None and gt_id in gt_frame:
                gt_to_pred_frames[gt_id][pid].add(f)

    idtp = 0
    idfn = 0

    for gt_id in all_gt_ids:
        total_gt_frames = len(gt_frames_present[gt_id])
        if total_gt_frames == 0:
            idfn += 1
            continue

        best_frames = 0
        for pid, frames in gt_to_pred_frames[gt_id].items():
            if len(frames) > best_frames:
                best_frames = len(frames)

        if best_frames > total_gt_frames / 2:
            idtp += 1
        else:
            idfn += 1

    idtp_pred_ids: set[int] = set()
    for gt_id in all_gt_ids:
        total_gt_frames = len(gt_frames_present[gt_id])
        for pid, frames in gt_to_pred_frames[gt_id].items():
            if len(frames) > total_gt_frames / 2:
                idtp_pred_ids.add(pid)

    idfp = 0
    for pid in all_pred_ids:
        if pid not in idtp_pred_ids and pid not in set(pred_to_gt.keys()):
            idfp += 1

    denom = idtp + 0.5 * (idfp + idfn)
    idf1 = float(idtp / denom) if denom > 0 else 0.0

    return idf1, {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "n_gt_ids": len(all_gt_ids),
        "n_pred_ids": len(all_pred_ids),
    }


def evaluate_tracking_sequence(
    tracks_pred: list[dict[str, Any]],
    tracks_gt: list[dict[str, Any]],
    iou_threshold: float = 0.3,
) -> dict[str, Any]:
    """Avalia uma sequência completa de tracking e retorna todas as métricas.

    Retorna um dict com as métricas.
    """
    pred_frames, _n = _parse_tracks_to_frames(tracks_pred)
    gt_frames, num_frames = _parse_tracks_to_frames(tracks_gt)

    all_frames = set(pred_frames.keys()) | set(gt_frames.keys())
    num_frames = max(all_frames) if all_frames else 0

    id_switches = count_id_switches(pred_frames, gt_frames, iou_threshold, num_frames)
    fragmentations = count_fragmentations(pred_frames, gt_frames, iou_threshold, num_frames)

    pred_to_gt, _, _ = match_global(pred_frames, gt_frames, iou_threshold, num_frames)
    idf1, id_details = compute_idf1(pred_frames, gt_frames, pred_to_gt, iou_threshold, num_frames)

    total_gt_detections = sum(len(gt_frames[f]) for f in range(1, num_frames + 1))

    fp_count = 0
    fn_count = 0
    for f in range(1, num_frames + 1):
        pred_frame = pred_frames.get(f, {})
        gt_frame = gt_frames.get(f, {})

        for pid, box_p in pred_frame.items():
            matched = False
            for gid, box_g in gt_frame.items():
                if compute_iou(box_p, box_g) >= iou_threshold:
                    # Verifica se este pred_id está associado a este gt_id no matching global
                    if pred_to_gt.get(pid) == gid:
                        matched = True
                        break
            if not matched:
                fp_count += 1

        for gid, box_g in gt_frame.items():
            matched = False
            for pid, box_p in pred_frame.items():
                if compute_iou(box_p, box_g) >= iou_threshold:
                    if pred_to_gt.get(pid) == gid:
                        matched = True
                        break
            if not matched:
                fn_count += 1

    mota = 1.0 - (fn_count + fp_count + id_switches) / total_gt_detections if total_gt_detections > 0 else 0.0

    n_gt_ids = len(set(d["id"] for d in tracks_gt))
    n_pred_ids = len(set(d["id"] for d in tracks_pred))

    # Mostly tracked / mostly lost
    gt_tracked_frames: dict[int, int] = defaultdict(int)
    for f in range(1, num_frames + 1):
        pred_frame = pred_frames.get(f, {})
        gt_frame = gt_frames.get(f, {})
        for pid, box_p in pred_frame.items():
            for gid, box_g in gt_frame.items():
                if compute_iou(box_p, box_g) >= iou_threshold:
                    if pred_to_gt.get(pid) == gid:
                        gt_tracked_frames[gid] += 1

    gt_total_frames: dict[int, int] = defaultdict(int)
    for f in range(1, num_frames + 1):
        for gid in gt_frames.get(f, {}):
            gt_total_frames[gid] += 1

    mostly_tracked = 0
    mostly_lost = 0
    for gid in gt_total_frames:
        total = gt_total_frames[gid]
        tracked = gt_tracked_frames.get(gid, 0)
        if total > 0:
            ratio = tracked / total
            if ratio >= 0.8:
                mostly_tracked += 1
            elif ratio < 0.2:
                mostly_lost += 1

    return {
        "idf1": idf1,
        "id_switches": id_switches,
        "fragmentations": fragmentations,
        "n_gt_ids": len(gt_total_frames),
        "n_pred_ids": n_pred_ids,
        "mostly_tracked": mostly_tracked / len(gt_total_frames) if gt_total_frames else 0.0,
        "mostly_lost": mostly_lost / len(gt_total_frames) if gt_total_frames else 0.0,
        "num_frames": num_frames,
        "mota": mota,
        "IDTP": id_details["IDTP"],
        "IDFP": id_details["IDFP"],
        "IDFN": id_details["IDFN"],
    }
