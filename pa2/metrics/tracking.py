"""Métricas de tracking para o PA2 — implementação própria (sem motmetrics/TrackEval).

Implementa:
- compute_idf1: IDF1 (Ristani et al., 2016), contado em detecções (caixas por quadro)
- count_id_switches / count_fragmentations: matching quadro a quadro estilo CLEAR-MOT
- match_global: atribuição global um-para-um entre trajetórias previstas e verdadeiras
- evaluate_tracking_sequence: todas as métricas de uma sequência

Convenções
----------
Tracks são listas de dicts no formato MOT: ``frame, id, bb_left, bb_top, bb_width,
bb_height`` (+ ``conf``). Internamente viram ``{frame: {id: caixa[4]}}``.

IDF1
    Cada identidade (verdadeira ou prevista) é uma trajetória. Para um par (g, p),
    ``IDTP(g, p)`` é o número de quadros em que ambos existem e IoU >= limiar. A
    atribuição um-para-um que maximiza a soma de IDTP é obtida com Hungarian.
    Com ``Ng``/``Np`` o total de caixas verdadeiras/previstas:

        IDTP = soma dos IDTP dos pares atribuídos
        IDFN = Ng - IDTP,  IDFP = Np - IDTP
        IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

ID switches / fragmentações (CLEAR-MOT)
    A cada quadro, correspondências do quadro anterior que continuam válidas
    (IoU >= limiar) são mantidas; o restante é casado por Hungarian sobre o IoU.
    Um ID switch é contado quando uma identidade verdadeira passa a ser casada
    com um id previsto diferente do último com que foi casada (mesmo após
    lacunas). Uma fragmentação é uma transição "rastreada -> não rastreada ->
    rastreada" dentro da vida da identidade verdadeira.

Ground truth com ``conf == 0`` é ignorado por ``evaluate_tracking_sequence``
(convenção do MOT17: caixas marcadas como "não considerar", p.ex. objetos
totalmente ocluídos ou distratores).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

FrameDict = dict[int, dict[int, np.ndarray]]


# ─────────────────────────────────────────────────────────────────────────────
# Geometria
# ─────────────────────────────────────────────────────────────────────────────
def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre duas caixas no formato MOT: [bb_left, bb_top, bb_width, bb_height]."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    inter_w = max(0.0, min(xa + wa, xb + wb) - max(xa, xb))
    inter_h = max(0.0, min(ya + ha, yb + hb) - max(ya, yb))
    inter = inter_w * inter_h
    union = wa * ha + wb * hb - inter
    return float(inter / union) if union > 0 else 0.0


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU vetorizado entre (N,4) e (M,4) caixas [left, top, w, h] -> (N,M)."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0:1], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1:2], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Conversão de formato
# ─────────────────────────────────────────────────────────────────────────────
def parse_tracks_to_frames(tracks: list[dict[str, Any]]) -> tuple[FrameDict, int]:
    """Converte lista de tracks MOT em ``{frame: {id: caixa}}`` e devolve o maior quadro."""
    frames: FrameDict = defaultdict(dict)
    max_frame = 0
    for det in tracks:
        frame = int(det["frame"])
        frames[frame][int(det["id"])] = np.array(
            [det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"]],
            dtype=np.float64,
        )
        max_frame = max(max_frame, frame)
    return dict(frames), max_frame


def frame_range(tracks_pred: FrameDict, tracks_gt: FrameDict, num_frames: int | None) -> list[int]:
    """Quadros a percorrer: união dos quadros presentes (ou 1..num_frames se informado)."""
    if num_frames is not None:
        return list(range(1, num_frames + 1))
    return sorted(set(tracks_pred) | set(tracks_gt))


def _stack(frame: dict[int, np.ndarray]) -> tuple[list[int], np.ndarray]:
    ids = sorted(frame)
    boxes = np.stack([frame[i] for i in ids]) if ids else np.zeros((0, 4))
    return ids, boxes


# ─────────────────────────────────────────────────────────────────────────────
# Matching global (IDF1)
# ─────────────────────────────────────────────────────────────────────────────
def _idtp_matrix(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    iou_threshold: float,
    frames: list[int],
) -> tuple[list[int], list[int], np.ndarray]:
    """Matriz (pred x gt) com o número de quadros em que o par coincide (IoU >= limiar)."""
    pred_ids = sorted({i for f in frames for i in tracks_pred.get(f, {})})
    gt_ids = sorted({i for f in frames for i in tracks_gt.get(f, {})})
    p_idx = {pid: k for k, pid in enumerate(pred_ids)}
    g_idx = {gid: k for k, gid in enumerate(gt_ids)}
    counts = np.zeros((len(pred_ids), len(gt_ids)), dtype=np.int64)

    for f in frames:
        pf, gf = tracks_pred.get(f, {}), tracks_gt.get(f, {})
        if not pf or not gf:
            continue
        p_list, p_boxes = _stack(pf)
        g_list, g_boxes = _stack(gf)
        hit = iou_matrix(p_boxes, g_boxes) >= iou_threshold
        for a, b in zip(*np.nonzero(hit)):
            counts[p_idx[p_list[a]], g_idx[g_list[b]]] += 1
    return pred_ids, gt_ids, counts


def match_global(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    iou_threshold: float = 0.5,
    num_frames: int | None = None,
) -> tuple[dict[int, int], set[int], set[int]]:
    """Atribuição global um-para-um entre identidades previstas e verdadeiras.

    Maximiza o número total de quadros coincidentes (IDTP) com Hungarian.
    Retorna ``(pred_id -> gt_id, preds_sem_par, gts_sem_par)``.
    """
    frames = frame_range(tracks_pred, tracks_gt, num_frames)
    pred_ids, gt_ids, counts = _idtp_matrix(tracks_pred, tracks_gt, iou_threshold, frames)
    if not pred_ids or not gt_ids:
        return {}, set(pred_ids), set(gt_ids)

    rows, cols = linear_sum_assignment(-counts)
    pred_to_gt = {pred_ids[i]: gt_ids[j] for i, j in zip(rows, cols) if counts[i, j] > 0}
    return (
        pred_to_gt,
        set(pred_ids) - set(pred_to_gt),
        set(gt_ids) - set(pred_to_gt.values()),
    )


def compute_idf1(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    pred_to_gt: dict[int, int] | None = None,
    iou_threshold: float = 0.5,
    num_frames: int | None = None,
) -> tuple[float, dict[str, int | float]]:
    """IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN), com contagens em caixas.

    Se ``pred_to_gt`` for dado, usa essa atribuição em vez de recalculá-la.
    """
    frames = frame_range(tracks_pred, tracks_gt, num_frames)
    pred_ids, gt_ids, counts = _idtp_matrix(tracks_pred, tracks_gt, iou_threshold, frames)

    n_pred_boxes = sum(len(tracks_pred.get(f, {})) for f in frames)
    n_gt_boxes = sum(len(tracks_gt.get(f, {})) for f in frames)

    idtp = 0
    if pred_ids and gt_ids:
        if pred_to_gt is None:
            rows, cols = linear_sum_assignment(-counts)
            idtp = int(counts[rows, cols].sum())
        else:
            p_idx = {pid: k for k, pid in enumerate(pred_ids)}
            g_idx = {gid: k for k, gid in enumerate(gt_ids)}
            idtp = int(sum(
                counts[p_idx[p], g_idx[g]]
                for p, g in pred_to_gt.items() if p in p_idx and g in g_idx
            ))

    idfn = n_gt_boxes - idtp
    idfp = n_pred_boxes - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = float(2 * idtp / denom) if denom > 0 else 0.0
    idp = float(idtp / n_pred_boxes) if n_pred_boxes else 0.0
    idr = float(idtp / n_gt_boxes) if n_gt_boxes else 0.0

    return idf1, {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "IDP": idp,
        "IDR": idr,
        "n_gt_ids": len(gt_ids),
        "n_pred_ids": len(pred_ids),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Matching quadro a quadro (CLEAR-MOT): ID switches, fragmentações, FP/FN
# ─────────────────────────────────────────────────────────────────────────────
def clear_match(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    iou_threshold: float,
    frames: list[int],
) -> dict[str, Any]:
    """Percorre os quadros fazendo matching com continuidade (estilo CLEAR-MOT)."""
    last_pred_for_gt: dict[int, int] = {}   # último pred casado por GT (persiste em lacunas)
    prev_match: dict[int, int] = {}          # gt -> pred casados no quadro anterior
    tracked_seq: dict[int, list[bool]] = defaultdict(list)  # por GT: rastreado em cada quadro de vida
    matches_per_frame: dict[int, dict[int, int]] = {}
    switches = fp = fn = tp = 0

    for f in frames:
        pf, gf = tracks_pred.get(f, {}), tracks_gt.get(f, {})
        p_list, p_boxes = _stack(pf)
        g_list, g_boxes = _stack(gf)
        iou = iou_matrix(g_boxes, p_boxes)
        g_pos = {g: k for k, g in enumerate(g_list)}
        p_pos = {p: k for k, p in enumerate(p_list)}

        match: dict[int, int] = {}
        # 1) mantém correspondências do quadro anterior que continuam válidas
        for g, p in prev_match.items():
            if g in g_pos and p in p_pos and iou[g_pos[g], p_pos[p]] >= iou_threshold:
                match[g] = p
        # 2) Hungarian sobre o restante
        free_g = [g for g in g_list if g not in match]
        used_p = set(match.values())
        free_p = [p for p in p_list if p not in used_p]
        if free_g and free_p:
            sub = np.array([[iou[g_pos[g], p_pos[p]] for p in free_p] for g in free_g])
            sub = np.where(sub >= iou_threshold, sub, 0.0)
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] > 0:
                    match[free_g[r]] = free_p[c]

        for g, p in match.items():
            if g in last_pred_for_gt and last_pred_for_gt[g] != p:
                switches += 1
            last_pred_for_gt[g] = p

        tp += len(match)
        fn += len(g_list) - len(match)
        fp += len(p_list) - len(match)
        for g in g_list:
            tracked_seq[g].append(g in match)
        matches_per_frame[f] = match
        prev_match = match

    fragmentations = 0
    for seq in tracked_seq.values():
        was_tracked = False
        in_gap = False
        for t in seq:
            if t:
                if in_gap:
                    fragmentations += 1
                    in_gap = False
                was_tracked = True
            elif was_tracked:
                in_gap = True

    return {
        "id_switches": switches,
        "fragmentations": fragmentations,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tracked_seq": tracked_seq,
        "matches": matches_per_frame,
    }


def count_id_switches(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    iou_threshold: float = 0.5,
    num_frames: int | None = None,
) -> int:
    """Número de ID switches (ver docstring do módulo)."""
    frames = frame_range(tracks_pred, tracks_gt, num_frames)
    return int(clear_match(tracks_pred, tracks_gt, iou_threshold, frames)["id_switches"])


def count_fragmentations(
    tracks_pred: FrameDict,
    tracks_gt: FrameDict,
    iou_threshold: float = 0.5,
    num_frames: int | None = None,
) -> int:
    """Número de fragmentações (rastreado -> perdido -> rastreado, por identidade GT)."""
    frames = frame_range(tracks_pred, tracks_gt, num_frames)
    return int(clear_match(tracks_pred, tracks_gt, iou_threshold, frames)["fragmentations"])


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação completa de uma sequência
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_tracking_sequence(
    tracks_pred: list[dict[str, Any]],
    tracks_gt: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Avalia uma sequência e devolve IDF1, ID switches, fragmentações, contagens etc.

    GT com ``conf == 0`` é descartado antes da avaliação (convenção MOT17).
    ``count_error`` é o erro de contagem de identidades únicas: |n_pred_ids - n_gt_ids|
    (sinal em ``count_error_signed = n_pred_ids - n_gt_ids``).
    """
    tracks_gt = [d for d in tracks_gt if d.get("conf", 1.0) > 0]

    pred_frames, _ = parse_tracks_to_frames(tracks_pred)
    gt_frames, _ = parse_tracks_to_frames(tracks_gt)
    frames = sorted(set(pred_frames) | set(gt_frames))
    num_frames = max(frames) if frames else 0

    clear = clear_match(pred_frames, gt_frames, iou_threshold, frames)
    pred_to_gt, _, _ = match_global(pred_frames, gt_frames, iou_threshold, num_frames or None)
    idf1, det = compute_idf1(pred_frames, gt_frames, pred_to_gt, iou_threshold, num_frames or None)

    n_gt_boxes = clear["tp"] + clear["fn"]
    mota = 1.0 - (clear["fn"] + clear["fp"] + clear["id_switches"]) / n_gt_boxes if n_gt_boxes else 0.0

    mostly_tracked = mostly_lost = 0
    for seq in clear["tracked_seq"].values():
        ratio = sum(seq) / len(seq)
        mostly_tracked += ratio >= 0.8
        mostly_lost += ratio < 0.2
    n_gt = len(clear["tracked_seq"])

    n_gt_ids = det["n_gt_ids"]
    n_pred_ids = det["n_pred_ids"]
    return {
        "idf1": idf1,
        "idp": det["IDP"],
        "idr": det["IDR"],
        "id_switches": clear["id_switches"],
        "fragmentations": clear["fragmentations"],
        "n_gt_ids": n_gt_ids,
        "n_pred_ids": n_pred_ids,
        "count_error": abs(n_pred_ids - n_gt_ids),
        "count_error_signed": n_pred_ids - n_gt_ids,
        "mostly_tracked": mostly_tracked / n_gt if n_gt else 0.0,
        "mostly_lost": mostly_lost / n_gt if n_gt else 0.0,
        "num_frames": num_frames,
        "mota": mota,
        "IDTP": det["IDTP"],
        "IDFP": det["IDFP"],
        "IDFN": det["IDFN"],
        "FP": clear["fp"],
        "FN": clear["fn"],
    }
