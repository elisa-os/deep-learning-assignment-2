"""Reconexão de identidades depois de um buraco de rastreamento (análise da Parte 2).

Para cada identidade verdadeira, a vida inteira é uma sequência de quadros; em cada quadro a
identidade está casada com um id previsto (matching CLEAR-MOT de ``tracking._clear_match``)
ou não. Um **buraco** é uma sequência máxima de quadros sem casamento *entre* dois quadros
casados. O desfecho do buraco é:

- ``kept``: o mesmo id previsto antes e depois (a memória sobreviveu ao buraco);
- ``switched``: outro id depois (a track morreu e uma nova nasceu = id duplicado / ID switch);
- ``lost``: a identidade não volta a ser casada até o fim da sua vida.

Um buraco pode vir de oclusão ou de detecção perdida; ``mean_visibility`` (campo do GT) diz
qual. A duração do buraco é a do ``min_hits``-inclusive: quadros sem casamento, contados no GT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa2.metrics.detection import drop_ignored_detections
from pa2.metrics.tracking import _clear_match, _frame_range, _parse_tracks_to_frames
from pa2.mot17.loader import Sequence, to_mot_records

BUCKETS = ((1, 5), (6, 15), (16, 30), (31, 10**9))
BUCKET_NAMES = ("1-5", "6-15", "16-30", "31+")


def bucket_of(k: int) -> str:
    for (lo, hi), name in zip(BUCKETS, BUCKET_NAMES):
        if lo <= k <= hi:
            return name
    raise ValueError(k)


def gap_episodes(seq: Sequence, tracks: list[dict], iou_threshold: float = 0.5) -> pd.DataFrame:
    """Um registro por buraco: ``gt_id, start, length, outcome, mean_visibility``."""
    gt = seq.gt_pedestrians()
    if tracks:
        arr = np.array([[t["frame"], t["id"], t["bb_left"], t["bb_top"], t["bb_width"],
                         t["bb_height"]] for t in tracks], dtype=np.float64)
        _, keep = drop_ignored_detections(arr, gt, seq.gt_distractors(), iou_threshold,
                                          boxes_cols=slice(2, 6))
        tracks = [t for t, k in zip(tracks, keep) if k]
    pred_frames, _ = _parse_tracks_to_frames(tracks)
    gt_frames, _ = _parse_tracks_to_frames(to_mot_records(gt))
    frames = _frame_range(pred_frames, gt_frames, None)
    matches = _clear_match(pred_frames, gt_frames, iou_threshold, frames)["matches"]

    vis = {(int(f), int(i)): v for f, i, v in zip(gt[:, 0], gt[:, 1], gt[:, 6])}
    life: dict[int, list[int]] = {}
    for f in sorted(gt_frames):
        for g in gt_frames[f]:
            life.setdefault(g, []).append(f)

    rows = []
    for g, fs in life.items():
        seq_ids = [matches.get(f, {}).get(g) for f in fs]
        last_id, gap_start = None, None
        for i, pid in enumerate(seq_ids):
            if pid is None:
                if last_id is not None and gap_start is None:
                    gap_start = i
                continue
            if gap_start is not None:
                rows.append(_row(g, fs, gap_start, i, vis, "kept" if pid == last_id else "switched"))
                gap_start = None
            last_id = pid
        if gap_start is not None:
            rows.append(_row(g, fs, gap_start, len(fs), vis, "lost"))
    return pd.DataFrame(rows, columns=["gt_id", "start", "length", "outcome", "mean_visibility"])


def _row(g, fs, a, b, vis, outcome) -> dict:
    gap = fs[a:b]
    return {"gt_id": g, "start": gap[0], "length": len(gap), "outcome": outcome,
            "mean_visibility": float(np.mean([vis[(f, g)] for f in gap]))}


def reconnection_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Por método e faixa de duração: nº de buracos, % que mantêm o id e % que trocam."""
    ep = episodes.assign(bucket=episodes["length"].map(bucket_of))
    rows = []
    for (m, b), g in ep.groupby(["method", "bucket"]):
        n = len(g)
        rows.append({"method": m, "gap": b, "n": n, "kept": (g.outcome == "kept").mean(),
                     "switched": (g.outcome == "switched").mean(), "lost": (g.outcome == "lost").mean(),
                     "mean_visibility": g.mean_visibility.mean()})
    t = pd.DataFrame(rows)
    t["gap"] = pd.Categorical(t["gap"], BUCKET_NAMES, ordered=True)
    return t.sort_values(["method", "gap"]).reset_index(drop=True)
