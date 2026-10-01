"""Rastreador com modelo de movimento (Parte 2): a mesma gestão de tracks da Parte 1.

Só muda *onde a track espera encontrar o objeto*: em vez da última caixa observada, a
associação compara as detecções com a caixa **prevista** pelo modelo de movimento
(``pa2/association/motion.py``). Nascimento, morte, ``min_hits``, ``max_age``, limiar de
IoU e o casamento guloso/Hungarian são os de ``IoUTracker`` (herdados).

Sob oclusão a track continua viva: o modelo avança sem observação (alimentado com a
própria previsão) e a caixa prevista segue andando até a track casar de novo ou passar
de ``max_age`` quadros. Com ``StaticMotion`` este rastreador é idêntico ao da Parte 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pa2.association.tracker import IoUTracker, _Track
from pa2.metrics.tracking import iou_matrix


@dataclass
class _MTrack(_Track):
    pred: np.ndarray = None      # caixa esperada no próximo quadro
    fed: np.ndarray = None       # última caixa que entrou no modelo
    prev: np.ndarray | None = None
    state: object = None
    last_observed: bool = True


class MotionTracker(IoUTracker):
    def __init__(self, motion, **kwargs):
        self.motion = motion
        super().__init__(**kwargs)

    def _advance(self, tracks: list[_MTrack]) -> None:
        """Um passo do modelo de movimento para todas as tracks (em lote)."""
        if not tracks:
            return
        states, pred = self.motion.step(
            [t.state for t in tracks], np.stack([t.fed for t in tracks]),
            [t.prev for t in tracks], np.array([t.last_observed for t in tracks]))
        for t, s, p in zip(tracks, states, pred):
            t.state, t.pred = s, p

    def update(self, boxes: np.ndarray, scores: np.ndarray | None = None) -> list[tuple[int, int]]:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        ok = (np.nonzero(np.asarray(scores) >= self.min_conf)[0] if scores is not None
              else np.arange(len(boxes)))

        pred_boxes = (np.stack([t.pred for t in self.tracks]) if self.tracks else np.zeros((0, 4)))
        pairs = self._assign(iou_matrix(pred_boxes, boxes[ok]))
        matched_t = {r for r, _ in pairs}
        matched_d = {c for _, c in pairs}
        out: list[tuple[int, int]] = []

        for r, c in pairs:
            t = self.tracks[r]
            t.box = boxes[ok[c]].copy()
            t.prev, t.fed, t.last_observed = t.fed, t.box, True
            t.hits += 1
            t.since_update = 0
            if t.track_id is None and t.hits >= self.min_hits:
                t.track_id = self._next_id
                self._next_id += 1
            if t.track_id is not None:
                out.append((t.track_id, int(ok[c])))

        survivors: list[_MTrack] = []
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                survivors.append(t)
                continue
            t.since_update += 1
            if t.track_id is not None and t.since_update <= self.max_age:
                t.prev, t.fed, t.last_observed = t.fed, t.pred, False   # roda sem observação
                survivors.append(t)

        for c in range(len(ok)):
            if c in matched_d:
                continue
            b = boxes[ok[c]].copy()
            t = _MTrack(box=b, fed=b, pred=b)
            if self.min_hits <= 1:
                t.track_id = self._next_id
                self._next_id += 1
                out.append((t.track_id, int(ok[c])))
            survivors.append(t)

        self._advance(survivors)
        self.tracks = survivors
        return out


def track_sequence_motion(dets: np.ndarray, num_frames: int, motion, **tracker_kwargs) -> list[dict]:
    """Como ``track_sequence`` (Parte 1), com o modelo de movimento ``motion``."""
    tracker = MotionTracker(motion, **tracker_kwargs)
    order = np.argsort(dets[:, 0], kind="stable")
    dets = dets[order]
    frames = dets[:, 0].astype(int)
    out: list[dict] = []
    for f in range(1, num_frames + 1):
        d = dets[np.nonzero(frames == f)[0]]
        for tid, k in tracker.update(d[:, 1:5], d[:, 5]):
            l, t, w, h, s = d[k, 1:6]
            out.append({"frame": f, "id": tid, "bb_left": float(l), "bb_top": float(t),
                        "bb_width": float(w), "bb_height": float(h), "conf": float(s)})
    return out
