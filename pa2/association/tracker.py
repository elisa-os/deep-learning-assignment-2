"""Rastreador ingênuo por IoU (baseline da Parte 1) — sem estado de movimento, sem aparência.

Regra de associação
    A cada quadro, cada track compara a *última caixa observada* com as detecções do quadro
    atual por IoU (matriz IoU tracks x detecções). Casamento ``hungarian`` (ótimo, maximiza
    a soma de IoU) ou ``greedy`` (pares por IoU decrescente). Só vale par com
    IoU >= ``iou_threshold`` (limiar fixo).

Gestão de tracks
    - Nascimento: toda detecção sem par vira uma track *tentativa*. Ela vira *confirmada*
      (e ganha um id novo) ao acumular ``min_hits`` observações consecutivas; uma tentativa
      que falha em um quadro é descartada. Com ``min_hits = 1`` toda detecção sem par
      ganha id na hora.
    - Morte: uma track confirmada sem observação por mais de ``max_age`` quadros morre.
      Durante a espera ela *não se move*: a comparação continua sendo com a última caixa
      observada (por isso um objeto que anda enquanto está ocluído não é recuperado).
    - Saída: só as tracks confirmadas e observadas no quadro. Não há caixa extrapolada.
    - Detecções com score < ``min_conf`` são ignoradas.

Nada aqui é rastreador de biblioteca: é a associação "ingênua" descrita no enunciado.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from pa2.metrics.tracking import iou_matrix


@dataclass
class _Track:
    box: np.ndarray          # última caixa observada [l, t, w, h]
    hits: int = 1
    since_update: int = 0
    track_id: int | None = None   # None = ainda tentativa


class IoUTracker:
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        min_hits: int = 1,
        method: str = "hungarian",
        min_conf: float = -np.inf,
    ):
        if method not in ("hungarian", "greedy"):
            raise ValueError(f"method deve ser 'hungarian' ou 'greedy', veio {method!r}")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = max(1, min_hits)
        self.method = method
        self.min_conf = min_conf
        self.reset()

    def reset(self) -> None:
        self.tracks: list[_Track] = []
        self._next_id = 1

    def _assign(self, iou: np.ndarray) -> list[tuple[int, int]]:
        """Pares (track, detecção) com IoU >= limiar."""
        if iou.size == 0:
            return []
        if self.method == "hungarian":
            rows, cols = linear_sum_assignment(-iou)
            return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= self.iou_threshold]
        pairs = [(iou[r, c], r, c) for r, c in zip(*np.nonzero(iou >= self.iou_threshold))]
        pairs.sort(key=lambda p: -p[0])
        used_r: set[int] = set()
        used_c: set[int] = set()
        out = []
        for _, r, c in pairs:
            if r not in used_r and c not in used_c:
                used_r.add(r)
                used_c.add(c)
                out.append((int(r), int(c)))
        return out

    def update(self, boxes: np.ndarray, scores: np.ndarray | None = None) -> list[tuple[int, int]]:
        """Processa um quadro. ``boxes`` (N, 4) em [l, t, w, h].

        Devolve ``[(track_id, índice_da_detecção)]`` das tracks confirmadas e observadas.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        if scores is not None:
            ok = np.nonzero(np.asarray(scores) >= self.min_conf)[0]
        else:
            ok = np.arange(len(boxes))

        track_boxes = (np.stack([t.box for t in self.tracks]) if self.tracks
                       else np.zeros((0, 4)))
        iou = iou_matrix(track_boxes, boxes[ok])
        pairs = self._assign(iou)

        matched_t = {r for r, _ in pairs}
        matched_d = {c for _, c in pairs}
        out: list[tuple[int, int]] = []

        for r, c in pairs:
            t = self.tracks[r]
            t.box = boxes[ok[c]].copy()
            t.hits += 1
            t.since_update = 0
            if t.track_id is None and t.hits >= self.min_hits:
                t.track_id = self._next_id
                self._next_id += 1
            if t.track_id is not None:
                out.append((t.track_id, int(ok[c])))

        survivors: list[_Track] = []
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                survivors.append(t)
                continue
            t.since_update += 1
            if t.track_id is not None and t.since_update <= self.max_age:
                survivors.append(t)           # confirmada: espera até max_age quadros
            # tentativa sem par: descartada

        for c in range(len(ok)):
            if c in matched_d:
                continue
            t = _Track(box=boxes[ok[c]].copy())
            if self.min_hits <= 1:
                t.track_id = self._next_id
                self._next_id += 1
                out.append((t.track_id, int(ok[c])))
            survivors.append(t)

        self.tracks = survivors
        return out


def track_sequence(
    dets: np.ndarray,
    num_frames: int,
    **tracker_kwargs,
) -> list[dict]:
    """Roda o ``IoUTracker`` sobre as detecções (M, 6: frame, l, t, w, h, score).

    Devolve as tracks no formato MOT (lista de dicts) para ``metrics.evaluate_tracking_sequence``.
    """
    tracker = IoUTracker(**tracker_kwargs)
    order = np.argsort(dets[:, 0], kind="stable")
    dets = dets[order]
    frames = dets[:, 0].astype(int)
    out: list[dict] = []
    for f in range(1, num_frames + 1):
        sel = np.nonzero(frames == f)[0]
        d = dets[sel]
        for tid, k in tracker.update(d[:, 1:5], d[:, 5]):
            l, t, w, h, s = d[k, 1:6]
            out.append({"frame": f, "id": tid, "bb_left": float(l), "bb_top": float(t),
                        "bb_width": float(w), "bb_height": float(h), "conf": float(s)})
    return out
