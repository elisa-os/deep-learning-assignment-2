"""Subamostragem temporal de uma sequência do MOT17 (taxa 1/k).

Para a taxa 1/k e a fase ``p`` (0 <= p < k) ficam os quadros ``f`` com ``(f - 1 - p) % k == 0``, renumerados
para 1..n'. O GT (pedestres e distratores) e as detecções são cortados nos MESMOS quadros, então o
vídeo subamostrado é uma sequência completa e consistente: o intervalo entre amostras vira Δt = k
quadros originais (k/fps segundos). Os ids do GT não mudam; objetos que só existiam entre dois quadros
mantidos desaparecem.

``SubsampledSequence`` imita o que o resto do código usa de ``Sequence`` (``name``, ``video``, ``info``,
``gt_pedestrians()``, ``gt_distractors()``), então ``evaluate_tracks`` e ``track_sequence_motion`` funcionam
sem mudança.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pa2.mot17.loader import Sequence


def phases(k: int) -> range:
    """Fases possíveis de uma subamostragem 1/k."""
    return range(k)


def cut_frames(arr: np.ndarray, k: int, phase: int = 0) -> np.ndarray:
    """Mantém as linhas cujo quadro (coluna 0, 1-based) cai na subamostragem 1/k e renumera o quadro."""
    if k < 1 or not 0 <= phase < k:
        raise ValueError(f"taxa 1/{k} com fase {phase} inválida")
    arr = np.asarray(arr)
    if k == 1 or len(arr) == 0:
        return arr.copy()
    off = arr[:, 0].astype(int) - 1 - phase
    keep = (off >= 0) & (off % k == 0)
    out = arr[keep].copy()
    out[:, 0] = off[keep] // k + 1
    return out


def subsample_dets(dets: np.ndarray, k: int, phase: int = 0) -> np.ndarray:
    """Detecções (M, 6: frame, l, t, w, h, score) da sequência subamostrada."""
    return cut_frames(dets, k, phase)


class SubsampledSequence:
    """Uma ``Sequence`` vista a 1/k da taxa original, com a fase ``phase``."""

    def __init__(self, seq: Sequence, k: int, phase: int = 0):
        if k < 1 or not 0 <= phase < k:
            raise ValueError(f"taxa 1/{k} com fase {phase} inválida")
        self.base, self.k, self.phase = seq, k, phase
        self.video = seq.video
        self.name = seq.name
        n = len(range(1 + phase, seq.info.seq_length + 1, k))
        self.info = replace(seq.info, seq_length=n, frame_rate=seq.info.frame_rate / k)
        self._ped = cut_frames(seq.gt_pedestrians(), k, phase)
        self._dis = cut_frames(seq.gt_distractors(), k, phase)

    def gt_pedestrians(self) -> np.ndarray:
        return self._ped

    def gt_distractors(self) -> np.ndarray:
        return self._dis
