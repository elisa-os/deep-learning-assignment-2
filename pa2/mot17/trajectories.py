"""Trajetórias do GT como dados de treino da RNN de movimento (Parte 2, Trilha A).

O GT do MOT17 é contínuo: um pedestre ocluído continua em todos os quadros da sua vida.
Uma trajetória só é quebrada se faltar algum quadro (p.ex. ``conf == 0`` em alguns quadros).
Cada pedaço contínuo é um ``Segment``; as janelas de treino saem desses segmentos.

Caixas aqui ficam em ``[cx, cy, w, h]`` (centro e tamanho, em pixels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pa2.mot17.loader import Sequence


@dataclass
class Segment:
    video: str
    track_id: int
    frames: np.ndarray       # (n,) quadros consecutivos
    boxes: np.ndarray        # (n, 4) cx, cy, w, h
    visibility: np.ndarray   # (n,)
    size: tuple[float, float]  # (largura, altura) da imagem

    def __len__(self) -> int:
        return len(self.frames)


def ltwh_to_cxcywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    return np.stack([b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2, b[..., 2], b[..., 3]], -1)


def cxcywh_to_ltwh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64)
    return np.stack([b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2, b[..., 2], b[..., 3]], -1)


def gt_segments(seq: Sequence, min_len: int = 3) -> list[Segment]:
    """Pedaços contínuos das trajetórias de pedestres do GT (``class == 1``, ``conf == 1``)."""
    gt = seq.gt_pedestrians()          # frame, id, l, t, w, h, vis
    size = (float(seq.info.im_width), float(seq.info.im_height))
    out: list[Segment] = []
    for tid in np.unique(gt[:, 1]):
        rows = gt[gt[:, 1] == tid]
        rows = rows[np.argsort(rows[:, 0])]
        cuts = np.nonzero(np.diff(rows[:, 0]) != 1)[0] + 1
        for part in np.split(rows, cuts):
            if len(part) >= min_len:
                out.append(Segment(seq.video, int(tid), part[:, 0].astype(int),
                                   ltwh_to_cxcywh(part[:, 2:6]), part[:, 6], size))
    return out


def load_segments(root, videos: list[str], detector: str = "SDP") -> list[Segment]:
    segs: list[Segment] = []
    for v in videos:
        segs += gt_segments(Sequence(root, v, detector))
    return segs


def _window(seg: Segment, start: int, stride: int, length: int):
    idx = start + stride * np.arange(length)
    idx = idx[idx < len(seg)]
    return idx


class WindowSampler:
    """Amostra janelas de ``L + 1`` caixas (L passos de previsão) dos segmentos.

    Cada janela é percorrida com um passo ``stride`` (Δt = stride quadros entre amostras
    consecutivas). Janelas curtas são completadas com ``valid = False``.
    """

    def __init__(self, segments: list[Segment], L: int, strides=(1,), seed: int = 0):
        self.segments = [s for s in segments if len(s) >= 3]
        if not self.segments:
            raise ValueError("nenhum segmento de GT com pelo menos 3 quadros")
        self.L = L
        self.strides = list(strides)
        self.rng = np.random.default_rng(seed)
        lens = np.array([len(s) for s in self.segments], dtype=np.float64)
        self.p = lens / lens.sum()                 # cada caixa do GT é igualmente provável
        self.total_boxes = int(lens.sum())

    def _pack(self, items: list[tuple[Segment, np.ndarray, int]]):
        B, L = len(items), self.L
        boxes = np.ones((B, L + 1, 4))
        valid = np.zeros((B, L + 1), dtype=bool)
        size = np.zeros((B, 2))
        dt = np.ones((B, L))
        for b, (seg, idx, stride) in enumerate(items):
            n = len(idx)
            boxes[b, :n] = seg.boxes[idx]
            boxes[b, n:] = seg.boxes[idx[-1]]
            valid[b, :n] = True
            size[b] = seg.size
            dt[b] = stride
        return boxes, valid, size, dt

    def sample(self, batch_size: int):
        items = []
        for _ in range(batch_size):
            seg = self.segments[self.rng.choice(len(self.segments), p=self.p)]
            stride = int(self.rng.choice(self.strides))
            if len(seg) < 1 + 2 * stride:            # segmento curto demais para esse passo
                stride = 1
            start = int(self.rng.integers(0, len(seg) - 2 * stride))   # deixa >= 3 amostras
            idx = _window(seg, start, stride, self.L + 1)
            items.append((seg, idx, stride))
        return self._pack(items)

    def fixed_windows(self, stride: int = 1):
        """Janelas sem sobreposição cobrindo todos os segmentos (para validação)."""
        items = []
        for seg in self.segments:
            for start in range(0, max(1, len(seg) - 2), self.L):
                idx = _window(seg, start, stride, self.L + 1)
                if len(idx) >= 3:
                    items.append((seg, idx, stride))
        return self._pack(items)
