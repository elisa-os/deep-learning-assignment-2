"""Degradação proposital das detecções (Parte 5, teste alternativo: qualidade do detector).

Recebe detecções no formato do projeto (M, 6): frame, left, top, w, h, score e as estraga de três
jeitos independentes, como no enunciado:

- ``drop``: descarta cada detecção com probabilidade p (falso negativo);
- ``noise``: ruído nas caixas, **relativo ao tamanho da caixa** (não depende da resolução):
  ``left += N(0, noise)·w``, ``top += N(0, noise)·h``, ``w *= exp(N(0, noise))``, ``h *= exp(N(0, noise))``;
- ``fp``: falsos positivos, em média ``fp`` por quadro (Poisson); cada um copia o tamanho de uma detecção
  real sorteada, tem posição uniforme na imagem e score uniforme em [``score_min``, 1] (ou seja,
  passam pelo limiar de score do rastreador: é o caso mais difícil para ele).
"""

from __future__ import annotations

import numpy as np


def degrade_detections(dets: np.ndarray, *, drop: float = 0.0, noise: float = 0.0, fp: float = 0.0,
                       n_frames: int, image_size: tuple[int, int], score_min: float,
                       rng: np.random.Generator) -> np.ndarray:
    d = np.array(dets, dtype=float, copy=True)
    if drop > 0 and len(d):
        d = d[rng.random(len(d)) >= drop]
    if noise > 0 and len(d):
        w, h = d[:, 3].copy(), d[:, 4].copy()
        d[:, 1] += rng.normal(0, noise, len(d)) * w
        d[:, 2] += rng.normal(0, noise, len(d)) * h
        d[:, 3] = w * np.exp(rng.normal(0, noise, len(d)))
        d[:, 4] = h * np.exp(rng.normal(0, noise, len(d)))
    if fp > 0 and len(dets):
        W, H = image_size
        rows = []
        for f in range(1, n_frames + 1):
            for _ in range(int(rng.poisson(fp))):
                src = dets[rng.integers(len(dets))]
                w, h = src[3], src[4]
                rows.append([f, rng.uniform(0, max(1.0, W - w)), rng.uniform(0, max(1.0, H - h)), w, h,
                             rng.uniform(score_min, 1.0)])
        if rows:
            extra = np.zeros((len(rows), d.shape[1])) if len(d) else np.zeros((len(rows), 6))
            extra[:, :6] = np.asarray(rows)
            d = np.vstack([d, extra]) if len(d) else extra
    return d[np.argsort(d[:, 0], kind="stable")] if len(d) else d
