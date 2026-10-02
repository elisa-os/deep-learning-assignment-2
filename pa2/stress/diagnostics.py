"""Diagnósticos do "por quê" da queda de taxa de quadros, em trajetórias do GT (baratos).

1. ``consecutive_iou``: IoU entre as caixas do MESMO objeto em amostras consecutivas a stride k. Quando cai,
   a associação por IoU (e a caixa parada) perde o objeto: é o efeito geométrico, que independe do modelo.
2. ``onestep_displacement_regression``: o deslocamento que o modelo prevê para a próxima amostra (depois de ver
   ``context`` amostras com ruído de detector) contra o deslocamento verdadeiro. A inclinação (regressão pela
   origem) é 1 para um modelo que acompanha o movimento e < 1 se ele encolhe o deslocamento.
3. ``dt_sensitivity``: o quanto a saída do modelo muda quando o recurso Δt passa de 1 para k, com a MESMA
   entrada. Se for ~0, alimentar Δt a um modelo que nunca o viu variar não tem como ajudar.

As amostras de um stride k são quadros originais espaçados de k: é o que o rastreador vê num vídeo a 1/k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa2.mot17.trajectories import cxcywh_to_ltwh


def _pair_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU linha a linha entre pilhas de caixas [l, t, w, h]."""
    iw = np.clip(np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def consecutive_iou(segments, strides=(1, 2, 5)) -> pd.DataFrame:
    """Para cada stride, a distribuição do IoU entre amostras consecutivas do mesmo objeto (GT)."""
    rows = []
    for k in strides:
        ious = []
        for s in segments:
            if len(s) > k:
                a, b = cxcywh_to_ltwh(s.boxes[:-k]), cxcywh_to_ltwh(s.boxes[k:])
                ious.append(_pair_iou(a, b))
        v = np.concatenate(ious)
        rows.append({"stride": k, "n_pares": len(v), "iou_medio": float(v.mean()),
                     "iou_mediano": float(np.median(v)), "frac_iou_abaixo_0.3": float((v < 0.3).mean()),
                     "frac_iou_abaixo_0.5": float((v < 0.5).mean())})
    return pd.DataFrame(rows)


def _stride_groups(segments, k: int, context: int, n: int, sigma, rng: np.random.Generator):
    """Janelas de ``context + 1`` amostras espaçadas de k quadros, agrupadas por tamanho de imagem.

    Devolve ``[(size, obs_ltwh (m, context, 4), gt_ltwh (m, context + 1, 4))]``; ``obs`` são as ``context``
    primeiras caixas com o ruído de detector da Parte 2 e ``gt`` a janela inteira sem ruído."""
    need = k * context + 1
    eligible = [s for s in segments if len(s) >= need]
    if not eligible:
        return []
    lens = np.array([len(s) - need + 1 for s in eligible], dtype=float)
    picks = [(eligible[i], int(rng.integers(0, int(lens[i])))) for i in
             rng.choice(len(eligible), size=n, p=lens / lens.sum())]
    sg = np.asarray(sigma)
    out = []
    for size in sorted({p[0].size for p in picks}):
        idx = [i for i, p in enumerate(picks) if p[0].size == size]
        gt = np.stack([picks[i][0].boxes[picks[i][1]: picks[i][1] + need: k] for i in idx])   # (m, context+1, 4)
        nz = rng.standard_normal((len(idx), context, 4)) * sg
        c = gt[:, :context]
        obs = np.stack([c[..., 0] + nz[..., 0] * c[..., 2], c[..., 1] + nz[..., 1] * c[..., 3],
                        c[..., 2] * np.exp(nz[..., 2]), c[..., 3] * np.exp(nz[..., 3])], -1)
        out.append((size, cxcywh_to_ltwh(obs), cxcywh_to_ltwh(gt)))
    return out


def _predict_next(motion, obs: np.ndarray) -> np.ndarray:
    """Alimenta o modelo de movimento com as ``context`` observações e devolve a previsão da próxima caixa."""
    m, context = obs.shape[:2]
    states, prev = [None] * m, [None] * m
    pred = None
    for j in range(context):
        states, pred = motion.step(states, obs[:, j], prev, np.ones(m, bool))
        prev = list(obs[:, j])
    return pred


def onestep_displacement_regression(make_motion, segments, sigma, strides=(1, 2, 5), context: int = 8,
                                    n: int = 1500, seed: int = 1) -> pd.DataFrame:
    """Deslocamento horizontal previsto × verdadeiro para a próxima amostra, em larguras da caixa.

    ``make_motion(size, k)`` devolve o modelo de movimento a usar no stride k (assim o chamador decide que Δt
    alimentar). O previsto é ``pred - última observação`` (a saída do modelo, sem o ruído da âncora) e o
    verdadeiro é ``gt_próxima - gt_última``. ``slope`` é a regressão pela origem (1 = acompanha, < 1 = encolhe).
    """
    rows = []
    for k in strides:
        rng = np.random.default_rng(seed + k)
        dp, dt = [], []
        for size, obs, gt in _stride_groups(segments, k, context, n, sigma, rng):
            pred = _predict_next(make_motion(size, k), obs)
            w = obs[:, -1, 2]
            dp.append((pred[:, 0] + pred[:, 2] / 2 - (obs[:, -1, 0] + obs[:, -1, 2] / 2)) / w)
            dt.append(((gt[:, -1, 0] + gt[:, -1, 2] / 2) - (gt[:, -2, 0] + gt[:, -2, 2] / 2)) / w)
        dp, dt = np.concatenate(dp), np.concatenate(dt)
        moving = np.abs(dt) > 0.05
        rows.append({"stride": k, "n": len(dp), "slope": float((dp * dt).sum() / (dt * dt).sum()),
                     "corr": float(np.corrcoef(dp, dt)[0, 1]),
                     "mean_abs_true": float(np.abs(dt).mean()), "mean_abs_pred": float(np.abs(dp).mean()),
                     "wrong_sign": float((np.sign(dp[moving]) != np.sign(dt[moving])).mean())})
    return pd.DataFrame(rows)


def dt_sensitivity(make_motion, segments, sigma, strides=(2, 5), context: int = 8, n: int = 1000,
                   seed: int = 1) -> pd.DataFrame:
    """Mudança relativa na previsão ao trocar o recurso Δt de 1 para k, com a mesma entrada.

    ``make_motion(size, dt)`` devolve o modelo com o Δt dado. ``rel_change`` é a mediana de
    ``|deslocamento(dt=k) - deslocamento(dt=1)| / |deslocamento(dt=1)|`` (só onde o deslocamento passa de 0,05 largura)."""
    rows = []
    for k in strides:
        rng = np.random.default_rng(seed + k)
        rel, d1s, dks = [], [], []
        for size, obs, _ in _stride_groups(segments, k, context, n, sigma, rng):
            w = obs[:, -1, 2]
            cx = lambda p: (p[:, 0] + p[:, 2] / 2 - (obs[:, -1, 0] + obs[:, -1, 2] / 2)) / w
            d1 = cx(_predict_next(make_motion(size, 1.0), obs))
            dk = cx(_predict_next(make_motion(size, float(k)), obs))
            ok = np.abs(d1) > 0.05
            rel.append(np.abs(dk[ok] - d1[ok]) / np.abs(d1[ok]))
            d1s.append(d1)
            dks.append(dk)
        rel = np.concatenate(rel)
        d1, dk = np.concatenate(d1s), np.concatenate(dks)
        rows.append({"stride": k, "n": len(d1), "rel_change_mediana": float(np.median(rel)),
                     "rel_change_media": float(rel.mean()), "corr_dt1_dtk": float(np.corrcoef(d1, dk)[0, 1])})
    return pd.DataFrame(rows)
