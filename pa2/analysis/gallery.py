"""Galeria de falhas (Parte 4): escolha dos trechos, medidas de diagnóstico e figuras.

Sem as imagens do MOT17 (só as anotações), cada quadro é desenhado como um fundo vazio com as
caixas: GT (cinza, a identidade em foco em preto e grossa), tracks previstas coloridas pelo id
da track (tracejadas) e, em magenta pontilhado, a caixa que a RECORRÊNCIA previu para a track
que a identidade tinha antes do buraco. Embaixo: centro x, centro y e IoU entre a caixa prevista
pela recorrência e o GT ao longo do tempo, com o buraco sombreado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pa2.metrics.tracking import compute_iou
from pa2.mot17.loader import CAMERA_MOVING, Sequence

PALETTE = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#17becf", "#bcbd22", "#e377c2", "#ff7f0e",
           "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#31a354"]


def track_color(tid: int) -> str:
    return PALETTE[int(tid) % len(PALETTE)]


def episode_measures(seq: Sequence, ep: pd.Series, assign, gates, D: np.ndarray, match_gt_det: dict,
                     gt_frames: dict) -> dict:
    """Medidas do trecho: o que a recorrência esperava e o que aconteceu no reaparecimento."""
    s, N = int(ep.start), int(ep.length)
    gid = int(ep.gt_id)
    tid = None if pd.isna(ep.pred_before) else int(ep.pred_before)
    after = None if pd.isna(ep.pred_after) else int(ep.pred_after)
    end = s + N                                              # primeiro quadro depois do buraco
    gtb = lambda f: gt_frames.get((f, gid))
    ious = {}
    for f in range(s - 3, min(seq.info.seq_length, end + 3) + 1):
        g = gates[f - 1].get(tid) if tid is not None else None
        b = gtb(f)
        if g is not None and b is not None:
            ious[f] = compute_iou(g, b)
    pre = [gtb(f) for f in range(max(1, s - 5), s) if gtb(f) is not None]
    speed = 0.0
    if len(pre) >= 2:
        c = np.array([[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in pre])
        speed = float(np.linalg.norm(c[-1] - c[0]) / (len(c) - 1) / max(np.mean([b[2] for b in pre]), 1))
    first_below = next((f - s for f in sorted(ious) if f >= s and ious[f] < 0.3), None)
    return {"gt_id": gid, "video": seq.video, "start": s, "length": N, "outcome": ep.outcome,
            "pred_before": tid, "pred_after": after, "mean_visibility": float(ep.mean_visibility),
            "camera": "móvel" if CAMERA_MOVING.get(seq.video) else "parada",
            "speed_boxwidths_per_frame": speed,
            "gate_iou_at_end": ious.get(end), "gate_iou_min_in_gap": min([v for f, v in ious.items() if s <= f < end],
                                                                         default=None),
            "frames_until_gate_iou_below_0.3": first_below,
            "track_alive_at_end": bool(tid is not None and tid in gates[min(end, seq.info.seq_length) - 1])}


def gt_boxes_by_frame(seq: Sequence) -> dict:
    g = seq.gt_pedestrians()
    return {(int(f), int(i)): np.array([l, t, w, h]) for f, i, l, t, w, h in g[:, :6]}


def pick_failures(eps, meas: pd.DataFrame) -> dict[str, int]:
    """Escolhe 3 trechos por regras fixas (índices de ``meas``; vídeos de validação, modelo final):

    - ``oclusao_longa``: câmera parada, buraco de 15 a 30 quadros (dentro de ``max_age``, para que a
      track só possa morrer por falta de memória e não pela regra), visibilidade média < 0,3, sem manter
      o id e com a track original ainda viva no fim; o de MENOR IoU entre a caixa prevista e o GT;
    - ``buraco_curto_camera_movel``: câmera móvel, buraco de até 10 quadros que terminou em id novo e
      com a track original viva; o de menor IoU entre a caixa prevista e o GT no fim;
    - ``troca_entre_pessoas``: o id que assumiu a identidade já era de OUTRA pessoa pouco antes (swap),
      a caixa prevista pela recorrência ainda estava certa (IoU >= 0,3) e o buraco tem >= 5 quadros; o
      mais longo, de preferência em outro vídeo que o trecho anterior.
    """
    out: dict[str, int] = {}
    bad = meas[meas.outcome != "kept"]
    c = bad[(bad.camera == "parada") & bad.length.between(15, 30) & (bad.mean_visibility < 0.3)
            & bad.track_alive_at_end & bad.gate_iou_at_end.notna()]
    if len(c):
        out["oclusao_longa"] = int(c.sort_values("gate_iou_at_end").index[0])
    c = bad[(bad.camera == "móvel") & (bad.length <= 10) & (bad.outcome == "switched")
            & bad.track_alive_at_end & bad.gate_iou_at_end.notna()]
    if len(c):
        out["buraco_curto_camera_movel"] = int(c.sort_values("gate_iou_at_end").index[0])
    c = bad[bad.swap_partner.notna() & (bad.gate_iou_at_end >= 0.3) & (bad.length >= 5)]
    if "buraco_curto_camera_movel" in out:
        other = c[c.video != meas.loc[out["buraco_curto_camera_movel"], "video"]]
        c = other if len(other) else c
    if len(c):
        out["troca_entre_pessoas"] = int(c.sort_values("length", ascending=False).index[0])
    return out


def draw_failure(seq: Sequence, m: dict, assign, gates, D, gt_frames: dict, gt_all: np.ndarray, path,
                 title: str, caption: str, save_figure) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    s, N, gid, tid = m["start"], m["length"], m["gt_id"], m["pred_before"]
    T = seq.info.seq_length
    f0, f1 = max(1, s - 3), min(T, s + N + 3)
    shown = sorted({int(round(x)) for x in np.linspace(f0, f1, 6)} | {s, min(T, s + N)})[:7]
    W, H = seq.info.im_width, seq.info.im_height

    # recorte em torno da trajetória da identidade em foco
    foc = np.array([gt_frames[(f, gid)] for f in range(f0, f1 + 1) if (f, gid) in gt_frames])
    x0, x1 = (foc[:, 0].min(), (foc[:, 0] + foc[:, 2]).max())
    y0, y1 = (foc[:, 1].min(), (foc[:, 1] + foc[:, 3]).max())
    mx, my = max(120, 1.2 * (x1 - x0)), max(120, 0.8 * (y1 - y0))
    xl, xr = max(0, x0 - mx), min(W, x1 + mx)
    yt, yb = max(0, y0 - my), min(H, y1 + my)

    rows = {int(f): [] for f in shown}
    for f in shown:
        for tr, r in assign[f - 1]:
            rows[f].append((tr, D[r, 1:5]))

    fig = plt.figure(figsize=(2.6 * len(shown), 8.4))
    gs = fig.add_gridspec(2, len(shown), height_ratios=[1.35, 1])
    for j, f in enumerate(shown):
        ax = fig.add_subplot(gs[0, j])
        ax.set_xlim(xl, xr); ax.set_ylim(yb, yt); ax.set_aspect("equal"); ax.set_facecolor("#f5f5f5")
        ax.set_xticks([]); ax.set_yticks([])
        for g in gt_all[gt_all[:, 0] == f]:
            gi, (l, t, w, h) = int(g[1]), g[2:6]
            if l > xr or l + w < xl or t > yb or t + h < yt:
                continue
            foc_g = gi == gid
            ax.add_patch(Rectangle((l, t), w, h, fill=False, ec="k" if foc_g else "#9a9a9a",
                                   lw=2.4 if foc_g else 0.8))
            if foc_g:
                ax.text(l, t - 4, f"GT {gi}", fontsize=7, color="k", va="bottom", clip_on=True)
        for tr, (l, t, w, h) in rows[f]:
            if l > xr or l + w < xl or t > yb or t + h < yt:
                continue
            ax.add_patch(Rectangle((l, t), w, h, fill=False, ec=track_color(tr), lw=1.6, ls="--"))
            ax.text(l + w, t + h + 4, f"T{tr}", fontsize=7, color=track_color(tr), va="top", ha="right",
                    clip_on=True)
        gb = gates[f - 1].get(tid) if tid is not None else None
        if gb is not None:
            ax.add_patch(Rectangle((gb[0], gb[1]), gb[2], gb[3], fill=False, ec="magenta", lw=1.8, ls=":"))
        phase = "antes" if f < s else ("buraco" if f < s + N else "depois")
        ax.set_title(f"quadro {f} ({phase})", fontsize=8,
                     color="#b00020" if phase == "buraco" else "k")

    # séries: centro x, centro y, IoU(caixa prevista pela recorrência, GT)
    fr = np.arange(f0, f1 + 1)
    cx = [(f, gt_frames[(f, gid)][0] + gt_frames[(f, gid)][2] / 2, gt_frames[(f, gid)][1] + gt_frames[(f, gid)][3] / 2)
          for f in fr if (f, gid) in gt_frames]
    gx = [(f, gates[f - 1][tid][0] + gates[f - 1][tid][2] / 2, gates[f - 1][tid][1] + gates[f - 1][tid][3] / 2,
           compute_iou(gates[f - 1][tid], gt_frames[(f, gid)]) if (f, gid) in gt_frames else np.nan)
          for f in fr if tid is not None and tid in gates[f - 1]]
    spans = [(0, 2), (2, 4), (4, len(shown))]
    for (a, b), what in zip(spans, ("x", "y", "iou")):
        ax = fig.add_subplot(gs[1, a:b])
        ax.axvspan(s, s + N - 1, color="#ffd6d6", alpha=.7, label="buraco (sem detecção)")
        if what in ("x", "y"):
            i = 1 if what == "x" else 2
            ax.plot([c[0] for c in cx], [c[i] for c in cx], "k-", label="GT")
            ax.plot([g[0] for g in gx], [g[i] for g in gx], "m:", lw=2, label="caixa prevista pela recorrência")
            ax.set_ylabel(f"centro {what} (px)")
        else:
            ax.plot([g[0] for g in gx], [g[3] for g in gx], "m-", lw=2)
            ax.axhline(0.3, color="k", ls=":", lw=1)
            ax.text(f0, 0.31, "limiar de IoU da associação (0,3)", fontsize=7)
            ax.set_ylim(0, 1.02); ax.set_ylabel("IoU previsão × GT")
        ax.set_xlabel("quadro"); ax.grid(alpha=.3)
        if what == "x":
            ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11, y=0.995)
    fig.text(0.01, 0.005, caption, fontsize=8, va="bottom", ha="left", wrap=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.12, wspace=0.12, hspace=0.35)
    save_figure(fig, path, dpi=110)
