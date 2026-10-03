"""Parte 5 — teste de estresse alternativo (EXTRA): qualidade do detector. Sem retreinar.

A Parte 5 entregue é a queda de taxa de quadros. Este módulo faz, como extra, o outro teste do enunciado:
degradar as detecções (descartar p%, ruído nas caixas, falsos positivos) em 3 intensidades e perguntar se
o modelo temporal **absorve ou amplifica** a falha do detector, reportando **mAP e IDF1 juntos**.

Hipóteses escritas antes de rodar (ver ``RELATORY_PART5.md`` §8):
- H1: o IDF1 cai com todas as degradações, e cai mais com falsos positivos e descarte do que com ruído.
- H2: a RNN (modelo final, Δt = 1) **absorve** parte do descarte (a caixa prevista sustenta a track nos
  quadros sem detecção), mas **não** os falsos positivos (eles passam pelo limiar de score).
- H3: o ruído alto prejudica a RNN mais que a caixa parada (a rede foi treinada com ruído moderado).

Desenho: 3 intensidades combinadas (leve/média/forte) + os 3 fatores isolados na intensidade forte; 3 seeds de
degradação; mesmos 3 métodos de movimento (caixa parada, velocidade constante, RNN final), mesma regra de
associação das Partes 1-5. ``uv run pa2 5b`` ⇒ ``outputs/final_parte5/parte5b_*``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.ablation import build_context
from pa2.association.motion_tracker import track_sequence_motion
from pa2.config import Config
from pa2.metrics.detection import evaluate_detections
from pa2.models import load_checkpoint
from pa2.mot17.evaluate import evaluate_tracks
from pa2.part5 import make_motion
from pa2.stress import degrade_detections
from pa2.utils.visualize import save_figure

METHODS = ("static", "const_vel", "rnn_dt1")
LABELS = {"static": "caixa parada (Parte 1)", "const_vel": "velocidade constante", "rnn_dt1": "RNN final"}
COLORS = {"static": "tab:gray", "const_vel": "tab:orange", "rnn_dt1": "tab:red"}
SEEDS = (0, 1, 2)
# nome -> (descarte, ruído relativo, falsos positivos por quadro). O ruído relativo máximo (10% da caixa) já leva
# o AP50 do vídeo 09 de 0,64 a 0,57 e o mAP de 0,46 a 0,19; com 20% o detector colapsa (AP50 0,17), o que não é
# um teste útil. Para comparação, o ruído de treino da RNN é ~2,5-8% (``obs_noise``).
CONDITIONS = {
    "original": (0.0, 0.0, 0.0),
    "leve": (0.10, 0.03, 0.5),
    "média": (0.25, 0.06, 1.5),
    "forte": (0.40, 0.10, 3.0),
    "só descarte (40%)": (0.40, 0.0, 0.0),
    "só ruído (10%)": (0.0, 0.10, 0.0),
    "só falsos positivos (3/q)": (0.0, 0.0, 3.0),
}


def _eval_cell(ctx, model, cond: str, seed: int, video: str) -> list[dict]:
    drop, noise, fp = CONDITIONS[cond]
    seq = ctx.src.seqs[video]
    rng = np.random.default_rng(1000 * seed + int(video))
    d = degrade_detections(ctx.src.dets[video], drop=drop, noise=noise, fp=fp, n_frames=seq.info.seq_length,
                           image_size=(seq.info.im_width, seq.info.im_height), score_min=ctx.min_conf, rng=rng)
    det = evaluate_detections(d, seq.gt_pedestrians(), None, full=True)
    op = evaluate_detections(d, seq.gt_pedestrians(), None, ctx.min_conf, full=False)
    rows = []
    for m in METHODS:
        tracks = track_sequence_motion(d, seq.info.seq_length, make_motion(m, model, seq, 1),
                                       min_conf=ctx.min_conf, **ctx.assoc)
        r = evaluate_tracks(seq, tracks)
        rows.append({"condition": cond, "seed": seed, "video": video, "split": ctx.splits[video], "method": m,
                     "drop": drop, "noise": noise, "fp": fp, "mAP": det["map"], "AP50": det["ap50"],
                     "precision": op["precision"], "recall": op["recall"], "det_F1": op["f1"],
                     "IDF1": r["idf1"], "ID_switches": r["id_switches"], "n_gt_ids": r["n_gt_ids"],
                     "n_pred_ids": r["n_pred_ids"], "ids_pred/gt": r["n_pred_ids"] / max(1, r["n_gt_ids"]),
                     "IDsw/id": r["id_switches"] / max(1, r["n_gt_ids"]), "MOTA": r["mota"]})
    return rows


def summarize(raw: pd.DataFrame, subset: str) -> pd.DataFrame:
    d = raw if subset == "todos" else raw[raw.split == "val"]
    # média entre vídeos dentro de cada seed; depois média ± desvio entre seeds
    per_seed = d.groupby(["condition", "method", "seed"], observed=True)[
        ["mAP", "AP50", "precision", "recall", "det_F1", "IDF1", "ID_switches", "ids_pred/gt", "IDsw/id"]].mean().reset_index()
    g = per_seed.groupby(["condition", "method"], observed=True)
    out = g.mean(numeric_only=True).drop(columns="seed").add_suffix("_mean").join(
        g.std(numeric_only=True, ddof=1).drop(columns="seed").add_suffix("_std")).reset_index()
    order = {c: i for i, c in enumerate(CONDITIONS)}
    out["_o"] = out.condition.map(order)
    return out.sort_values(["_o", "method"]).drop(columns="_o").reset_index(drop=True)


def _plot(summ: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = list(CONDITIONS)
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    # (a) o detector: mAP por condição
    ax = axes[0]
    det = summ[summ.method == "rnn_dt1"].set_index("condition").loc[conds]
    ax.bar(x, det["mAP_mean"], color="tab:blue", alpha=.8, label="mAP do detector")
    ax.plot(x, det["det_F1_mean"], "ko--", ms=4, label="F1 do detector (limiar do rastreador)")
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" (", "\n(") for c in conds], rotation=40, ha="right", fontsize=8)
    ax.set_title("o que o detector entrega (mAP e F1)", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    # (b) o rastreamento: IDF1 por condição e método
    ax = axes[1]
    for i, m in enumerate(METHODS):
        s = summ[summ.method == m].set_index("condition").loc[conds]
        ax.errorbar(x + (i - 1) * 0.12, s["IDF1_mean"], yerr=np.nan_to_num(s["IDF1_std"]), marker="o", ms=4, linestyle="none",
                    capsize=2, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels([c.replace(" (", "\n(") for c in conds], rotation=40, ha="right", fontsize=8)
    ax.set_title("o que o rastreamento entrega (IDF1)", fontsize=10); ax.set_ylabel("IDF1"); ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    # (c) mAP × IDF1 (cada ponto é uma condição)
    ax = axes[2]
    for m in METHODS:
        s = summ[summ.method == m]
        ax.plot(s["mAP_mean"], s["IDF1_mean"], "o", color=COLORS[m], label=LABELS[m], alpha=.8)
    ax.set_xlabel("mAP do detector (degradado)"); ax.set_ylabel("IDF1"); ax.grid(alpha=.3)
    ax.set_title("mAP × IDF1 juntos (um ponto por condição)", fontsize=10); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    save_figure(fig, path, dpi=120)


def run_parte5_detector(cfg: Config, device: torch.device) -> None:
    print("\n[*] Parte 5 (extra) — qualidade do detector: descarte, ruído e falsos positivos")
    out = Path(cfg.output_dir) / "final_parte5" if Path(cfg.output_dir).name == "outputs" else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.train.checkpoint) if cfg.train.checkpoint else Path("outputs/checkpoints/final_motion_rnn.pt")
    model, meta = load_checkpoint(ckpt)
    ctx = build_context(cfg)
    print(f"  modelo final: {ckpt} | detecções {ctx.detector} (score >= {ctx.min_conf:.3f}); associação {ctx.assoc}")
    raw_csv = out / "parte5b_detector_curva.csv"
    done = pd.read_csv(raw_csv, dtype={"video": str}) if raw_csv.exists() else pd.DataFrame()
    keys = set(zip(done.condition, done.seed, done.video)) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    todo = [(c, s, v) for c in CONDITIONS for s in (SEEDS if CONDITIONS[c] != CONDITIONS["original"] else (0,))
            for v in ctx.src.seqs if (c, s, v) not in keys]
    print(f"  {len(CONDITIONS)} condições × {len(SEEDS)} seeds × {len(ctx.src.seqs)} vídeos × {len(METHODS)} métodos; "
          f"{len(todo)} célula(s) a fazer")
    for i, (c, s, v) in enumerate(todo, 1):
        rows += _eval_cell(ctx, model, c, s, v)
        if i % 14 == 0 or i == len(todo):
            pd.DataFrame(rows).to_csv(raw_csv, index=False)
            print(f"  {i}/{len(todo)}", flush=True)
    raw = pd.DataFrame(rows)
    raw["video"] = raw["video"].astype(str)
    # a condição original não tem aleatoriedade: replica a seed 0 nas outras para a média/desvio fazerem sentido
    base = raw[raw.condition == "original"]
    raw = pd.concat([raw[raw.condition != "original"]] + [base.assign(seed=s) for s in SEEDS], ignore_index=True)
    for subset in ("val", "todos"):
        summ = summarize(raw, subset)
        summ.to_csv(out / f"parte5b_detector_resumo_{subset}.csv", index=False)
        print(f"\n  --- {subset} (média ± desvio entre as {len(SEEDS)} seeds de degradação) ---")
        cols = ["condition", "method", "mAP_mean", "det_F1_mean", "IDF1_mean", "IDF1_std", "ids_pred/gt_mean", "IDsw/id_mean"]
        print(summ[cols].round(3).to_string(index=False))
    _plot(summarize(raw, "val"), out / "parte5b_detector_val.png",
          "Parte 5 (extra) — degradação do detector, vídeos de validação (09, 13): média ± desvio de 3 seeds")
    _plot(summarize(raw, "todos"), out / "parte5b_detector_todos.png",
          "Parte 5 (extra) — degradação do detector, 7 vídeos (treino é in-sample para a RNN)")
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 5 (extra, detector) concluída.")
