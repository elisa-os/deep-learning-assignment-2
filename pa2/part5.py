"""Parte 5 — Teste de estresse: queda de taxa de quadros (sem retreinar, em cima do modelo final).

O vídeo é subamostrado a 1/k (k = 2 e 5, além do original) e o modelo roda como está. As detecções
públicas (SDP) e o GT são cortados nos mesmos quadros e renumerados (``pa2/stress/subsample.py``); cada
taxa é avaliada em TODAS as k fases de subamostragem (offsets) e reportada como média ± desvio entre fases.

Etapas:
  A. Diagnósticos em trajetórias do GT: por que o modelo quebra quando Δt muda (IoU entre amostras
     consecutivas, inclinação do deslocamento previsto, sensibilidade ao recurso Δt).
  B. Curva de degradação do IDF1 (principal): caixa parada, velocidade constante, RNN final com Δt = 1
     (como foi treinada) e RNN final alimentada com Δt = k; regra de associação inalterada, e uma variante
     com ``max_age`` casado no tempo (mesmo horizonte em segundos).
  C. EXPLORATÓRIO, fora da regra "sem retreinar": o mesmo GRU treinado com Δt em {1, 2, 5} (3 seeds),
     comparado com os checkpoints de Δt = 1 da Parte 3, para responder "alimentar Δt resolveria?".
  D. Figuras e resumo.

Saídas em ``outputs/final_parte5/``. As etapas B e C são retomáveis (pulam o que já está no CSV).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.ablation import Context, build_context, settings_for
from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.association.motion_tracker import track_sequence_motion
from pa2.config import Config
from pa2.models import MotionRNN, load_checkpoint, params_finite, train_motion_rnn
from pa2.mot17 import CAMERA_MOVING
from pa2.mot17.evaluate import evaluate_tracks
from pa2.mot17.trajectories import WindowSampler
from pa2.stress import SubsampledSequence, subsample_dets
from pa2.stress.diagnostics import consecutive_iou, dt_sensitivity, onestep_displacement_regression
from pa2.utils.visualize import save_figure

METHODS = ("static", "const_vel", "rnn_dt1", "rnn_dtk")
LABELS = {"static": "caixa parada (Parte 1)", "const_vel": "velocidade constante",
          "rnn_dt1": "RNN final, Δt = 1 (sem retreinar)", "rnn_dtk": "RNN final, Δt = k alimentado"}
COLORS = {"static": "tab:gray", "const_vel": "tab:orange", "rnn_dt1": "tab:red", "rnn_dtk": "tab:purple"}
KEY = ["variant", "method", "k", "phase", "video"]


# ─────────────────────────────────────────────────────────────────────────────
# uma célula: (vídeo, taxa, fase, modelo de movimento)
# ─────────────────────────────────────────────────────────────────────────────
def eval_cell(ctx: Context, video: str, k: int, phase: int, motion, max_age: int) -> dict:
    seq = SubsampledSequence(ctx.src.seqs[video], k, phase)
    d = subsample_dets(ctx.src.dets[video], k, phase)
    assoc = {**ctx.assoc, "max_age": max_age}
    tracks = track_sequence_motion(d, seq.info.seq_length, motion, min_conf=ctx.min_conf, **assoc)
    r = evaluate_tracks(seq, tracks)
    return {"split": ctx.splits[video], "n_frames": seq.info.seq_length, "fps_eff": seq.info.frame_rate,
            "max_age": max_age, "IDF1": r["idf1"], "ID_switches": r["id_switches"],
            "fragmentations": r["fragmentations"], "n_gt_ids": r["n_gt_ids"], "n_pred_ids": r["n_pred_ids"],
            "MOTA": r["mota"], "ids_pred/gt": r["n_pred_ids"] / max(1, r["n_gt_ids"]),
            "IDsw/id": r["id_switches"] / max(1, r["n_gt_ids"])}


def make_motion(method: str, model: MotionRNN, seq, k: int):
    size = (seq.info.im_width, seq.info.im_height)
    if method == "static":
        return StaticMotion()
    if method == "const_vel":
        return ConstantVelocityMotion()
    if method == "rnn_dt1":
        return RNNMotion(model, size, dt=1.0)
    if method == "rnn_dtk":
        return RNNMotion(model, size, dt=float(k))
    raise ValueError(method)


def _load_done(csv: Path) -> pd.DataFrame:
    return pd.read_csv(csv, dtype={"video": str}) if csv.exists() else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# A. diagnósticos
# ─────────────────────────────────────────────────────────────────────────────
def stage_a_diagnostics(cfg: Config, ctx: Context, model: MotionRNN, out: Path) -> dict:
    print("\n--- A. Diagnósticos: por que o modelo quebra quando Δt muda ---\n")
    sigma = tuple(cfg.rnn.obs_noise)
    strides = tuple(cfg.stress.rates)
    allseg = ctx.seg_tr + ctx.seg_va
    iou = consecutive_iou(allseg, strides)
    print("  IoU entre amostras consecutivas do mesmo objeto (GT, 7 vídeos):\n" + iou.round(3).to_string(index=False))
    iou.to_csv(out / "parte5_diag_iou_consecutivo.csv", index=False)

    makers = {"const_vel": lambda size, k: ConstantVelocityMotion(),
              "rnn_dt1": lambda size, k: RNNMotion(model, size, dt=1.0),
              "rnn_dtk": lambda size, k: RNNMotion(model, size, dt=float(k))}
    reg = []
    for split, segs in (("val", ctx.seg_va), ("treino", ctx.seg_tr)):
        for name, mk in makers.items():
            reg.append(onestep_displacement_regression(mk, segs, sigma, strides).assign(split=split, method=name))
    reg = pd.concat(reg, ignore_index=True)
    print("\n  Deslocamento previsto × verdadeiro em 1 passo (inclinação 1 = acompanha, < 1 = encolhe):")
    print(reg.pivot_table(index=["split", "method"], columns="stride", values="slope").round(3).to_string())
    reg.to_csv(out / "parte5_diag_inclinacao.csv", index=False)

    sens = dt_sensitivity(lambda size, dt: RNNMotion(model, size, dt=dt), ctx.seg_va, sigma,
                          tuple(k for k in strides if k > 1))
    print("\n  Sensibilidade ao recurso Δt (mesma entrada, Δt = 1 vs k), modelo final:\n" + sens.round(3).to_string(index=False))
    sens.to_csv(out / "parte5_diag_sensibilidade_dt.csv", index=False)
    return {"iou": iou, "slope": reg, "sens": sens}


# ─────────────────────────────────────────────────────────────────────────────
# B. curva de degradação
# ─────────────────────────────────────────────────────────────────────────────
def stage_b_curve(cfg: Config, ctx: Context, model: MotionRNN, out: Path) -> pd.DataFrame:
    print("\n--- B. Curva de degradação do IDF1 (sem retreinar) ---\n")
    csv = out / "parte5_curva.csv"
    done = _load_done(csv)
    have = set(zip(*(done[c] for c in KEY))) if len(done) else set()
    rows = done.to_dict("records")
    variants = [("fixa", ctx.assoc["max_age"])]
    if cfg.stress.time_matched_max_age:
        variants.append(("casada", None))
    t0 = time.time()
    for variant, _ in variants:
        for k in cfg.stress.rates:
            if variant == "casada" and k == 1:
                continue                                  # igual à regra fixa; copiado na hora de resumir
            age = ctx.assoc["max_age"] if variant == "fixa" else max(1, round(ctx.assoc["max_age"] / k))
            for phase in range(k):
                new = 0
                for video in ctx.splits:
                    for method in METHODS:
                        if method == "rnn_dtk" and k == 1:
                            continue                      # igual a rnn_dt1; copiado na hora de resumir
                        if (variant, method, k, phase, video) in have:
                            continue
                        seq = ctx.src.seqs[video]
                        r = eval_cell(ctx, video, k, phase, make_motion(method, model, seq, k), age)
                        rows.append({"variant": variant, "method": method, "k": k, "phase": phase,
                                     "video": video, **r})
                        new += 1
                if new:
                    pd.DataFrame(rows).to_csv(csv, index=False)
                    print(f"  [{variant}] 1/{k} fase {phase}: {new} células ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    return df


def expand(df: pd.DataFrame) -> pd.DataFrame:
    """Completa as células que são iguais por construção: rnn_dtk em k = 1 e a variante casada em k = 1."""
    extra = [df[(df.variant == "fixa") & (df.method == "rnn_dt1") & (df.k == 1)].assign(method="rnn_dtk")]
    if (df.variant == "casada").any():
        extra.append(df[(df.variant == "fixa") & (df.k == 1)].assign(variant="casada"))
        extra.append(df[(df.variant == "fixa") & (df.method == "rnn_dt1") & (df.k == 1)]
                     .assign(variant="casada", method="rnn_dtk"))
    return pd.concat([df] + extra, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Por (variante, método, k, subconjunto): média entre vídeos e, sobre as fases, média ± desvio."""
    df = expand(df)
    metrics = ["IDF1", "ids_pred/gt", "IDsw/id", "fragmentations", "MOTA"]
    rows = []
    mov = df.video.map(lambda v: bool(CAMERA_MOVING.get(v, False)))
    for (var, m, k, ph), g in df.groupby(["variant", "method", "k", "phase"]):
        gm = mov.loc[g.index]
        for subset, sub in (("val", g[g.split == "val"]), ("todos", g),
                            ("camera_parada", g[~gm]), ("camera_movel", g[gm])):
            if len(sub):
                rows.append({"variant": var, "method": m, "k": k, "phase": ph, "subset": subset,
                             **{c: sub[c].mean() for c in metrics}})
    per_phase = pd.DataFrame(rows)
    agg = per_phase.groupby(["variant", "method", "k", "subset"]).agg(
        n_fases=("phase", "nunique"), **{c: (c, "mean") for c in metrics},
        **{f"{c}_std": (c, "std") for c in metrics}).reset_index()
    base = agg[agg.k == 1].set_index(["variant", "method", "subset"])["IDF1"]
    agg["IDF1_rel"] = [r.IDF1 / base[(r.variant, r.method, r.subset)] for r in agg.itertuples()]
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# C. exploratório: treino multi-Δt
# ─────────────────────────────────────────────────────────────────────────────
def stage_c_multidt(cfg: Config, ctx: Context, out: Path) -> pd.DataFrame | None:
    if not cfg.stress.multi_dt:
        return None
    print("\n--- C. EXPLORATÓRIO (fora da regra 'sem retreinar'): treino multi-Δt ---\n")
    ck_dir = out / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)
    for seed in cfg.stress.multi_dt_seeds:
        ckpt = ck_dir / f"multidt_s{seed}.pt"
        if ckpt.exists():
            continue
        settings = settings_for(cfg, {"name": "multidt"}, seed)
        torch.manual_seed(seed)
        model = MotionRNN(cfg.model.rnn_type, cfg.model.hidden_size, cfg.model.use_delta_t,
                          cfg.model.num_layers, cfg.model.dropout, cfg.model.predict_uncertainty)
        sampler = WindowSampler(ctx.seg_tr, settings.window_T, cfg.stress.multi_dt_strides, seed=seed)
        print(f"  treinando multidt_s{seed}: strides {cfg.stress.multi_dt_strides}, {settings.epochs} épocas")
        hist = train_motion_rnn(model, sampler, ctx.val_batch, settings, save_to=ckpt,
                                extra_meta={"strides": cfg.stress.multi_dt_strides, "seed": seed, "exploratorio": True})
        if hist["diverged"]:
            print(f"  !! multidt_s{seed} divergiu")

    csv = out / "parte5_multidt.csv"
    done = _load_done(csv)
    key = ["family", "seed", "k", "phase", "video"]
    have = set(zip(*(done[c] for c in key))) if len(done) else set()
    rows = done.to_dict("records")
    t0 = time.time()
    for seed in cfg.stress.multi_dt_seeds:
        models = {"stride1_dt1": (load_checkpoint(cfg.stress.stride1_checkpoints.format(seed=seed))[0], "rnn_dt1"),
                  "multidt_dtk": (load_checkpoint(ck_dir / f"multidt_s{seed}.pt")[0], "rnn_dtk")}
        for family, (mdl, method) in models.items():
            if not params_finite(mdl):
                continue
            for k in cfg.stress.rates:
                for phase in range(k):
                    new = 0
                    for video in ctx.splits:
                        if (family, seed, k, phase, video) in have:
                            continue
                        r = eval_cell(ctx, video, k, phase, make_motion(method, mdl, ctx.src.seqs[video], k),
                                      ctx.assoc["max_age"])
                        rows.append({"family": family, "seed": seed, "k": k, "phase": phase, "video": video, **r})
                        new += 1
                    if new:
                        pd.DataFrame(rows).to_csv(csv, index=False)
            print(f"  seed {seed} {family}: ok ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)

    sigma = tuple(cfg.rnn.obs_noise)
    reg = []
    for seed in cfg.stress.multi_dt_seeds:
        for family, path, dtk in (("stride1_dt1", cfg.stress.stride1_checkpoints.format(seed=seed), False),
                                  ("multidt_dtk", ck_dir / f"multidt_s{seed}.pt", True)):
            mdl, _ = load_checkpoint(path)
            mk = (lambda size, k, mdl=mdl: RNNMotion(mdl, size, dt=float(k))) if dtk else \
                 (lambda size, k, mdl=mdl: RNNMotion(mdl, size, dt=1.0))
            reg.append(onestep_displacement_regression(mk, ctx.seg_va, sigma, tuple(cfg.stress.rates))
                       .assign(family=family, seed=seed))
    pd.concat(reg, ignore_index=True).to_csv(out / "parte5_multidt_inclinacao.csv", index=False)
    return df


def summarize_multidt(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, seed, k, ph), g in df.groupby(["family", "seed", "k", "phase"]):
        for subset, sub in (("val", g[g.split == "val"]), ("todos", g)):
            rows.append({"family": fam, "seed": seed, "k": k, "phase": ph, "subset": subset,
                         "IDF1": sub.IDF1.mean(), "IDsw/id": sub["IDsw/id"].mean(), "ids_pred/gt": sub["ids_pred/gt"].mean()})
    per = pd.DataFrame(rows)
    per_seed = per.groupby(["family", "seed", "k", "subset"]).mean(numeric_only=True).reset_index()
    return per_seed.groupby(["family", "k", "subset"]).agg(
        n_seeds=("seed", "nunique"), IDF1=("IDF1", "mean"), IDF1_std=("IDF1", "std"),
        IDsw_id=("IDsw/id", "mean"), ids_ratio=("ids_pred/gt", "mean")).reset_index()


# ─────────────────────────────────────────────────────────────────────────────
# D. figuras
# ─────────────────────────────────────────────────────────────────────────────
def _xlabels(rates, fps):
    return [("original" if k == 1 else f"1/{k}") + f"\n({fps / k:.0f} fps)" for k in rates]


def _plot_curve(summ: pd.DataFrame, variant: str, rates, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ[summ.variant == variant]
    x = np.arange(len(rates))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("val", "IDF1", "IDF1 — vídeos de validação (09, 13)"),
              ("todos", "IDF1", "IDF1 — 7 vídeos (treino é in-sample p/ a RNN)"),
              ("val", "IDF1_rel", "IDF1 relativo ao original — validação")]
    for ax, (subset, col, ttl) in zip(axes, panels):
        for m in METHODS:
            d = s[(s.method == m) & (s.subset == subset)].set_index("k").reindex(rates)
            if d[col].isna().all():
                continue
            err = d[f"{col}_std"] if f"{col}_std" in d else None
            ax.errorbar(x, d[col], yerr=err, marker="o", ms=4, capsize=3, color=COLORS[m], label=LABELS[m],
                        ls="--" if m == "rnn_dtk" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(rates, 30))
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("IDF1")
    axes[2].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[2].set_ylabel("IDF1 / IDF1 do vídeo original")
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_camera(summ: pd.DataFrame, rates, path: Path) -> None:
    """A degradação por tipo de câmera: é aqui que a queda de taxa realmente dói."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ[summ.variant == "fixa"]
    x = np.arange(len(rates))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, subset, ttl in ((axes[0], "camera_parada", "câmera parada (02, 04, 09)"),
                            (axes[1], "camera_movel", "câmera móvel (05, 10, 11, 13)")):
        for m in METHODS:
            d = s[(s.method == m) & (s.subset == subset)].set_index("k").reindex(rates)
            ax.errorbar(x, d.IDF1, yerr=d.IDF1_std, marker="o", ms=4, capsize=3, color=COLORS[m],
                        label=LABELS[m], ls="--" if m == "rnn_dtk" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(rates, 30))
        ax.set_title(ttl, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("IDF1 (média entre vídeos; ± desvio entre fases)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Parte 5 — IDF1 × taxa de quadros por tipo de câmera (regra de associação inalterada; "
                 "7 vídeos, treino é in-sample p/ a RNN)", fontsize=9)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_structure(summ: pd.DataFrame, rates, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summ[(summ.variant == "fixa") & (summ.subset == "val")]
    x = np.arange(len(rates))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, lab in ((axes[0], "ids_pred/gt", "ids previstos / verdadeiros"),
                         (axes[1], "IDsw/id", "ID switches / id verdadeiro")):
        for m in METHODS:
            d = s[s.method == m].set_index("k").reindex(rates)
            ax.errorbar(x, d[col], yerr=d[f"{col}_std"], marker="o", ms=4, capsize=3, color=COLORS[m],
                        label=LABELS[m], ls="--" if m == "rnn_dtk" else "-")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(rates, 30))
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axes[0].axhline(1.0, color="k", lw=0.8, ls=":")
    axes[0].legend(fontsize=7)
    fig.suptitle("Fragmentação de identidades com a queda de taxa — validação, regra de associação fixa", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_per_video(raw: pd.DataFrame, rates, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = expand(raw)
    d = d[d.variant == "fixa"].groupby(["video", "method", "k"]).IDF1.mean().reset_index()
    vids = sorted(d.video.unique())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=True)
    x = np.arange(len(rates))
    for ax, v in zip(axes.ravel(), vids):
        for m in METHODS:
            g = d[(d.video == v) & (d.method == m)].set_index("k").reindex(rates)
            ax.plot(x, g.IDF1, marker="o", ms=3, color=COLORS[m], ls="--" if m == "rnn_dtk" else "-",
                    label=LABELS[m])
        ax.set_title(f"MOT17-{v}", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(["1", "1/2", "1/5"][:len(rates)] if rates == [1, 2, 5] else [str(k) for k in rates])
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(vids):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("IDF1 (média entre fases)")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("IDF1 por vídeo × taxa de quadros (regra fixa)", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_diagnostics(diag: dict, rates, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    iou = diag["iou"]
    a = axes[0]
    a.plot(iou.stride, iou["frac_iou_abaixo_0.3"], "o-", color="tab:blue", label="IoU < 0,3 (some do portão)")
    a.plot(iou.stride, iou["frac_iou_abaixo_0.5"], "s--", color="tab:cyan", label="IoU < 0,5")
    a.set_xlabel("stride k (quadros entre amostras)")
    a.set_ylabel("fração dos pares de caixas do mesmo objeto")
    a.set_title("A caixa parada deixa de alcançar o objeto", fontsize=9)
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    b = axes[1]
    reg = diag["slope"][diag["slope"].split == "val"]
    for m in ("const_vel", "rnn_dt1", "rnn_dtk"):
        d = reg[reg.method == m]
        b.plot(d.stride, d.slope, marker="o", color=COLORS[m], label=LABELS[m], ls="--" if m == "rnn_dtk" else "-")
    b.axhline(1.0, color="k", lw=0.8, ls=":")
    b.set_xlabel("stride k")
    b.set_ylabel("inclinação: deslocamento previsto / verdadeiro")
    b.set_title("O deslocamento previsto em 1 passo (validação)", fontsize=9)
    b.legend(fontsize=7)
    b.grid(alpha=0.3)
    c = axes[2]
    s = diag["sens"]
    c.bar([str(k) for k in s.stride], s.rel_change_mediana, color="tab:purple")
    c.set_xlabel("Δt alimentado (k) vs Δt = 1")
    c.set_ylabel("mudança relativa mediana da previsão")
    c.set_title("O quanto o recurso Δt muda a saída do modelo final", fontsize=9)
    c.grid(alpha=0.3, axis="y")
    for ax in axes[:2]:
        ax.set_xticks(list(rates))
    fig.suptitle("Parte 5 — por que o modelo aprendido em Δt = 1 degrada quando Δt muda", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_multidt(msum: pd.DataFrame, summ: pd.DataFrame, rates, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rates))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    fam = {"stride1_dt1": ("Δt = 1 no treino (Parte 3, 3 seeds), Δt = 1 no teste", "tab:red", "-"),
           "multidt_dtk": ("EXPLORATÓRIO: Δt ∈ {1,2,5} no treino (3 seeds), Δt = k no teste", "tab:green", "-")}
    for ax, subset in zip(axes, ("val", "todos")):
        for f, (lab, col, ls) in fam.items():
            d = msum[(msum.family == f) & (msum.subset == subset)].set_index("k").reindex(rates)
            ax.errorbar(x, d.IDF1, yerr=d.IDF1_std, marker="o", ms=4, capsize=3, color=col, ls=ls, label=lab)
        ref = summ[(summ.variant == "fixa") & (summ.method == "static") & (summ.subset == subset)].set_index("k").reindex(rates)
        ax.plot(x, ref.IDF1, "o:", color="tab:gray", label="caixa parada (referência)")
        ax.set_xticks(x)
        ax.set_xticklabels(_xlabels(rates, 30))
        ax.set_ylabel("IDF1")
        ax.set_title("validação (09, 13)" if subset == "val" else "7 vídeos", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle("Alimentar Δt resolveria? — retreino multi-Δt exploratório (fora da regra 'sem retreinar')", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _print_tables(summ: pd.DataFrame, rates) -> None:
    for variant in summ.variant.unique():
        for subset in ("val", "camera_parada", "camera_movel"):
            print(f"\n  [{variant}] IDF1 médio, {subset} (± desvio entre fases):")
            t = summ[(summ.variant == variant) & (summ.subset == subset)]
            out = pd.DataFrame({m: [f"{r.IDF1:.3f}" + (f" ± {r.IDF1_std:.3f}" if pd.notna(r.IDF1_std) else "")
                                    for r in t[t.method == m].set_index("k").reindex(rates).itertuples()]
                                for m in METHODS}, index=[f"1/{k}" if k > 1 else "original" for k in rates])
            print(out.to_string())


# ─────────────────────────────────────────────────────────────────────────────
def run_parte5(cfg: Config, device: torch.device) -> None:
    print("\n[*] Parte 5 — Teste de estresse: queda de taxa de quadros")
    out = Path(cfg.output_dir) / "final_parte5" if Path(cfg.output_dir).name == "outputs" else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.train.checkpoint) if cfg.train.checkpoint else Path("outputs/checkpoints/final_motion_rnn.pt")
    model, meta = load_checkpoint(ckpt)
    rates = list(cfg.stress.rates)
    print(f"  modelo final: {ckpt} ({model.rnn_type}, h={model.hidden_size}, use_delta_t={model.use_delta_t}, "
          f"regime {meta.get('regime', '?')}, seed {meta.get('seed', '?')})")
    ctx = build_context(cfg)
    print(f"  detecções {ctx.detector} (score >= {ctx.min_conf:.3f}); associação {ctx.assoc}; taxas 1/k, k = {rates}")
    print(f"  split — treino: {ctx.train} | validação: {ctx.val}")

    diag = stage_a_diagnostics(cfg, ctx, model, out)
    raw = stage_b_curve(cfg, ctx, model, out)
    summ = summarize(raw)
    summ.to_csv(out / "parte5_resumo.csv", index=False)
    _print_tables(summ, rates)
    _plot_curve(summ, "fixa", rates, out / "parte5_idf1_fixa.png",
                "Parte 5 — IDF1 × taxa de quadros (sem retreinar; regra de associação inalterada: max_age = 30)")
    if "casada" in set(summ.variant):
        _plot_curve(summ, "casada", rates, out / "parte5_idf1_casada.png",
                    "Parte 5 — IDF1 × taxa de quadros com max_age casado no tempo (round(30/k))")
    _plot_camera(summ, rates, out / "parte5_idf1_camera.png")
    _plot_structure(summ, rates, out / "parte5_estrutura.png")
    _plot_per_video(raw, rates, out / "parte5_por_video.png")
    _plot_diagnostics(diag, rates, out / "parte5_diagnosticos.png")

    msum = None
    mdt = stage_c_multidt(cfg, ctx, out)
    if mdt is not None and len(mdt):
        msum = summarize_multidt(mdt)
        msum.to_csv(out / "parte5_multidt_resumo.csv", index=False)
        print("\n  multi-Δt (IDF1, média ± desvio entre as 3 seeds):\n" + msum.round(3).to_string(index=False))
        _plot_multidt(msum, summ, rates, out / "parte5_multidt.png")

    with open(out / "parte5_resumo.json", "w") as f:
        json.dump({"rates": rates, "detector": ctx.detector, "min_conf": ctx.min_conf, "association": ctx.assoc,
                   "model": str(ckpt), "summary": summ.replace({np.nan: None}).to_dict(orient="records"),
                   "multidt": None if msum is None else msum.replace({np.nan: None}).to_dict(orient="records")},
                  f, indent=2)
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 5 concluída.")
