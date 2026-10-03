"""Extra da Parte 3 — Eixo 1: a célula recorrente (RNN simples × GRU × LSTM) e a janela de BPTT.

A Parte 3 entregue é o Eixo 2 (regime de treino). Este módulo roda, como EXTRA, o Eixo 1 do enunciado:
RNN simples vs. GRU vs. LSTM **com orçamento de parâmetros aproximadamente igual** (o ``hidden_size`` de
cada célula é escolhido para chegar perto dos parâmetros da GRU de 64 unidades), com a janela de BPTT
truncado T ∈ {4, 8, 16, 32}, 3 seeds por configuração (média ± desvio). Treino igual ao do modelo final
(teacher forcing puro, clipping ligado, 20 épocas × 100 passos, checkpoint da última época); a
validação durante o treino e o rastreamento usam sempre as janelas T = 32 (protocolo fixo).

Também mede a curva de gradiente ‖∂L_t/∂h_{t−k}‖ (a mesma da Parte 4) de cada célula em T = 32, que é a
comparação "RNN simples × modelo com portas" que o enunciado pede na Parte 4 para quem faz o Eixo 1.

Uso: ``uv run pa2 eixo1`` (≈36 treinos); ``--seeds 42`` roda uma seed (vários processos); ``--aggregate-only``.
Saídas em ``outputs/extra_eixo1/``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.ablation import (REGIME_DEFAULTS, _clean, aggregate, build_context, run_metrics,
                          train_and_evaluate)
from pa2.analysis.memory import gradient_horizon, horizon_summary
from pa2.config import Config
from pa2.models import MotionRNN, load_checkpoint
from pa2.utils.visualize import save_figure

CELLS = ("RNN", "GRU", "LSTM")
WINDOWS = (4, 8, 16, 32)
COLORS = {"RNN": "tab:red", "GRU": "tab:blue", "LSTM": "tab:green"}
GRAD_T = 32


def n_params(cell: str, hidden: int) -> int:
    return sum(p.numel() for p in MotionRNN(cell, hidden).parameters())


def matched_hidden(cell: str, target: int) -> int:
    """``hidden_size`` da célula cujo total de parâmetros fica mais perto de ``target``."""
    return min(range(8, 257), key=lambda h: abs(n_params(cell, h) - target))


def build_regimes(cfg: Config) -> list[dict]:
    target = n_params(cfg.model.rnn_type, cfg.model.hidden_size)
    hidden = {c: matched_hidden(c, target) for c in CELLS}
    return [{"name": f"{c.lower()}_T{T}", "rnn_type": c, "hidden_size": hidden[c], "window_T": T,
             "params": n_params(c, hidden[c]), **REGIME_DEFAULTS} for c in CELLS for T in WINDOWS]


def _plot_vs_window(summary: pd.DataFrame, regimes: list[dict], col: str, ylabel: str, ax) -> None:
    for c in CELLS:
        rows = [(r["window_T"], summary[summary.regime == r["name"]]) for r in regimes if r["rnn_type"] == c]
        T = [t for t, d in rows if len(d)]
        m = [float(d[f"{col}_mean"].iloc[0]) for t, d in rows if len(d)]
        s = [float(np.nan_to_num(d[f"{col}_std"].iloc[0])) for t, d in rows if len(d)]
        hid = next(r["hidden_size"] for r in regimes if r["rnn_type"] == c)
        ax.errorbar(T, m, yerr=s, marker="o", capsize=3, color=COLORS[c], label=f"{c} (h={hid})")
    ax.set_xscale("log", base=2); ax.set_xticks(WINDOWS); ax.set_xticklabels(WINDOWS)
    ax.set_xlabel("janela de BPTT T"); ax.set_ylabel(ylabel); ax.grid(alpha=.3)


def _mean_k(hs: list[dict], key: str) -> float:
    """Média do k em que a curva cai abaixo do limiar; ``nan`` se alguma seed nunca chega lá."""
    v = [h[key] for h in hs]
    return float(np.mean(v)) if all(x is not None for x in v) else float("nan")


def _gradient_curves(cfg: Config, ctx, regimes: list[dict], seeds: list[int], ckpt_dir: Path, out: Path) -> dict:
    """Curva ‖∂L_t/∂h_{t−k}‖ (mediana de 400 janelas de validação), média entre seeds, T = 32."""
    curves = {}
    for c in CELLS:
        name = next(r["name"] for r in regimes if r["rnn_type"] == c and r["window_T"] == GRAD_T)
        per_seed = {"observed": [], "blind": []}
        for s in seeds:
            f = ckpt_dir / f"{name}_s{s}.pt"
            if not f.exists():
                continue
            m, _ = load_checkpoint(f)
            for mode in per_seed:
                per_seed[mode].append(horizon_summary(gradient_horizon(
                    m, ctx.seg_va, tuple(cfg.rnn.obs_noise), 48, 8, n=400, mode=mode, seed=0)))
        if per_seed["observed"]:
            curves[c] = {mode: {"relative": np.mean([h["relative"] for h in hs], axis=0).tolist(),
                                "k_below_0.5": _mean_k(hs, "k_below_0.5"),
                                "k_below_0.1": _mean_k(hs, "k_below_0.1"),
                                "k_below_0.01": _mean_k(hs, "k_below_0.01")}
                         for mode, hs in per_seed.items()}
    with open(out / "eixo1_gradiente.json", "w") as f:
        json.dump(_clean(curves), f)
    return curves


def write_outputs(cfg: Config, ctx, out: Path, runs: list[dict], regimes: list[dict], seeds: list[int],
                  ckpt_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_run, summary = aggregate(runs, regimes)
    per_run.to_csv(out / "eixo1_runs.csv", index=False)
    summary.to_csv(out / "eixo1_summary.csv", index=False)
    info = pd.DataFrame([{"regime": r["name"], "rnn_type": r["rnn_type"], "hidden_size": r["hidden_size"],
                          "window_T": r["window_T"], "params": r["params"]} for r in regimes])
    info.to_csv(out / "eixo1_parametros.csv", index=False)

    tab = summary[["regime", "n_runs", "n_diverged", "IDF1_val_mean", "IDF1_val_std", "blind_val_k10_mean",
                   "blind_val_k10_std", "blind_val_k30_mean", "grad_norm_mean_mean", "seconds_mean"]]
    tab = info.merge(tab, on="regime")
    print("\n" + tab.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    _plot_vs_window(summary, regimes, "IDF1_val", "IDF1 (vídeos de validação)", axes[0])
    _plot_vs_window(summary, regimes, "blind_val_k10", "IoU após 10 quadros às cegas (validação)", axes[1])
    _plot_vs_window(summary, regimes, "blind_val_k30", "IoU após 30 quadros às cegas (validação)", axes[2])
    axes[0].legend(fontsize=8)
    fig.suptitle("Extra (Eixo 1) — célula recorrente × janela de BPTT, mesmo orçamento de parâmetros "
                 "(média ± desvio de 3 seeds)", fontsize=11)
    save_figure(fig, out / "eixo1_celula_x_janela.png", dpi=120)

    curves = _gradient_curves(cfg, ctx, regimes, seeds, ckpt_dir, out)
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        for ax, mode, ttl in zip(axes, ("observed", "blind"), ("com observações", "às cegas (depois de 8 quadros)")):
            for c, cv in curves.items():
                rel = np.asarray(cv[mode]["relative"])
                ax.semilogy(np.arange(len(rel)), rel, color=COLORS[c],
                            label=f"{c}: cai a 1/10 em k = {cv[mode]['k_below_0.1']:.0f}" if mode == "observed" else c)
            ax.axvline(GRAD_T, color="k", ls=":", lw=1); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3, which="both")
            ax.set_xlabel("k (passos para trás)"); ax.set_ylabel("‖∂L_t/∂h_{t−k}‖ / valor em k = 0")
            ax.legend(fontsize=8)
        fig.suptitle(f"Extra (Eixo 1) — curva de gradiente por célula (T = {GRAD_T}, média de {len(seeds)} seeds)", fontsize=11)
        save_figure(fig, out / "eixo1_gradiente.png", dpi=120)
        print("\n  gradiente (k em que a norma cai a 1/10, com observações): " +
              ", ".join(f"{c} {cv['observed']['k_below_0.1']:.1f}" for c, cv in curves.items()))


def run_eixo1(cfg: Config, device: torch.device, only_seeds: list[int] | None = None,
              only_cells: list[str] | None = None, aggregate_only: bool = False) -> None:
    print("\n[*] Extra — Parte 3, Eixo 1: célula recorrente × janela de BPTT")
    regimes = build_regimes(cfg)
    cells = [c.upper() for c in only_cells] if only_cells else list(CELLS)
    seeds = only_seeds or cfg.ablation.seeds
    out = Path(cfg.output_dir) / "extra_eixo1"
    runs_dir, ckpt_dir = out / "runs", out / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for c in CELLS:
        r = next(r for r in regimes if r["rnn_type"] == c)
        print(f"  {c}: hidden {r['hidden_size']} -> {r['params']} parâmetros")

    todo = [(r, s) for r in regimes if r["rnn_type"] in cells for s in seeds
            if not (runs_dir / f"{r['name']}_s{s}.json").exists()]
    print(f"  seeds {seeds} | janelas {list(WINDOWS)} | {len(todo)} run(s) a fazer (o resto já existe)")
    ctx = build_context(cfg)
    if todo and not aggregate_only:
        for i, (regime, seed) in enumerate(todo, 1):
            tag = f"{regime['name']}_s{seed}"
            print(f"\n--- run {i}/{len(todo)}: {tag} ---")
            t0 = time.time()
            res = train_and_evaluate(cfg, ctx, regime, seed, ckpt_dir)
            res["total_seconds"] = time.time() - t0
            m = run_metrics(res)
            print(f"  -> IDF1 val {m.get('IDF1_val', float('nan')):.3f} | "
                  f"cego k=10 {m.get('blind_val_k10', float('nan')):.3f} | {res['total_seconds']:.0f}s"
                  f"{' | DIVERGIU' if res['diverged'] else ''}")
            with open(runs_dir / f"{tag}.json", "w") as f:
                json.dump(_clean(res), f)

    names = {r["name"] for r in regimes}
    runs = [json.load(open(f)) for f in sorted(runs_dir.glob("*.json"))]
    runs = [r for r in runs if r["regime"] in names]
    if not runs:
        print("  nenhum run encontrado")
        return
    print("\n--- Agregação ---")
    write_outputs(cfg, ctx, out, runs, regimes, seeds, ckpt_dir)
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Extra do Eixo 1 concluído.")
