"""Parte 3 — Ablação, Eixo 2: o regime de treino.

Teacher forcing → scheduled sampling → free-running, com gradient clipping ligado/desligado,
3 seeds por configuração (média ± desvio). O modelo é o da Parte 2 (GRU 64) e a fonte de
detecções, a mesma (SDP congelado). O que muda de uma configuração para outra é só o regime
de treino (ver ``regimes`` em ``config.yaml → parte3``).

Protocolo (igual para todas as configurações, para a comparação ser justa)
- mesmas épocas, mesmo lr/batch/ruído de observação; checkpoint da ÚLTIMA época (sem escolher
  a época pela validação, que significaria coisas diferentes em cada regime);
- validação durante o treino com protocolo fixo (teacher forcing + buracos simulados);
- depois do treino, em cada run:
    1. deriva: IoU alimentado com observações × alimentado só com as próprias previsões;
    2. IoU após k quadros às cegas (k = 1..30), em trajetórias do GT de validação e de treino;
    3. rastreamento nos 7 vídeos com a regra de associação da Parte 2 (os 2 de validação são o
       resultado principal; os de treino são "in-sample" para o modelo);
    4. reconexão depois de buracos de rastreamento (mesmo desfecho kept/switched/lost da Parte 2);
    5. estabilidade do treino: norma do gradiente, passos não finitos, divergência (pesos NaN).

Cada run grava ``outputs/parte3_ablation/runs/<regime>_s<seed>.json``; rodar de novo pula o que
já existe (retomável). ``uv run pa2 3 --seeds 42`` roda só uma seed (dá para lançar seeds em
processos separados); ``--aggregate-only`` só agrega o que existe.

Uso das saídas: ``parte3_*.csv|json|png`` em ``outputs/parte3_ablation/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.association.motion import RNNMotion
from pa2.association.motion_tracker import track_sequence_motion
from pa2.config import Config
from pa2.metrics.reconnection import BUCKET_NAMES, bucket_of, gap_episodes
from pa2.models import (MotionRNN, TrainSettings, eval_shift, load_checkpoint, params_finite,
                        train_motion_rnn)
from pa2.mot17 import resolve_split
from pa2.mot17.evaluate import evaluate_tracks
from pa2.mot17.trajectories import WindowSampler, load_segments
from pa2.part1 import best_f1_threshold, Source
from pa2.part2 import gap_rollout_iou
from pa2.utils.visualize import save_figure

REGIME_DEFAULTS = {"teacher_forcing_ratio": 1.0, "tf_schedule": "constant", "tf_end": 0.0,
                   "gap_prob": 0.0, "max_gap": 20, "gradient_clipping": True, "grad_clip_value": 1.0}
BLIND_KS = (1, 2, 5, 10, 15, 20, 30)
OUTCOME_CODE = {"kept": 0, "switched": 1, "lost": 2}
CODE_OUTCOME = {v: k for k, v in OUTCOME_CODE.items()}


# ─────────────────────────────────────────────────────────────────────────────
# configuração dos runs
# ─────────────────────────────────────────────────────────────────────────────
def settings_for(cfg: Config, regime: dict, seed: int) -> TrainSettings:
    r = {**REGIME_DEFAULTS, **regime}
    return TrainSettings(
        epochs=cfg.train.epochs, steps_per_epoch=cfg.rnn.steps_per_epoch,
        batch_size=cfg.data.batch_size, lr=cfg.train.lr, window_T=cfg.rnn.window_T,
        tf_ratio=r["teacher_forcing_ratio"], tf_schedule=r["tf_schedule"], tf_end=r["tf_end"],
        gap_prob=r["gap_prob"], max_gap=r["max_gap"], obs_noise=tuple(cfg.rnn.obs_noise),
        clip=r["gradient_clipping"], clip_value=r["grad_clip_value"], seed=seed, select="last")


@dataclass
class Context:
    root: Path
    train: list[str]
    val: list[str]
    detector: str
    seg_tr: list
    seg_va: list
    val_batch: tuple
    src: Source
    min_conf: float
    assoc: dict
    splits: dict


def build_context(cfg: Config) -> Context:
    root = Path(cfg.data.data_dir)
    if not (root / "train").exists():
        raise SystemExit(f"MOT17 não encontrado em {root}/train. Veja o README (download).")
    split = resolve_split(cfg.data.sequence_split)
    train, val = split["train"], split["val"]
    detector = cfg.detector_source or "SDP"
    seg_tr, seg_va = load_segments(root, train, detector), load_segments(root, val, detector)
    val_batch = WindowSampler(seg_va, cfg.rnn.window_T, seed=0).fixed_windows()
    src = Source.public(root, train + val, detector)
    thr, _ = best_f1_threshold(src, train)
    min_conf = cfg.association.min_conf if cfg.association.min_conf is not None else thr
    a = cfg.association
    assoc = dict(method=a.method, iou_threshold=a.iou_threshold, max_age=a.max_age, min_hits=a.min_hits)
    return Context(root, train, val, detector, seg_tr, seg_va, val_batch, src, min_conf, assoc,
                   {v: ("train" if v in train else "val") for v in train + val})


# ─────────────────────────────────────────────────────────────────────────────
# um run: treino + avaliação
# ─────────────────────────────────────────────────────────────────────────────
def _clean(x):
    """Converte numpy para tipos JSON."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def _tracking_eval(model: MotionRNN, ctx: Context) -> tuple[list[dict], list[dict]]:
    """Rastreia os 7 vídeos com a RNN; devolve (linhas por vídeo, episódios de buraco)."""
    rows, eps = [], []
    for v, seq in ctx.src.seqs.items():
        motion = RNNMotion(model, (seq.info.im_width, seq.info.im_height))
        tracks = track_sequence_motion(ctx.src.dets[v], seq.info.seq_length, motion,
                                       min_conf=ctx.min_conf, **ctx.assoc)
        r = evaluate_tracks(seq, tracks)
        rows.append({"video": v, "split": ctx.splits[v], "IDF1": r["idf1"],
                     "ID_switches": r["id_switches"], "fragmentations": r["fragmentations"],
                     "n_gt_ids": r["n_gt_ids"], "n_pred_ids": r["n_pred_ids"], "MOTA": r["mota"],
                     "ids_pred/gt": r["n_pred_ids"] / max(1, r["n_gt_ids"]),
                     "IDsw/id": r["id_switches"] / max(1, r["n_gt_ids"])})
        e = gap_episodes(seq, tracks)
        for length, outcome in zip(e["length"], e["outcome"]):
            eps.append({"video": v, "split": ctx.splits[v], "length": int(length),
                        "outcome": OUTCOME_CODE[outcome]})
    return rows, eps


def train_and_evaluate(cfg: Config, ctx: Context, regime: dict, seed: int, ckpt_dir: Path,
                       log=print) -> dict:
    name = regime["name"]
    settings = settings_for(cfg, regime, seed)
    torch.manual_seed(seed)
    model = MotionRNN(cfg.model.rnn_type, cfg.model.hidden_size, cfg.model.use_delta_t,
                      cfg.model.num_layers, cfg.model.dropout, cfg.model.predict_uncertainty)
    sampler = WindowSampler(ctx.seg_tr, settings.window_T, cfg.rnn.train_strides, seed=seed)
    ckpt = ckpt_dir / f"{name}_s{seed}.pt"
    hist = train_motion_rnn(model, sampler, ctx.val_batch, settings, log=log, save_to=ckpt,
                            extra_meta={"regime": name, "seed": seed})
    model, _ = load_checkpoint(ckpt)

    out = {"regime": name, "seed": seed, "settings": _clean(settings.__dict__), "history": _clean(hist),
           "diverged": bool(hist["diverged"] or not params_finite(model)), "tracking_error": None}
    if out["diverged"]:
        log(f"  [{name} s{seed}] DIVERGIU — avaliação pulada")
        return out

    out["shift"] = _clean(eval_shift(model, ctx.val_batch, settings))
    out["blind"] = {
        sp: [float(x) for x in gap_rollout_iou(lambda size: RNNMotion(model, size), segs,
                                              settings.obs_noise, n=600, seed=1)]
        for sp, segs in (("val", ctx.seg_va), ("train", ctx.seg_tr))}
    try:
        out["tracking"], out["episodes"] = _tracking_eval(model, ctx)
    except ValueError as e:      # p.ex. caixas não finitas durante o rastreamento
        out["tracking_error"] = str(e)
        log(f"  [{name} s{seed}] erro no rastreamento: {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# agregação
# ─────────────────────────────────────────────────────────────────────────────
def _load_runs(runs_dir: Path, regimes: list[dict]) -> list[dict]:
    names = {r["name"] for r in regimes}
    runs = []
    for f in sorted(runs_dir.glob("*.json")):
        d = json.load(open(f))
        if d["regime"] in names:
            runs.append(d)
    return runs


def run_metrics(run: dict) -> dict:
    """Uma linha plana de métricas para o run (NaN onde não se aplica)."""
    h = run["history"]
    m = {"regime": run["regime"], "seed": run["seed"], "diverged": run["diverged"],
         "epochs_run": h.get("epochs_run", np.nan),
         "train_loss_last": h["train_loss"][-1] if h["train_loss"] else np.nan,
         "val_loss_last": h["val_loss"][-1] if h["val_loss"] else np.nan,
         "val_iou_last": h["val_iou"][-1] if h["val_iou"] else np.nan,
         "grad_norm_mean": float(np.nanmean(h["grad_norm"])) if h["grad_norm"] else np.nan,
         "grad_norm_max": float(np.nanmax(h["grad_norm_max"])) if h["grad_norm_max"] else np.nan,
         "clipped_frac": float(np.mean(h["clipped_frac"])) if h["clipped_frac"] else np.nan,
         "nonfinite_loss_steps": int(np.sum(h["nonfinite_loss"])),
         "nonfinite_grad_steps": int(np.sum(h["nonfinite_grad"])),
         "seconds": h.get("seconds", np.nan)}
    if run["diverged"] or "shift" not in run:
        return m
    sh = run["shift"]
    m.update({"iou_obs": sh["iou_obs"], "iou_gaps": sh["iou_gaps"], "iou_free": sh["iou_free"],
              "nonfinite_preds_free": sh["nonfinite_frac_free"]})
    for sp in ("val", "train"):
        for k in BLIND_KS:
            m[f"blind_{sp}_k{k}"] = run["blind"][sp][k - 1]
    if run.get("tracking"):
        t = pd.DataFrame(run["tracking"])
        for sp, sub in (("val", t[t.split == "val"]), ("train", t[t.split == "train"]), ("all", t)):
            for c in ("IDF1", "ID_switches", "ids_pred/gt", "IDsw/id", "fragmentations", "MOTA"):
                m[f"{c}_{sp}"] = float(sub[c].mean())
        e = pd.DataFrame(run["episodes"])
        if len(e):
            e["bucket"] = e["length"].map(bucket_of)
            for sub_name, sub in (("val", e[e.split == "val"]), ("all", e)):
                m[f"gaps_{sub_name}"] = len(sub)
                m[f"gaps_switched_{sub_name}"] = float((sub.outcome == OUTCOME_CODE["switched"]).mean())
                long = sub[sub.length >= 6]
                m[f"gaps_switched_long_{sub_name}"] = (float((long.outcome == OUTCOME_CODE["switched"]).mean())
                                                       if len(long) else np.nan)
    return m


def aggregate(runs: list[dict], regimes: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = pd.DataFrame([run_metrics(r) for r in runs])
    order = [r["name"] for r in regimes]
    per_run["regime"] = pd.Categorical(per_run["regime"], order, ordered=True)
    per_run = per_run.sort_values(["regime", "seed"]).reset_index(drop=True)
    num = [c for c in per_run.columns if c not in ("regime", "seed", "diverged") and
           pd.api.types.is_numeric_dtype(per_run[c])]
    rows = []
    for name, g in per_run.groupby("regime", observed=True):
        row = {"regime": name, "n_runs": len(g), "n_diverged": int(g.diverged.sum())}
        for c in num:
            x = g[c].astype(float)
            row[f"{c}_mean"] = x.mean()
            row[f"{c}_std"] = x.std(ddof=1) if x.notna().sum() > 1 else np.nan
            row[f"{c}_n"] = int(x.notna().sum())
        rows.append(row)
    return per_run, pd.DataFrame(rows)


def _ms(summary: pd.DataFrame, col: str, fmt: str = "{:.3f}") -> list[str]:
    out = []
    for _, r in summary.iterrows():
        m, s = r.get(f"{col}_mean", np.nan), r.get(f"{col}_std", np.nan)
        out.append("—" if pd.isna(m) else (fmt.format(m) + (f" ± {fmt.format(s)}" if pd.notna(s) else "")))
    return out


def print_tables(summary: pd.DataFrame) -> None:
    cols = [("IDF1_val", "IDF1 val"), ("IDsw/id_val", "sw/id val"), ("ids_pred/gt_val", "ids p/v val"),
            ("IDF1_all", "IDF1 7 vídeos"), ("iou_obs", "IoU obs."), ("iou_free", "IoU free"),
            ("blind_val_k10", "IoU k=10"), ("blind_val_k30", "IoU k=30")]
    t = pd.DataFrame({"regime": summary.regime, "n": summary.n_runs, "diverg.": summary.n_diverged})
    for c, lab in cols:
        t[lab] = _ms(summary, c)
    print(t.to_string(index=False))
    s = pd.DataFrame({"regime": summary.regime,
                      "|g| média": _ms(summary, "grad_norm_mean", "{:.2f}"),
                      "|g| máx": _ms(summary, "grad_norm_max", "{:.1f}"),
                      "passos clip %": [f"{100 * v:.1f}" if pd.notna(v) else "—"
                                        for v in summary["clipped_frac_mean"]],
                      "perda não finita": _ms(summary, "nonfinite_loss_steps", "{:.0f}"),
                      "grad não finito": _ms(summary, "nonfinite_grad_steps", "{:.0f}"),
                      "val loss final": _ms(summary, "val_loss_last")})
    print("\n" + s.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# figuras
# ─────────────────────────────────────────────────────────────────────────────
def _style(name: str) -> dict:
    base = {"teacher_forcing": "tab:blue", "scheduled_sampling": "tab:green",
            "free_running": "tab:red", "part2_recipe": "tab:gray"}
    for k, c in base.items():
        if name.startswith(k):
            return {"color": c, "ls": "--" if name.endswith("no_clip") else "-"}
    return {"color": "k", "ls": "-"}


def _plot_bars(per_run: pd.DataFrame, summary: pd.DataFrame, panels, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summary.regime)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (col, lab) in zip(axes, panels):
        m = summary[f"{col}_mean"].to_numpy(float)
        s = np.nan_to_num(summary[f"{col}_std"].to_numpy(float))
        ax.bar(x, m, yerr=s, capsize=3, color=[_style(n)["color"] for n in names],
               hatch=None, alpha=0.8)
        for i, n in enumerate(names):
            if n.endswith("no_clip"):
                ax.patches[i].set_hatch("//")
            pts = per_run[per_run.regime == n][col].dropna()
            ax.scatter(np.full(len(pts), i), pts, color="k", s=10, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_blind(runs: list[dict], regimes: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, sp in zip(axes, ("val", "train")):
        for r in regimes:
            curves = np.array([x["blind"][sp] for x in runs
                               if x["regime"] == r["name"] and "blind" in x])
            if not len(curves):
                continue
            k = np.arange(1, curves.shape[1] + 1)
            mu, sd = curves.mean(0), (curves.std(0, ddof=1) if len(curves) > 1 else 0 * curves[0])
            st = _style(r["name"])
            ax.plot(k, mu, label=r["name"], **st)
            ax.fill_between(k, mu - sd, mu + sd, color=st["color"], alpha=0.12)
        ax.set_title(f"trajetórias do GT — {'validação' if sp == 'val' else 'treino'}")
        ax.set_xlabel("quadros sem observação (k)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("IoU entre a caixa prevista e o GT")
    axes[1].legend(fontsize=7)
    fig.suptitle("Parte 3 — rollout às cegas depois de 8 quadros observados (média ± desvio, 3 seeds)", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def _plot_grad(runs: list[dict], regimes: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, lab in ((axes[0], "grad_norm", "norma do gradiente (média por época)"),
                         (axes[1], "grad_norm_max", "norma do gradiente (máximo por época)")):
        for r in regimes:
            hs = [x["history"][key] for x in runs if x["regime"] == r["name"]]
            if not hs:
                continue
            n = min(len(h) for h in hs)
            arr = np.array([h[:n] for h in hs], dtype=float)
            ax.plot(np.arange(1, n + 1), np.nanmean(arr, 0), label=r["name"], **_style(r["name"]))
        ax.axhline(1.0, color="gray", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("época")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7)
    fig.suptitle("Parte 3 — norma do gradiente durante o treino (linha pontilhada = limite do clipping)", fontsize=10)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


def write_outputs(out: Path, runs: list[dict], regimes: list[dict]) -> None:
    if not runs:
        print("  nenhum run encontrado em", out / "runs")
        return
    per_run, summary = aggregate(runs, regimes)
    per_run.to_csv(out / "parte3_runs.csv", index=False)
    summary.to_csv(out / "parte3_summary.csv", index=False)
    print("\n  média ± desvio (amostral, ddof=1) sobre as seeds:\n")
    print_tables(summary)

    _plot_bars(per_run, summary, [("IDF1_val", "IDF1 (vídeos de validação)"),
                                  ("IDsw/id_val", "ID switches / id verdadeiro (validação)"),
                                  ("ids_pred/gt_val", "ids previstos / verdadeiros (validação)")],
               "Parte 3 — regime de treino: rastreamento nos vídeos de validação "
               "(barras = média, pontos = seeds, hachurado = sem clipping)", out / "parte3_tracking.png")
    _plot_bars(per_run, summary, [("iou_obs", "IoU com observações"), ("iou_gaps", "IoU com buracos"),
                                  ("iou_free", "IoU só com as próprias previsões")],
               "Parte 3 — deriva: o mesmo modelo, alimentado com observações × com as próprias previsões "
               "(janelas de validação)", out / "parte3_shift.png")
    _plot_blind(runs, regimes, out / "parte3_blind_rollout.png")
    _plot_grad(runs, regimes, out / "parte3_grad_norms.png")

    # reconexão por faixa de duração (média ± desvio entre seeds), validação e 7 vídeos
    rows = []
    for r in runs:
        for e in r.get("episodes", []):
            rows.append({"regime": r["regime"], "seed": r["seed"], **e})
    if rows:
        e = pd.DataFrame(rows)
        e["bucket"] = e["length"].map(bucket_of)
        tab = []
        for subset, d in (("val", e[e.split == "val"]), ("todos", e)):
            for (reg, b, seed), g in d.groupby(["regime", "bucket", "seed"]):
                tab.append({"subset": subset, "regime": reg, "gap": b, "seed": seed, "n": len(g),
                            "kept": (g.outcome == 0).mean(), "switched": (g.outcome == 1).mean()})
        tab = pd.DataFrame(tab)
        agg = tab.groupby(["subset", "regime", "gap"], observed=True).agg(
            n=("n", "mean"), kept_mean=("kept", "mean"), kept_std=("kept", "std"),
            switched_mean=("switched", "mean"), switched_std=("switched", "std")).reset_index()
        agg["gap"] = pd.Categorical(agg["gap"], BUCKET_NAMES, ordered=True)
        agg["regime"] = pd.Categorical(agg["regime"], [r["name"] for r in regimes], ordered=True)
        agg.sort_values(["subset", "regime", "gap"]).to_csv(out / "parte3_reconnection.csv", index=False)

    with open(out / "parte3_summary.json", "w") as f:
        json.dump(_clean(summary.replace({np.nan: None}).to_dict(orient="records")), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
def run_ablation(cfg: Config, device: torch.device, only_seeds: list[int] | None = None,
                 only_regimes: list[str] | None = None, aggregate_only: bool = False) -> None:
    print("\n[*] Parte 3 — Ablação, Eixo 2: regime de treino")
    regimes = cfg.ablation.regimes
    if not regimes:
        raise SystemExit("config.yaml → parte3 → ablation.regimes está vazio")
    seeds = only_seeds or cfg.ablation.seeds
    names = only_regimes or [r["name"] for r in regimes]
    unknown = set(names) - {r["name"] for r in regimes}
    if unknown:
        raise SystemExit(f"regimes desconhecidos: {sorted(unknown)}")

    out = Path(cfg.output_dir) / "parte3_ablation"
    runs_dir, ckpt_dir = out / "runs", out / "checkpoints"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    todo = [(r, s) for r in regimes if r["name"] in names for s in seeds
            if not (runs_dir / f"{r['name']}_s{s}.json").exists()]
    print(f"  regimes: {[r['name'] for r in regimes]}")
    print(f"  seeds: {seeds} | épocas: {cfg.train.epochs} × {cfg.rnn.steps_per_epoch} passos | "
          f"{len(todo)} run(s) a fazer (o resto já existe)")

    if todo and not aggregate_only:
        ctx = build_context(cfg)
        print(f"  detector {ctx.detector}, score >= {ctx.min_conf:.3f}, associação {ctx.assoc}")
        print(f"  split — treino: {ctx.train} | validação: {ctx.val}")
        for i, (regime, seed) in enumerate(todo, 1):
            tag = f"{regime['name']}_s{seed}"
            print(f"\n--- run {i}/{len(todo)}: {tag} ---")
            print(f"  {_clean({k: v for k, v in {**REGIME_DEFAULTS, **regime}.items() if k != 'name'})}")
            t0 = time.time()
            res = train_and_evaluate(cfg, ctx, regime, seed, ckpt_dir)
            res["total_seconds"] = time.time() - t0
            m = run_metrics(res)
            print(f"  -> IDF1 val {m.get('IDF1_val', float('nan')):.3f} | IoU obs "
                  f"{m.get('iou_obs', float('nan')):.3f} / free {m.get('iou_free', float('nan')):.3f} | "
                  f"{res['total_seconds']:.0f}s{' | DIVERGIU' if res['diverged'] else ''}")
            with open(runs_dir / f"{tag}.json", "w") as f:
                json.dump(_clean(res), f)

    print("\n--- Agregação ---")
    write_outputs(out, _load_runs(runs_dir, regimes), regimes)
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 3 concluída.")
