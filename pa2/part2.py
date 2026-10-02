"""Parte 2 — Trilha A: RNN como modelo de movimento.

A fonte de detecções é a da Parte 1 (SDP, score >= limiar da Parte 1) e fica congelada. O que
muda é o que acontece *entre* os quadros: em vez de comparar a detecção com a última caixa
vista, a associação compara com a caixa que a RNN prevê.

Fluxo:
  A. Dados: trajetórias do GT dos vídeos de treino (janelas de T quadros, com ruído de
     detector e buracos de observação simulados); a validação escolhe a melhor época.
  B. Treino (ou carrega ``--checkpoint``).
  C. Movimento isolado: IoU da caixa prevista após um buraco de k quadros, RNN vs. caixa
     parada (Parte 1) vs. velocidade constante (baseline de comparação).
  D. Rastreamento nos 7 vídeos, lado a lado: Parte 1 (estático), velocidade constante e RNN,
     com a MESMA regra de associação e gestão de tracks; depois cada um com a sua melhor
     regra (escolhida só no treino).

Saídas em ``outputs/``: ``parte2_*.csv|json|png`` e ``checkpoints/parte2_motion_rnn.pt``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.association.motion_tracker import track_sequence_motion
from pa2.config import Config
from pa2.models import MotionRNN, TrainSettings, load_checkpoint, train_motion_rnn
from pa2.mot17 import Sequence, resolve_split
from pa2.metrics.reconnection import BUCKET_NAMES, gap_episodes, reconnection_table
from pa2.mot17.evaluate import evaluate_tracks
from pa2.mot17.trajectories import WindowSampler, load_segments
from pa2.part1 import best_f1_threshold, pick_variant, Source
from pa2.utils.visualize import save_figure

GAP_KS = (1, 2, 5, 10, 15, 20, 30)
SWEEP = {"iou_threshold": [0.1, 0.2, 0.3, 0.4, 0.5], "max_age": [10, 30, 60]}
METHODS = ("static", "const_vel", "rnn")
LABELS = {"static": "Parte 1 (caixa parada)", "const_vel": "velocidade constante (baseline)",
          "rnn": "RNN (Trilha A)"}
COLORS = {"static": "tab:gray", "const_vel": "tab:orange", "rnn": "tab:red"}


# ─────────────────────────────────────────────────────────────────────────────
# C. movimento isolado
# ─────────────────────────────────────────────────────────────────────────────
def _iou_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU linha a linha entre duas pilhas de caixas [l, t, w, h]."""
    iw = np.clip(np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / np.maximum(a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter, 1e-9)


def gap_rollout_iou(make_motion, segments, sigma, context: int = 8, k_max: int = 30,
                    n: int = 600, seed: int = 0) -> np.ndarray:
    """IoU médio (k = 1..k_max) entre a caixa prevista e o GT depois de ``context`` quadros
    observados (com ruído de detector) seguidos de k quadros SEM observação."""
    from pa2.mot17.trajectories import cxcywh_to_ltwh
    rng = np.random.default_rng(seed)
    eligible = [s for s in segments if len(s) >= context + k_max]
    if not eligible:
        return np.full(k_max, np.nan)
    lens = np.array([len(s) - context - k_max + 1 for s in eligible], dtype=float)
    picks = [(eligible[i], int(rng.integers(0, int(lens[i])))) for i in
             rng.choice(len(eligible), size=n, p=lens / lens.sum())]
    sg = np.asarray(sigma)
    ious = np.zeros((k_max, n))
    for size in sorted({p[0].size for p in picks}):
        idx = [i for i, p in enumerate(picks) if p[0].size == size]
        gt = np.stack([picks[i][0].boxes[picks[i][1]: picks[i][1] + context + k_max] for i in idx])
        nz = rng.standard_normal((len(idx), context, 4)) * sg
        c = gt[:, :context]
        obs = np.stack([c[..., 0] + nz[..., 0] * c[..., 2], c[..., 1] + nz[..., 1] * c[..., 3],
                        c[..., 2] * np.exp(nz[..., 2]), c[..., 3] * np.exp(nz[..., 3])], -1)
        obs, gt = cxcywh_to_ltwh(obs), cxcywh_to_ltwh(gt)
        motion = make_motion(size)
        m = len(idx)
        states, prev = [None] * m, [None] * m
        for j in range(context):
            states, pred = motion.step(states, obs[:, j], prev, np.ones(m, bool))
            prev = list(obs[:, j])
        for k in range(1, k_max + 1):
            ious[k - 1, idx] = _iou_pairs(pred, gt[:, context - 1 + k])
            fed = pred                                     # sem observação: entra a própria previsão
            states, pred = motion.step(states, fed, prev, np.zeros(m, bool))
            prev = list(fed)
    return ious.mean(1)


def _motion_factories(model: MotionRNN) -> dict:
    return {"static": lambda size: StaticMotion(),
            "const_vel": lambda size: ConstantVelocityMotion(),
            "rnn": lambda size: RNNMotion(model, size)}


# ─────────────────────────────────────────────────────────────────────────────
# D. rastreamento
# ─────────────────────────────────────────────────────────────────────────────
def track_and_evaluate(src: Source, video: str, motion, assoc: dict, min_conf: float) -> dict:
    seq = src.seqs[video]
    tracks = track_sequence_motion(src.dets[video], seq.info.seq_length, motion,
                                   min_conf=min_conf, **assoc)
    return evaluate_tracks(seq, tracks)


def _row(method: str, seq: Sequence, split: str, assoc: dict, r: dict, peds: float) -> dict:
    return {"method": method, "sequence": seq.name, "split": split, "peds_per_frame": peds,
            "iou_threshold": assoc["iou_threshold"], "max_age": assoc["max_age"],
            "IDF1": r["idf1"], "ID_switches": r["id_switches"], "fragmentations": r["fragmentations"],
            "n_gt_ids": r["n_gt_ids"], "n_pred_ids": r["n_pred_ids"],
            "count_error": r["count_error_signed"],
            "ids_pred/gt": r["n_pred_ids"] / max(1, r["n_gt_ids"]),
            "IDsw/id": r["id_switches"] / max(1, r["n_gt_ids"]), "MOTA": r["mota"]}


def _evaluate_methods(src, motions, assocs: dict, min_conf, splits: dict, peds: dict) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for v, seq in src.seqs.items():
            r = track_and_evaluate(src, v, motions[m](seq), assocs[m], min_conf)
            rows.append(_row(m, seq, splits[v], assocs[m], r, peds[v]))
    return pd.DataFrame(rows)


def _tune(src, motions, base_assoc, train, min_conf) -> dict:
    """Melhor (iou_threshold, max_age) de cada método no TREINO (mesma regra de desempate da Parte 1)."""
    out = {}
    for m in METHODS:
        rows = []
        for iou_t, age in itertools.product(SWEEP["iou_threshold"], SWEEP["max_age"]):
            a = {**base_assoc, "iou_threshold": iou_t, "max_age": age}
            rs = [track_and_evaluate(src, v, motions[m](src.seqs[v]), a, min_conf) for v in train]
            rows.append({**a, "IDF1_train": np.mean([r["idf1"] for r in rs]),
                         "IDsw_train": np.mean([r["id_switches"] for r in rs])})
        out[m] = (pick_variant(pd.DataFrame(rows)), pd.DataFrame(rows))
    return out


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["method", "split"])[["IDF1", "ID_switches", "fragmentations", "ids_pred/gt",
                                         "IDsw/id", "MOTA"]].mean()
    return g.reset_index().sort_values(["split", "method"], key=lambda c: c.map(
        {**{m: i for i, m in enumerate(METHODS)}, "train": 0, "val": 1}))


# ─────────────────────────────────────────────────────────────────────────────
# figuras
# ─────────────────────────────────────────────────────────────────────────────
def _plot_training(hist: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    a.plot(ep, hist["train_loss"], label="treino"); a.plot(ep, hist["val_loss"], label="validação")
    a.axvline(hist["best_epoch"], color="gray", ls=":", label=f"melhor época ({hist['best_epoch']})")
    a.set_xlabel("época"); a.set_ylabel("smooth-L1 (unidades da caixa)"); a.legend(); a.grid(alpha=.3)
    b.plot(ep, hist["val_iou"], color="tab:green"); b.set_xlabel("época")
    b.set_ylabel("IoU médio previsto vs. GT (validação)"); b.grid(alpha=.3)
    fig.suptitle("Parte 2 — treino da RNN de movimento (trajetórias do GT)")
    plt.tight_layout(); save_figure(fig, path, dpi=120)


def _plot_gap_curves(curves: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(curves), figsize=(5.2 * len(curves), 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (split, cur) in zip(axes, curves.items()):
        for m in METHODS:
            ax.plot(range(1, len(cur[m]) + 1), cur[m], color=COLORS[m], label=LABELS[m])
        ax.set_title(f"trajetórias do GT — {split}"); ax.set_xlabel("quadros sem observação (k)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("IoU entre a caixa prevista e o GT"); axes[0].legend(fontsize=8)
    fig.suptitle("Movimento isolado: 8 quadros observados (ruído de detector) e depois k às cegas")
    plt.tight_layout(); save_figure(fig, path, dpi=120)


def _plot_side_by_side(df: pd.DataFrame, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seqs = (df.drop_duplicates("sequence").sort_values("peds_per_frame"))
    names = list(seqs.sequence)
    x = np.arange(len(names)); w = 0.27
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (col, lab) in zip(axes, [("IDF1", "IDF1"), ("ids_pred/gt", "ids previstos / verdadeiros"),
                                      ("IDsw/id", "ID switches / id verdadeiro")]):
        for i, m in enumerate(METHODS):
            d = df[df.method == m].set_index("sequence").loc[names]
            ax.bar(x + (i - 1) * w, d[col], w, color=COLORS[m], label=LABELS[m])
        ax.set_ylabel(lab); ax.grid(alpha=.3, axis="y")
    axes[1].axhline(1.0, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([f"{s.replace('MOT17-', '')}{'*' if sp == 'val' else ''}\n"
                              f"{p:.0f} ped/q" for s, sp, p in
                              zip(names, seqs.split, seqs.peds_per_frame)], fontsize=8)
    axes[-1].set_xlabel("vídeos ordenados por densidade; * = validação")
    fig.suptitle(title, fontsize=10, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); save_figure(fig, path, dpi=120)


def _plot_reconnection(tab: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    w = 0.27
    for ax, (col, ttl) in zip(axes, [("kept", "mesmo id depois do buraco"), ("switched", "id novo depois do buraco")]):
        for i, m in enumerate(METHODS):
            d = tab[tab.method == m].set_index("gap").reindex(BUCKET_NAMES)
            xs = np.arange(len(BUCKET_NAMES)) + (i - 1) * w
            ax.bar(xs, d[col], w, color=COLORS[m], label=LABELS[m])
        ax.set_xticks(range(len(BUCKET_NAMES)))
        ns = tab[tab.method == "rnn"].set_index("gap").reindex(BUCKET_NAMES)["n"]
        ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(BUCKET_NAMES, ns)], fontsize=8)
        ax.set_xlabel("duração do buraco (quadros sem casamento com o GT)"); ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("fração dos buracos"); axes[0].legend(fontsize=8)
    fig.suptitle("Parte 2 — o que acontece com a identidade depois de um buraco de rastreamento (7 vídeos)", fontsize=10)
    plt.tight_layout(); save_figure(fig, path, dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
def run_parte2(cfg: Config, device: torch.device) -> None:
    print("\n[*] Parte 2 — Trilha A: RNN como modelo de movimento")
    root = Path(cfg.data.data_dir)
    if not (root / "train").exists():
        raise SystemExit(f"MOT17 não encontrado em {root}/train. Veja o README (download).")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.train.checkpoint) if cfg.train.checkpoint else out / "checkpoints" / "parte2_motion_rnn.pt"

    split = resolve_split(cfg.data.sequence_split)
    train, val = split["train"], split["val"]
    detector = cfg.detector_source or "SDP"
    print(f"  split por vídeo — treino: {train} | validação: {val} | detector: {detector}")

    # ── A. dados ────────────────────────────────────────────────────────────
    print("\n--- A. Trajetórias do GT ---\n")
    seg_tr, seg_va = load_segments(root, train, detector), load_segments(root, val, detector)
    T = cfg.rnn.window_T
    print(f"  treino: {len(seg_tr)} segmentos, {sum(map(len, seg_tr))} caixas | "
          f"validação: {len(seg_va)} segmentos, {sum(map(len, seg_va))} caixas")
    print(f"  janela de BPTT T = {T}; ruído de observação (cx/w, cy/h, log w, log h) = {cfg.rnn.obs_noise}")
    settings = TrainSettings(
        epochs=cfg.train.epochs, steps_per_epoch=cfg.rnn.steps_per_epoch, batch_size=cfg.data.batch_size,
        lr=cfg.train.lr, window_T=T, tf_ratio=cfg.rnn.teacher_forcing_ratio, gap_prob=cfg.rnn.gap_prob,
        max_gap=cfg.rnn.max_gap, obs_noise=tuple(cfg.rnn.obs_noise), clip=cfg.train.gradient_clipping,
        clip_value=cfg.train.grad_clip_value, seed=cfg.seed)
    sampler = WindowSampler(seg_tr, T, cfg.rnn.train_strides, seed=cfg.seed)
    val_batch = WindowSampler(seg_va, T, seed=0).fixed_windows()

    # ── B. treino ───────────────────────────────────────────────────────────
    print("\n--- B. Treino ---\n")
    if cfg.train.eval_only or (cfg.train.checkpoint and ckpt_path.exists()):
        model, meta = load_checkpoint(ckpt_path)
        print(f"  checkpoint carregado: {ckpt_path} ({model.rnn_type}, h={model.hidden_size}, "
              f"época {meta.get('epoch', '?')})")
        hist = None
    else:
        torch.manual_seed(cfg.seed)
        model = MotionRNN(cfg.model.rnn_type, cfg.model.hidden_size, cfg.model.use_delta_t,
                          cfg.model.num_layers, cfg.model.dropout, cfg.model.predict_uncertainty)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"  {model.rnn_type}, hidden {model.hidden_size}, {n_par} parâmetros; "
              f"tf = {settings.tf_ratio}, buracos: p = {settings.gap_prob}, até {settings.max_gap} quadros; "
              f"clipping {'ligado' if settings.clip else 'desligado'}")
        hist = train_motion_rnn(model, sampler, val_batch, settings, save_to=ckpt_path,
                                extra_meta={"split": split, "detector": detector})
        model, _ = load_checkpoint(ckpt_path)           # volta para a melhor época
        print(f"  melhor época: {hist['best_epoch']} | checkpoint: {ckpt_path}")
        _plot_training(hist, out / "parte2_treino.png")
        with open(out / "parte2_train_history.json", "w") as f:
            json.dump(hist, f, indent=2)

    # ── C. movimento isolado ────────────────────────────────────────────────
    print("\n--- C. Movimento isolado: IoU após k quadros sem observação ---\n")
    factories = _motion_factories(model)
    curves = {}
    for name, segs in (("treino", seg_tr), ("validação", seg_va)):
        curves[name] = {m: gap_rollout_iou(factories[m], segs, settings.obs_noise, n=600, seed=1)
                        for m in METHODS}
    table = pd.DataFrame({f"{sp}:{m}": [curves[sp][m][k - 1] for k in GAP_KS]
                          for sp in curves for m in METHODS}, index=[f"k={k}" for k in GAP_KS])
    print(table.round(3).to_string())
    table.to_csv(out / "parte2_gap_rollout.csv")
    _plot_gap_curves(curves, out / "parte2_gap_rollout.png")

    # ── D. rastreamento ─────────────────────────────────────────────────────
    print("\n--- D. Rastreamento: Parte 1 vs. velocidade constante vs. RNN ---\n")
    src = Source.public(root, train + val, detector)
    thr, _ = best_f1_threshold(src, train)
    min_conf = cfg.association.min_conf if cfg.association.min_conf is not None else thr
    a = cfg.association
    base = dict(method=a.method, iou_threshold=a.iou_threshold, max_age=a.max_age, min_hits=a.min_hits)
    print(f"  detecções congeladas: {detector}, score >= {min_conf:.3f}; regra comum: {base}")
    splits = {v: ("train" if v in train else "val") for v in train + val}
    peds = {v: src.seqs[v].stats()["peds_per_frame"] for v in train + val}
    motions = {"static": lambda s: StaticMotion(), "const_vel": lambda s: ConstantVelocityMotion(),
               "rnn": lambda s: RNNMotion(model, (s.info.im_width, s.info.im_height),
                                          dt=1.0)}

    df_same = _evaluate_methods(src, motions, {m: base for m in METHODS}, min_conf, splits, peds)
    df_same.to_csv(out / "parte2_per_sequence.csv", index=False)
    print("  [mesma regra de associação para os três]\n")
    print(df_same.pivot(index="sequence", columns="method", values="IDF1")[list(METHODS)].round(3).to_string())
    sm = _summary(df_same)
    print("\n" + sm.round(3).to_string(index=False))
    _plot_side_by_side(df_same, f"Parte 2 vs. Parte 1 — {detector}, mesma associação {base}",
                       out / "parte2_comparacao.png")

    print("\n--- E. Reconexão depois de um buraco de rastreamento (mesma regra) ---\n")
    eps = []
    for m in METHODS:
        for v, seq in src.seqs.items():
            tr = track_sequence_motion(src.dets[v], seq.info.seq_length, motions[m](seq),
                                       min_conf=min_conf, **base)
            e = gap_episodes(seq, tr)
            eps.append(e.assign(method=m, sequence=seq.name,
                                camera="móvel" if seq.stats()["camera"] == "móvel" else "parada"))
    eps = pd.concat(eps, ignore_index=True)
    eps.to_csv(out / "parte2_gap_episodes.csv", index=False)
    rec = reconnection_table(eps)
    rec.to_csv(out / "parte2_reconnection.csv", index=False)
    print(rec.round(3).to_string(index=False))
    tot = eps.groupby("method").outcome.value_counts().unstack(fill_value=0)
    print("\n  totais por desfecho:\n" + tot.reindex(list(METHODS)).to_string())
    for cam, g in eps.groupby("camera"):
        r = g.groupby("method").outcome.apply(lambda x: (x == "kept").sum() / max(1, ((x == "kept") | (x == "switched")).sum()))
        print(f"  câmera {cam}: fração dos buracos reconectados que mantêm o id -> "
              + ", ".join(f"{m} {r[m]:.2f}" for m in METHODS))
    low = (eps.mean_visibility < 0.5).mean()
    print(f"  {100 * low:.0f}% dos buracos têm visibilidade média < 0,5 no GT (oclusão); o resto é detecção perdida")
    _plot_reconnection(rec, out / "parte2_reconexao.png")

    print("\n  [cada método com a sua melhor (iou_threshold, max_age), escolhida só no treino]\n")
    tuned = _tune(src, motions, base, train, min_conf)
    best_assoc = {}
    for m in METHODS:
        best_assoc[m] = tuned[m][0]
        print(f"  {m:10s} -> {tuned[m][0]}")
    df_tuned = _evaluate_methods(src, motions, best_assoc, min_conf, splits, peds)
    df_tuned.to_csv(out / "parte2_per_sequence_tuned.csv", index=False)
    smt = _summary(df_tuned)
    print("\n" + smt.round(3).to_string(index=False))

    with open(out / "parte2_summary.json", "w") as f:
        json.dump({"detector": detector, "min_conf": min_conf, "association_common": base,
                   "association_tuned": best_assoc, "split": split,
                   "model": {**model.config, "params": sum(p.numel() for p in model.parameters())},
                   "settings": settings.__dict__,
                   "summary_common": sm.to_dict(orient="records"),
                   "summary_tuned": smt.to_dict(orient="records"),
                   "gap_rollout_iou": {sp: {m: list(map(float, curves[sp][m])) for m in METHODS}
                                       for sp in curves}}, f, indent=2)
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 2 concluída.")
