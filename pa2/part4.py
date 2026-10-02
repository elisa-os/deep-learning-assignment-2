"""Parte 4 — Horizonte de memória, galeria de falhas e correção (modelo final).

Etapas:
  A. Horizonte ANALÍTICO: ||dL_t/dh_{t-k}|| em função de k, no modelo final e nos vídeos de
     validação (e, para comparar, nos checkpoints dos outros regimes da Parte 3).
  B. Horizonte EMPÍRICO: oclusões de N quadros injetadas nas detecções; fração em que a identidade
     volta com o mesmo id, comparada com a duração dos buracos reais do dataset.
  C. Galeria: três trechos em que o modelo final erra feio (figura + diagnóstico medido).
  D. Correção: uma mudança sugerida por um dos diagnósticos, antes/depois.

Saídas em ``outputs/final_parte4/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pa2.analysis.memory import (blind_displacement_regression, bootstrap_n50, detection_gap_runs, gradient_horizon,
                                 horizon_summary, n50_curve, survival_experiment, survival_table, visibility_runs)
from pa2.analysis.gallery import (draw_failure, episode_measures, gt_boxes_by_frame, pick_failures)
from pa2.analysis.runner import records_from_assign, run_trace, sorted_dets
from pa2.metrics.reconnection import gap_episodes
from pa2.metrics.tracking import clear_match, frame_range, parse_tracks_to_frames
from pa2.mot17.loader import to_mot_records
from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.config import Config
from pa2.models import load_checkpoint
from pa2.mot17 import Sequence, resolve_split
from pa2.mot17.trajectories import load_segments
from pa2.part1 import best_f1_threshold, Source
from pa2.part2 import track_and_evaluate, gap_rollout_iou
from pa2.utils.visualize import save_figure

SURVIVAL_NS = (0, 1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60)
SURVIVAL_ASSOC_MAX_AGE = 90        # maior que o maior N: o que limita é a previsão, não a regra
GRAD_L, GRAD_CONTEXT = 48, 8
GRAD_REGIMES = {                   # checkpoints da Parte 3 (seed 42) para comparar com o final
    "final (teacher forcing)": "outputs/checkpoints/final_motion_rnn.pt",
    "receita Parte 2 (com buracos)": "outputs/parte3_ablation/checkpoints/part2_recipe_s42.pt",
    "scheduled sampling": "outputs/parte3_ablation/checkpoints/scheduled_sampling_s42.pt",
    "free-running": "outputs/parte3_ablation/checkpoints/free_running_s42.pt",
}
LABELS = {"static": "caixa parada (Parte 1)", "const_vel": "velocidade constante",
          "rnn": "RNN (modelo final)"}
COLORS = {"static": "tab:gray", "const_vel": "tab:orange", "rnn": "tab:red"}


# ─────────────────────────────────────────────────────────────────────────────
def _plot_gradient(curves: dict, T: int, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, mode, ttl in zip(axes, ("observed", "blind"),
                             ("com observações (inferência normal)",
                              f"às cegas (depois de {GRAD_CONTEXT} quadros, só as próprias previsões)")):
        for i, (name, c) in enumerate(curves.items()):
            rel = np.array(c[mode]["relative"])
            ax.plot(np.arange(len(rel)), rel, label=name, lw=2.2 if i == 0 else 1.3,
                    color=f"C{i}")
        ax.axvline(T, color="k", ls=":", lw=1)
        ax.text(T + .5, ax.get_ylim()[0] * 1.5 if False else 0.02, f"janela de BPTT T={T}", fontsize=8,
                rotation=90, va="bottom", transform=ax.get_xaxis_transform())
        ax.set_yscale("log"); ax.set_xlabel("k (passos para trás)")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3, which="both")
    axes[0].set_ylabel("mediana de ||dL_t/dh_{t-k}|| / valor em k = 0")
    axes[0].legend(fontsize=8)
    fig.suptitle("Horizonte de memória analítico — vídeos de validação (09 e 13)", fontsize=11)
    plt.tight_layout(); save_figure(fig, path, dpi=120)


def _n50(tab: pd.DataFrame, method: str) -> float:
    """N50 do método a partir da tabela de sobrevivência (ver ``analysis.memory.n50_curve``)."""
    d = tab[tab.method == method].sort_values("N")
    return n50_curve(d.N, d.kept)


def _plot_survival(tab: pd.DataFrame, gaps: pd.DataFrame, vis: pd.DataFrame, path: Path,
                   max_age: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for m in ("static", "const_vel", "rnn"):
        d = tab[tab.method == m].sort_values("N")
        a.plot(d.N, d.kept, "o-", color=COLORS[m], label=f"{LABELS[m]} (N50 ≈ {_n50(tab, m):.0f})", ms=4)
        a.fill_between(d.N, d.ci_lo, d.ci_hi, color=COLORS[m], alpha=.12)
    a.axhline(.5, color="k", ls=":", lw=1)
    a.set_xlabel("duração N da oclusão injetada (quadros sem detecção)")
    a.set_ylabel("fração em que a identidade volta com o MESMO id")
    a.set_title(f"Sobrevivência do estado (max_age = {max_age}; IC 95% de Wilson)", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_ylim(0, 1.02)
    xs = np.arange(0, 61)
    for d, lab, c in ((gaps, "buracos de detecção (SDP) no GT", "tab:blue"),
                      (vis, "visibilidade < 0,2 no GT", "tab:green")):
        L = d.length.to_numpy()
        b.plot(xs, [(L > x).mean() for x in xs], color=c, label=f"{lab} (n={len(L)})")
    b.set_xlabel("N (quadros)"); b.set_ylabel("fração dos buracos MAIORES que N")
    b.set_yscale("log"); b.set_title("Duração das oclusões que existem no dataset (7 vídeos)", fontsize=10)
    b.legend(fontsize=8); b.grid(alpha=.3, which="both")
    plt.tight_layout(); save_figure(fig, path, dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
def stage_a_gradient(cfg: Config, seg_val, sigma, out: Path) -> dict:
    print("\n--- A. Horizonte analítico: ||dL_t/dh_{t-k}|| ---\n")
    curves = {}
    for name, path in GRAD_REGIMES.items():
        if not Path(path).exists():
            print(f"  (pulando {name}: {path} não existe)")
            continue
        m, _ = load_checkpoint(path)
        curves[name] = {mode: horizon_summary(gradient_horizon(m, seg_val, sigma, GRAD_L, GRAD_CONTEXT,
                                                               n=400, mode=mode, seed=0))
                        for mode in ("observed", "blind")}
    rows = []
    for name, c in curves.items():
        o, bl = c["observed"], c["blind"]
        rows.append({"modelo": name, "k (cai p/ 1/2)": o["k_below_0.5"], "k (cai p/ 1/10)": o["k_below_0.1"],
                     "k (cai p/ 1/100)": o["k_below_0.01"], "razão k=31 / k=0": round(o["relative"][31], 4),
                     "cego: razão k=31 / k=0": round(bl["relative"][31], 2),
                     "cego: razão k=47 / k=0": round(bl["relative"][47], 2)})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    tab.to_csv(out / "parte4_gradiente.csv", index=False)
    with open(out / "parte4_gradiente.json", "w") as f:
        json.dump(curves, f)
    _plot_gradient(curves, cfg.rnn.window_T, out / "parte4_gradiente.png")
    return curves


def stage_b_survival(root, videos, detector, min_conf, model, out: Path, reps: int = 4):
    print("\n--- B. Horizonte empírico: oclusões injetadas ---\n")
    src = Source.public(root, videos, detector)
    motions = {"static": lambda s: StaticMotion(), "const_vel": lambda s: ConstantVelocityMotion(),
               "rnn": lambda s: RNNMotion(model, (s.info.im_width, s.info.im_height))}
    assoc = dict(method="greedy", iou_threshold=0.3, max_age=SURVIVAL_ASSOC_MAX_AGE, min_hits=3)
    trials, gaps, vis = [], [], []
    for v in videos:
        seq = src.seqs[v]
        D = sorted_dets(src.dets[v], min_conf)
        print(f"  {seq.name}...", flush=True)
        trials.append(survival_experiment(seq, D, motions, assoc, SURVIVAL_NS, reps=reps))
        gaps.append(detection_gap_runs(seq, D))
        vis.append(visibility_runs(seq))
    trials, gaps, vis = pd.concat(trials), pd.concat(gaps), pd.concat(vis)
    trials.to_csv(out / "parte4_oclusoes_injetadas.csv", index=False)
    gaps.to_csv(out / "parte4_buracos_deteccao.csv", index=False)
    vis.to_csv(out / "parte4_buracos_visibilidade.csv", index=False)
    tab = survival_table(trials)
    tab.to_csv(out / "parte4_sobrevivencia.csv", index=False)
    piv = tab.pivot(index="N", columns="method", values="kept")[["static", "const_vel", "rnn"]]
    piv["n (rnn)"] = tab[tab.method == "rnn"].set_index("N")["n"]
    print(piv.round(2).to_string())
    n50 = {m: _n50(tab, m) for m in ("static", "const_vel", "rnn")}
    L = gaps.length.to_numpy()
    summary = {"N50": n50, "natural_gaps": {
        "n": int(len(L)), "median": float(np.median(L)), "p90": float(np.percentile(L, 90)),
        "frac_longer_than_N50_rnn": float((L > n50["rnn"]).mean()),
        "frac_longer_than_30": float((L > 30).mean()), "max": int(L.max())},
        "visibility_runs": {"n": int(len(vis)), "frac_longer_than_N50_rnn": float((vis.length > n50["rnn"]).mean())}}
    print(f"\n  N50 (quadros em que metade das identidades ainda volta com o mesmo id): {n50}")
    print(f"  buracos de detecção reais: n={len(L)}, mediana {np.median(L):.0f}, p90 {np.percentile(L, 90):.0f}, "
          f"{100 * (L > n50['rnn']).mean():.1f}% maiores que N50 da RNN, {100 * (L > 30).mean():.1f}% maiores que 30")
    boot = bootstrap_n50(trials)
    boot.to_csv(out / "parte4_n50_bootstrap.csv", index=False)
    print("\n  N50 com IC95% (bootstrap das tentativas; otimista, ver docstring):\n" + boot.round(2).to_string(index=False))
    _plot_survival(tab, gaps, vis, out / "parte4_sobrevivencia.png", SURVIVAL_ASSOC_MAX_AGE)
    with open(out / "parte4_horizonte_empirico.json", "w") as f:
        json.dump(summary, f, indent=2)
    return tab, summary


def _swap_partners(seq: Sequence, tracks: list[dict], eps: pd.DataFrame, iou: float = 0.5) -> list:
    """Para cada buraco com id novo: a OUTRA identidade do GT que tinha esse id pouco antes (ou None)."""
    pred_f, _ = parse_tracks_to_frames(tracks)
    gt_f, _ = parse_tracks_to_frames(to_mot_records(seq.gt_pedestrians()))
    frames = frame_range(pred_f, gt_f, None)
    matches = clear_match(pred_f, gt_f, iou, frames)["matches"]
    out = []
    for ep in eps.itertuples():
        partner = None
        if ep.outcome == "switched" and ep.pred_after is not None:
            for f in range(int(ep.start) - 1, max(0, int(ep.start) - 8), -1):
                owners = [g for g, p in matches.get(f, {}).items() if p == ep.pred_after and g != ep.gt_id]
                if owners:
                    partner = owners[0]
                    break
        out.append(partner)
    return out


def stage_c_gallery(root, val, detector, min_conf, model, assoc, out: Path):
    print("\n--- C. Galeria de falhas (modelo final, vídeos de validação) ---\n")
    src = Source.public(root, val, detector)
    meas_rows, ctx = [], {}
    for v in val:
        seq = src.seqs[v]
        D = sorted_dets(src.dets[v], min_conf)
        assign, gates = run_trace(D, seq.info.seq_length, RNNMotion(model, (seq.info.im_width, seq.info.im_height)),
                                  **assoc)
        tracks = records_from_assign(D, assign)
        eps = gap_episodes(seq, tracks)
        eps["swap_partner"] = _swap_partners(seq, tracks, eps)
        gtb = gt_boxes_by_frame(seq)
        for _, ep in eps.iterrows():
            m = episode_measures(seq, ep, assign, gates, D, None, gtb)
            m["swap_partner"] = ep.swap_partner
            meas_rows.append(m)
        ctx[v] = (seq, D, assign, gates, gtb)
    meas = pd.DataFrame(meas_rows)
    meas.to_csv(out / "parte4_trechos.csv", index=False)
    print(f"  {len(meas)} buracos nos vídeos de validação; {int((meas.outcome != 'kept').sum())} sem manter o id")
    picks = pick_failures(None, meas)
    chosen = {}
    for key, idx in picks.items():
        m = {k: (None if isinstance(v, float) and v != v else v) for k, v in meas.loc[idx].to_dict().items()}
        seq, D, assign, gates, gtb = ctx[m["video"]]
        caption = (f"MOT17-{m['video']} ({m['camera']}), identidade GT {m['gt_id']}: buraco de {m['length']} quadros "
                   f"(visibilidade média {m['mean_visibility']:.2f}), desfecho: {m['outcome']} "
                   f"(T{m['pred_before']} -> T{m['pred_after']}); IoU previsão × GT no fim: "
                   f"{m['gate_iou_at_end'] if m['gate_iou_at_end'] is None else round(m['gate_iou_at_end'], 2)}; "
                   f"track original viva no fim: {m['track_alive_at_end']}.")
        draw_failure(seq, m, assign, gates, D, gtb, seq.gt_pedestrians(), out / f"parte4_falha_{key}.png",
                     f"Falha: {key.replace('_', ' ')}", caption, save_figure)
        chosen[key] = m
        print(f"  {key}: " + caption)
    with open(out / "parte4_falhas.json", "w") as f:
        json.dump(chosen, f, indent=2, default=lambda x: None if x is None else (int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, np.floating) else str(x)))
    return chosen


# ─────────────────────────────────────────────────────────────────────────────
# D. correção: amortecer a extrapolação às cegas
# ─────────────────────────────────────────────────────────────────────────────
DAMPINGS = (1.0, 0.95, 0.9, 0.8, 0.7)      # hipótese registrada antes de rodar
DAMPINGS_POSTHOC = (1.15, 1.3)             # exploratório: escolhido DEPOIS de ver que a rede encolhe o movimento
CORR_NS = (0, 5, 10, 15, 20, 30, 40, 60)


def stage_d_correction(root, train, val, detector, min_conf, model, assoc, sigma, out: Path):
    """Diagnóstico (falha 1 + curva de gradiente às cegas): sem observação a rede se alimenta da
    própria previsão e INTEGRA o erro de velocidade do estado (o gradiente às cegas cresce com k; na
    falha 1 a caixa derivou 200 px no sentido contrário em 18 quadros). Mudança sugerida: amortecer o
    deslocamento extrapolado nos passos sem observação (fator ``damping`` por passo).

    Hipótese registrada ANTES de rodar: o amortecimento (i) aumenta o IoU às cegas e o N50 para
    buracos longos, (ii) pode piorar um pouco os buracos curtos (a velocidade verdadeira é jogada
    fora) e (iii) muda pouco o IDF1 médio, porque 90% dos buracos reais têm <= 12 quadros. O fator é
    escolhido pelo IDF1 médio dos vídeos de TREINO.
    """
    print("\n--- D. Correção: amortecer a extrapolação sem observação ---\n")
    src = Source.public(root, train + val, detector)
    seg_tr, seg_va = load_segments(root, train, detector), load_segments(root, val, detector)
    csv = out / "parte4_correcao.csv"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()      # retomável: pula fatores já medidos
    rows = done.to_dict("records")
    have = {round(float(r["damping"]), 4) for r in rows}
    slopes = pd.concat([blind_displacement_regression(lambda size: RNNMotion(model, size), sg, sigma).assign(split=n)
                        for n, sg in (("treino", seg_tr), ("val", seg_va))])
    slopes.to_csv(out / "parte4_extrapolacao.csv", index=False)
    print("  A extrapolação às cegas (fator 1) exagera ou encolhe o movimento?\n" + slopes.round(3).to_string(index=False) + "\n")
    for g in DAMPINGS + DAMPINGS_POSTHOC:
        if round(g, 4) in have:
            continue
        mk = lambda s, g=g: RNNMotion(model, (s.info.im_width, s.info.im_height), blind_damping=g)
        row = {"damping": g}
        for name, segs in (("treino", seg_tr), ("val", seg_va)):
            c = gap_rollout_iou(lambda size, g=g: RNNMotion(model, size, blind_damping=g), segs, sigma, n=600, seed=1)
            for k in (1, 5, 10, 20, 30):
                row[f"IoU cego k={k} ({name})"] = float(c[k - 1])
        res = {v: track_and_evaluate(src, v, mk(src.seqs[v]), assoc, min_conf) for v in train + val}
        for name, vs in (("treino", train), ("val", val)):
            row[f"IDF1 ({name})"] = float(np.mean([res[v]["idf1"] for v in vs]))
            row[f"ID sw/id ({name})"] = float(np.mean([res[v]["id_switches"] / max(1, res[v]["n_gt_ids"]) for v in vs]))
            row[f"ids prev/verd ({name})"] = float(np.mean([res[v]["n_pred_ids"] / max(1, res[v]["n_gt_ids"]) for v in vs]))
        trials = pd.concat([survival_experiment(
            src.seqs[v], sorted_dets(src.dets[v], min_conf), {"rnn": mk},
            dict(method="greedy", iou_threshold=0.3, max_age=SURVIVAL_ASSOC_MAX_AGE, min_hits=3),
            CORR_NS, reps=4) for v in train + val])
        tab = survival_table(trials)
        row["N50"] = _n50(tab, "rnn")
        for N in (10, 20, 40):
            row[f"sobrevive N={N}"] = float(tab[tab.N == N].kept.iloc[0])
        row["grupo"] = "registrada" if g in DAMPINGS else "pós-hoc"
        rows.append(row)
        print(f"  damping {g}: IDF1 treino {row['IDF1 (treino)']:.3f} val {row['IDF1 (val)']:.3f} | "
              f"N50 {row['N50']:.1f} | IoU cego k=10 val {row['IoU cego k=10 (val)']:.3f} k=30 {row['IoU cego k=30 (val)']:.3f}",
              flush=True)
    df = pd.DataFrame(rows).sort_values("damping", ascending=False).reset_index(drop=True)
    df["grupo"] = np.where(df.damping.round(4).isin([round(g, 4) for g in DAMPINGS]), "registrada", "pós-hoc")
    df.to_csv(csv, index=False)
    best = float(df[df.grupo == "registrada"].sort_values("IDF1 (treino)", ascending=False).iloc[0]["damping"])
    best_all = float(df.sort_values("IDF1 (treino)", ascending=False).iloc[0]["damping"])
    print(f"\n  fator escolhido (maior IDF1 de treino): registrados {best}; com os pós-hoc {best_all}")
    print(df.round(3).T.to_string(header=False))
    _plot_correction(df, best, out / "parte4_correcao.png")
    return df, best


def _plot_correction(df: pd.DataFrame, best: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14, 3.8))
    ks = (1, 5, 10, 20, 30)
    for _, r in df.iterrows():
        a.plot(ks, [r[f"IoU cego k={k} (val)"] for k in ks], "o-", ms=3,
               lw=2.4 if r.damping == 1.0 else 1.4, label=f"fator {r.damping}" + (" (antes)" if r.damping == 1.0 else ""))
    a.set_xlabel("quadros sem observação (k)"); a.set_ylabel("IoU previsão × GT (validação)")
    a.set_title("movimento às cegas", fontsize=10); a.legend(fontsize=7); a.grid(alpha=.3)
    b.plot(df.damping, df["N50"], "o-", color="tab:red"); b.invert_xaxis()
    b.set_xlabel("fator de amortecimento (1 = antes)"); b.set_ylabel("N50 (quadros)")
    b.set_title("sobrevivência do estado (7 vídeos)", fontsize=10); b.grid(alpha=.3)
    for col, lab, st in (("IDF1 (treino)", "treino", "-"), ("IDF1 (val)", "validação", "--")):
        c.plot(df.damping, df[col], "o" + st, label=lab)
    c.axvline(best, color="k", ls=":", lw=1); c.invert_xaxis()
    c.set_xlabel("fator de amortecimento (1 = antes)"); c.set_ylabel("IDF1 médio")
    c.set_title("rastreamento (mesma regra)", fontsize=10); c.legend(fontsize=8); c.grid(alpha=.3)
    plt.tight_layout(); save_figure(fig, path, dpi=120)


def run_parte4(cfg: Config, device: torch.device) -> None:
    print("\n[*] Parte 4 — Horizonte de memória, galeria de falhas e correção")
    root = Path(cfg.data.data_dir)
    out = Path(cfg.output_dir) / "final_parte4" if Path(cfg.output_dir).name == "outputs" else Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.train.checkpoint) if cfg.train.checkpoint else Path("outputs/checkpoints/final_motion_rnn.pt")
    model, meta = load_checkpoint(ckpt)
    print(f"  modelo final: {ckpt} ({model.rnn_type}, h={model.hidden_size}, regime "
          f"{meta.get('regime', '?')}, seed {meta.get('seed', '?')})")
    split = resolve_split(cfg.data.sequence_split)
    train, val = split["train"], split["val"]
    detector = cfg.detector_source or "SDP"
    sigma = tuple(cfg.rnn.obs_noise)

    seg_val = load_segments(root, val, detector)
    stage_a_gradient(cfg, seg_val, sigma, out)

    src = Source.public(root, train + val, detector)
    thr, _ = best_f1_threshold(src, train)
    min_conf = cfg.association.min_conf if cfg.association.min_conf is not None else thr
    stage_b_survival(root, train + val, detector, min_conf, model, out)
    assoc = dict(method=cfg.association.method, iou_threshold=cfg.association.iou_threshold,
                 max_age=cfg.association.max_age, min_hits=cfg.association.min_hits)
    stage_c_gallery(root, val, detector, min_conf, model, assoc, out)
    stage_d_correction(root, train, val, detector, min_conf, model, assoc, sigma, out)
    print(f"\n  Resultados em: {out}")
    print("\n[✓] Parte 4 (etapas A a D) concluída.")
