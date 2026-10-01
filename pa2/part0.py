"""Parte 0 — Testes sintéticos: validação do pipeline completo antes do MOT17.

Os 4 artefatos do enunciado, cada um com verificações que *falham* (assert) se o
comportamento esperado não for atendido:

1. Gerador: oclusão real por ordem de profundidade (+ figura da trajetória que some N
   quadros e volta, com a visibilidade medida pixel a pixel).
2. Simulador de detector: taxas de descarte, ruído e falsos positivos conferem com o pedido.
3. Métricas: IDF1 / ID switches / fragmentações nos 3 casos feitos à mão, com valores
   esperados derivados analiticamente (ver ``pa2/metrics/cases.py``).
4. Baseline no piso fácil: IDF1 ≈ 1; depois a varredura dos botões do gerador mostra
   onde o baseline quebra (ensaio do gráfico da Parte 1).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from pa2.association.matching import GreedyMatcher
from pa2.config import Config
from pa2.metrics.cases import CASES, EXPECTED
from pa2.metrics.tracking import evaluate_tracking_sequence
from pa2.synthetic_video import (
    SimulatedDetector,
    SyntheticSequence,
    generate_synthetic_sequence,
)
from pa2.synthetic_video.synthetic import HIDDEN_VISIBILITY, MIN_VISIBILITY
from pa2.utils import PerSequenceMetricsWriter
from pa2.utils.visualize import generate_tracking_colors, save_figure

# Baseline ingênuo (o mesmo no piso fácil e na varredura, para os dois serem comparáveis)
BASELINE_IOU = 0.3
BASELINE_MAX_AGE = 5
BASELINE_MIN_HITS = 1
EVAL_IOU = 0.5

# Parâmetros usados para verificar o simulador de detector
SIM_DROP, SIM_NOISE, SIM_FP = 0.2, 2.0, 0.5

EASY = {"n_objects": 3, "velocity_scale": 0.3, "occlusion_duration": 0, "num_frames": 30}
EASY_SEEDS = 5

# Varredura um botão por vez, a partir de uma configuração de referência
SWEEP_REF = {"n_objects": 6, "velocity_scale": 1.0, "occlusion_duration": 0}
SWEEP_VALUES = {
    "occlusion_duration": [0, 3, 5, 10, 20, 30],
    "velocity_scale": [0.3, 1.0, 2.0, 3.0, 4.0],
    "n_objects": [3, 6, 9, 12, 15],
}
SWEEP_SEEDS = 10
SWEEP_FRAMES = 60

def _run_baseline(seq: SyntheticSequence, detections: list[dict] | None = None) -> dict:
    """Associação ingênua sobre as detecções e avaliação contra o GT da sequência."""
    dets = seq.true_boxes if detections is None else detections
    matcher = GreedyMatcher(iou_threshold=BASELINE_IOU, max_age=BASELINE_MAX_AGE,
                            min_hits=BASELINE_MIN_HITS)
    tracks = matcher.run([dict(d) for d in dets], seq.num_frames)
    res = evaluate_tracking_sequence(tracks, seq.gt_tracks, iou_threshold=EVAL_IOU)
    res["tracks"] = tracks
    return res


def run_parte0(cfg: Config, device: torch.device) -> None:
    """Executa a Parte 0: testes sintéticos."""
    print("\n[*] Parte 0 — Testes Sintéticos")
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq = _check_generator(cfg, output_dir)
    _check_detector_simulator(cfg, seq)
    _check_metrics()
    easy_rows = _check_baseline_easy(cfg, output_dir)
    sweep = _run_parameter_sweep(cfg, output_dir)

    writer = PerSequenceMetricsWriter(output_dir / "metrics")
    for r in easy_rows:
        writer.add(
            sequence=f"baseline_easy_seed{r['seed']}",
            n_gt_ids=r["n_gt_ids"], n_pred_ids=r["n_pred_ids"], idf1=r["idf1"],
            id_switches=r["id_switches"], fragmentations=r["fragmentations"],
            mota=r["mota"], mostly_tracked=r["mostly_tracked"], mostly_lost=r["mostly_lost"],
            num_frames=r["num_frames"],
            notes=f"baseline ingênuo (IoU={BASELINE_IOU}, max_age={BASELINE_MAX_AGE}), piso fácil",
        )
    writer.write("parte0_baseline_metrics.csv")
    s = writer.summarize()
    print(f"\n  Piso fácil ({s['n_sequences']} seeds): IDF1 médio={s['idf1_mean']:.4f}, "
          f"ID switches médio={s['id_switches_mean']:.1f}, "
          f"fragmentações médio={s['fragmentations_mean']:.1f}")
    print(f"  Resultados em: {output_dir}")
    print("\n[✓] Parte 0 concluída.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Gerador
# ─────────────────────────────────────────────────────────────────────────────
def _check_generator(cfg: Config, output_dir: Path) -> SyntheticSequence:
    print("\n--- 1. Gerador: oclusão por ordem de profundidade ---\n")
    syn = cfg.synthetic
    N = syn.occlusion_duration_max

    # procura uma seed cuja oclusão tenha exatamente N quadros escondidos (figura limpa)
    seq = None
    for k in range(200):
        cand = generate_synthetic_sequence(
            np.random.default_rng(cfg.seed + k),
            num_frames=syn.n_frames, frame_size=syn.frame_size,
            n_objects=syn.n_objects_min, velocity_scale=syn.velocity_scale,
            occlusion_duration=N, noise_level=syn.noise_level, contrast_scale=syn.contrast_scale,
        )
        if len(cand.params["hidden_frames"]) == N:
            seq = cand
            break
    assert seq is not None, "nenhuma seed gerou oclusão de duração exata"

    p = seq.params
    vis = np.array(p["target_visibility"])
    hidden = p["hidden_frames"]
    print(f"  {seq.num_frames} quadros, {seq.n_objects} elipses, "
          f"alvo ID={p['occlusion_ellipse_id']} escondido atrás do ID={p['covering_ellipse_id']}")
    print(f"  Quadros escondidos (visibilidade < {HIDDEN_VISIBILITY}): "
          f"{hidden[0]}..{hidden[-1]} ({len(hidden)} quadros, pedido: {N})")
    print(f"  Visibilidade do alvo: antes={vis[: hidden[0] - 1].max():.2f}, "
          f"mínima={vis.min():.2f}, depois={vis[hidden[-1]:].max():.2f}")

    assert len(hidden) == N
    assert hidden == list(range(hidden[0], hidden[-1] + 1)), "oclusão deveria ser contígua"
    assert vis[: hidden[0] - 1].max() >= MIN_VISIBILITY, "alvo deveria aparecer antes da oclusão"
    assert vis[hidden[-1]:].max() >= MIN_VISIBILITY, "alvo deveria reaparecer depois da oclusão"
    # o alvo escondido não é visto por um detector perfeito, mas continua no GT (conf = 0)
    tid = p["occlusion_ellipse_id"]
    det_frames = {b["frame"] for b in seq.true_boxes if b["id"] == tid}
    assert not det_frames & set(hidden), "detector perfeito não pode ver o objeto escondido"
    assert any(d["id"] == tid and d["frame"] in hidden and d["conf"] == 0 for d in seq.gt_tracks)
    print("  ✓ o alvo some por N quadros e volta; o detector perfeito não o enxerga")

    _save_occlusion_demo(seq, output_dir / "parte0_occlusion_demo.png")
    print("  ✓ figura salva: parte0_occlusion_demo.png")

    # parâmetros expostos: ruído e contraste realmente afetam a imagem
    base = generate_synthetic_sequence(np.random.default_rng(1), num_frames=10, n_objects=5)
    noisy = generate_synthetic_sequence(np.random.default_rng(1), num_frames=10, n_objects=5,
                                        noise_level=0.1)
    low_c = generate_synthetic_sequence(np.random.default_rng(1), num_frames=10, n_objects=5,
                                        contrast_scale=0.3)
    assert np.abs(noisy.frames[0].astype(float) - base.frames[0]).mean() > 3
    assert low_c.frames[0].std() < base.frames[0].std()
    print("  ✓ noise_level e contrast_scale alteram a imagem")
    return seq


def _draw_boxes(ax, seq: SyntheticSequence, frame: int, target_id: int, colors) -> None:
    from matplotlib.patches import Rectangle

    for d in seq.gt_tracks:
        if d["frame"] != frame:
            continue
        is_target = d["id"] == target_id
        hidden = d["conf"] == 0
        x, y, w, h = d["bb_left"], d["bb_top"], d["bb_width"], d["bb_height"]
        if not is_target and hidden:
            continue
        color = "yellow" if is_target else colors[d["id"] % len(colors)][:3]
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=color,
                               linewidth=2 if is_target else 0.8,
                               linestyle="--" if (is_target and hidden) else "-"))
        if is_target:
            ax.text(x, y - 2, f"alvo {d['id']}" + (" (escondido)" if hidden else ""),
                    color="yellow", fontsize=7)


def _save_occlusion_demo(seq: SyntheticSequence, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = seq.params
    hidden = p["hidden_frames"]
    tid = p["occlusion_ellipse_id"]
    first, last = hidden[0], hidden[-1]
    shown = [first - 4, first - 1, first, (first + last) // 2, last, last + 3]
    shown = [min(max(1, f), seq.num_frames) for f in shown]
    colors = generate_tracking_colors(seq.n_objects + 1, seed=42)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 0.9])
    for k, f in enumerate(shown):
        ax = fig.add_subplot(gs[k // 3, (k % 3) * 2:(k % 3) * 2 + 2])
        ax.imshow(seq.frames[f - 1])
        _draw_boxes(ax, seq, f, tid, colors)
        v = p["target_visibility"][f - 1]
        ax.set_title(f"quadro {f} — visibilidade do alvo {v:.2f}", fontsize=9)
        ax.axis("off")

    ax = fig.add_subplot(gs[2, :])
    t = np.arange(1, seq.num_frames + 1)
    ax.plot(t, p["target_visibility"], "o-", color="tab:orange", ms=3)
    ax.axvspan(first - 0.5, last + 0.5, color="red", alpha=0.15,
               label=f"escondido por {len(hidden)} quadros")
    ax.axhline(MIN_VISIBILITY, color="gray", ls=":", label="limiar de visibilidade (conf=1)")
    ax.set_xlabel("quadro")
    ax.set_ylabel("visibilidade do alvo")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"Oclusão por profundidade — elipse {tid} passa atrás da elipse {p['covering_ellipse_id']} "
        f"e some por {len(hidden)} quadros (caixa tracejada = GT do objeto escondido)",
        fontsize=11,
    )
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulador de detector
# ─────────────────────────────────────────────────────────────────────────────
def _check_detector_simulator(cfg: Config, seq: SyntheticSequence) -> None:
    print("\n--- 2. Simulador de detector ---\n")
    fs = seq.frame_size

    # caixas verdadeiras bem no meio do quadro (sem corte pelas bordas): 4000 caixas
    n_frames = 400
    gt = [{"frame": f, "id": i, "bb_left": 50, "bb_top": 50, "bb_width": 20, "bb_height": 20,
           "conf": 1.0} for f in range(1, n_frames + 1) for i in range(10)]

    # (a) sem degradação: devolve as mesmas caixas
    clean = SimulatedDetector(drop_rate=0, noise_std=0, fp_rate=0,
                              rng=np.random.default_rng(0)).detect(gt, fs, n_frames)
    assert len(clean) == len(gt) and not any(d["is_fp"] for d in clean)
    assert all(d["bb_left"] == 50 and d["bb_width"] == 20 for d in clean)

    # (b) degradação ligada
    dets = SimulatedDetector(drop_rate=SIM_DROP, noise_std=SIM_NOISE, fp_rate=SIM_FP,
                             rng=np.random.default_rng(0)).detect(gt, fs, n_frames)
    true_dets = [d for d in dets if not d["is_fp"]]
    fps = [d for d in dets if d["is_fp"]]
    drop = 1 - len(true_dets) / len(gt)
    noise_std = float(np.std([d["bb_left"] - 50 for d in true_dets]))
    fp_per_frame = len(fps) / n_frames
    print(f"  pedido : drop={SIM_DROP}, ruído={SIM_NOISE}px, FP/quadro={SIM_FP}")
    print(f"  medido : drop={drop:.3f}, ruído={noise_std:.2f}px, FP/quadro={fp_per_frame:.3f}")
    assert abs(drop - SIM_DROP) < 0.03, drop
    assert abs(noise_std - SIM_NOISE) < 0.15, noise_std
    assert abs(fp_per_frame - SIM_FP) < 0.08, fp_per_frame
    print("  ✓ taxas de descarte, ruído e falsos positivos conferem")

    # (c) sobre a sequência gerada, com a configuração do yaml (pode ser identidade)
    sim = SimulatedDetector(
        drop_rate=cfg.synthetic.detector_drop_rate, noise_std=cfg.synthetic.detector_noise,
        fp_rate=cfg.synthetic.detector_fp_rate, rng=np.random.default_rng(cfg.seed + 777),
    )
    out = sim.detect(seq.true_boxes, fs, seq.num_frames)
    print(f"  yaml: {len(seq.true_boxes)} caixas visíveis -> {len(out)} detecções "
          f"({sum(d['is_fp'] for d in out)} FP)")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Métricas
# ─────────────────────────────────────────────────────────────────────────────
def _check_metrics() -> None:
    print("\n--- 3. Métricas de tracking: três casos feitos à mão ---\n")
    titles = {"a": "predição = GT",
              "b": "ids 1 e 2 trocados a partir do quadro 16",
              "c": "track 1 partida no quadro 10 e sem detecção nos quadros 12-14"}
    results = {}
    for key, build in CASES.items():
        pred, gt = build()
        r = evaluate_tracking_sequence(pred, gt, iou_threshold=EVAL_IOU)
        exp = EXPECTED[key]
        print(f"  ({key}) {titles[key]}")
        print(f"      IDF1 = {r['idf1']:.4f} (esperado {exp['idf1']:.4f}) | "
              f"ID sw = {r['id_switches']} (esperado {exp['id_switches']}) | "
              f"frag = {r['fragmentations']} (esperado {exp['fragmentations']})")
        assert abs(r["idf1"] - exp["idf1"]) < 1e-9, (key, r["idf1"], exp["idf1"])
        assert r["id_switches"] == exp["id_switches"], (key, r["id_switches"])
        assert r["fragmentations"] == exp["fragmentations"], (key, r["fragmentations"])
        results[key] = r
    assert abs(results["b"]["idf1"] - results["c"]["idf1"]) > 0.1, "(b) e (c) deveriam diferir"
    print("  ✓ os três casos batem; IDF1 de (b) e (c) são diferentes, como esperado")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Baseline no piso fácil
# ─────────────────────────────────────────────────────────────────────────────
def _check_baseline_easy(cfg: Config, output_dir: Path) -> list[dict]:
    print("\n--- 4. Baseline no piso fácil ---\n")
    print(f"  {EASY} | detector perfeito | associação gulosa IoU={BASELINE_IOU}, "
          f"max_age={BASELINE_MAX_AGE}")
    rows = []
    first_seq = first_res = None
    for k in range(EASY_SEEDS):
        seq = generate_synthetic_sequence(np.random.default_rng(cfg.seed + 42 + k), **EASY)
        res = _run_baseline(seq)
        rows.append({"seed": cfg.seed + 42 + k, **{x: res[x] for x in res if x != "tracks"}})
        print(f"  seed {cfg.seed + 42 + k}: IDF1={res['idf1']:.4f} sw={res['id_switches']} "
              f"frag={res['fragmentations']} ids pred/gt={res['n_pred_ids']}/{res['n_gt_ids']}")
        if first_seq is None:
            first_seq, first_res = seq, res

    worst = min(r["idf1"] for r in rows)
    assert worst >= 0.99, f"baseline no piso fácil deveria ter IDF1 ≈ 1, pior = {worst:.4f}"
    print(f"  ✓ IDF1 ≥ 0.99 em todas as seeds (pior = {worst:.4f})")

    _save_tracking_demo(first_seq, first_res["tracks"], output_dir / "parte0_baseline_easy.png")
    return rows


def _save_tracking_demo(seq: SyntheticSequence, tracks_pred: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    f = seq.num_frames // 2
    colors = generate_tracking_colors(64, seed=42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, title, rows in ((axes[0], "Ground truth", seq.gt_tracks),
                            (axes[1], "Baseline ingênuo", tracks_pred)):
        ax.imshow(seq.frames[f - 1])
        ax.set_title(f"{title} — quadro {f}")
        ax.axis("off")
        for d in rows:
            if d["frame"] == f and d.get("conf", 1) > 0:
                c = colors[d["id"] % len(colors)][:3]
                ax.add_patch(Rectangle((d["bb_left"], d["bb_top"]), d["bb_width"], d["bb_height"],
                                       fill=False, edgecolor=c, linewidth=1.8))
                ax.text(d["bb_left"], d["bb_top"] - 2, str(d["id"]), color=c, fontsize=8)
    plt.tight_layout()
    save_figure(fig, path, dpi=120)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Varredura dos botões do gerador
# ─────────────────────────────────────────────────────────────────────────────
def _run_parameter_sweep(cfg: Config, output_dir: Path) -> list[dict]:
    """Gira um botão do gerador por vez (n_objects, velocidade, duração da oclusão).

    Os outros ficam na referência ``SWEEP_REF`` (6 objetos, velocidade 1, sem oclusão).
    O detector é perfeito *sobre o que é visível*: objetos escondidos não são
    detectados, então a degradação vem só dos parâmetros do gerador. Cada ponto é a
    média de ``SWEEP_SEEDS`` sequências.
    """
    print("\n--- 5. Varredura dos botões do gerador ---\n")
    print(f"  referência: {SWEEP_REF} | {SWEEP_SEEDS} seeds | {SWEEP_FRAMES} quadros")

    results: list[dict] = []
    for knob, values in SWEEP_VALUES.items():
        for v in values:
            setting = {**SWEEP_REF, knob: v}
            for s in range(SWEEP_SEEDS):
                seed = cfg.seed * 100003 + s * 7919 + int(v * 10) * 31 + len(knob)
                seq = generate_synthetic_sequence(
                    np.random.default_rng(seed), num_frames=SWEEP_FRAMES, **setting)
                res = _run_baseline(seq)
                results.append({
                    "knob": knob, "value": v, **setting, "seed": s,
                    "idf1": float(res["idf1"]),
                    "id_switches": int(res["id_switches"]),
                    "fragmentations": int(res["fragmentations"]),
                    "n_pred_ids": int(res["n_pred_ids"]), "n_gt_ids": int(res["n_gt_ids"]),
                    "ratio_pred_gt": res["n_pred_ids"] / max(1, res["n_gt_ids"]),
                    "switches_per_id": res["id_switches"] / max(1, res["n_gt_ids"]),
                })

    labels = {"occlusion_duration": "Duração da oclusão (quadros)",
              "velocity_scale": "Velocidade típica (escala)",
              "n_objects": "Número de objetos"}
    rows = [("idf1", "IDF1"), ("ratio_pred_gt", "ids previstos / verdadeiros"),
            ("switches_per_id", "ID switches / id verdadeiro")]

    curves: dict[str, dict[str, list[tuple[float, float, float]]]] = {}
    print("\n  média por ponto (IDF1 | ids prev/verd | switches/id):")
    for knob, values in SWEEP_VALUES.items():
        curves[knob] = {m: [] for m, _ in rows}
        print(f"    {knob}")
        for v in values:
            sub = [r for r in results if r["knob"] == knob and r["value"] == v]
            cells = []
            for m, _ in rows:
                arr = np.array([r[m] for r in sub])
                curves[knob][m].append((v, arr.mean(), arr.std()))
                cells.append(f"{arr.mean():.3f}")
            print(f"      {v:>5}: " + " | ".join(cells))

    # sanidade: do canto fácil ao difícil o baseline tem que piorar claramente
    easy = np.mean([r["idf1"] for r in results if r["knob"] == "n_objects" and r["value"] == 3])
    hard_vel = np.mean([r["idf1"] for r in results if r["knob"] == "velocity_scale"
                        and r["value"] == max(SWEEP_VALUES["velocity_scale"])])
    occ0 = np.mean([r["switches_per_id"] for r in results
                    if r["knob"] == "occlusion_duration" and r["value"] == 0])
    occ_n = np.mean([r["switches_per_id"] for r in results
                     if r["knob"] == "occlusion_duration" and r["value"] >= 10])
    print(f"\n  IDF1: fácil (3 obj) = {easy:.3f} → velocidade máxima = {hard_vel:.3f}")
    print(f"  ID switches/id: sem oclusão = {occ0:.3f} → oclusão ≥ 10 quadros = {occ_n:.3f}")
    assert easy > hard_vel + 0.2, "velocidade alta deveria derrubar o IDF1"
    assert occ_n > occ0 + 0.05, "oclusão deveria aumentar os ID switches"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex="col")
    for col, knob in enumerate(SWEEP_VALUES):
        for row, (m, ylabel) in enumerate(rows):
            vs, mean, std = (np.array(x) for x in zip(*curves[knob][m]))
            ax = axes[row, col]
            ax.plot(vs, mean, "o-", color=f"C{row}")
            ax.fill_between(vs, mean - std, mean + std, alpha=0.2, color=f"C{row}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            if m == "idf1":
                ax.set_ylim(0, 1.05)
            if m == "ratio_pred_gt":
                ax.axhline(1.0, color="gray", ls="--", lw=1)
        axes[2, col].set_xlabel(labels[knob])
    fig.suptitle(
        f"Onde o baseline ingênuo quebra — um botão por vez a partir de {SWEEP_REF} "
        f"(detector perfeito sobre o visível; média ± desvio, {SWEEP_SEEDS} seeds)", fontsize=10)
    plt.tight_layout()
    save_figure(fig, output_dir / "parte0_parameter_sweep.png", dpi=120)

    with open(output_dir / "parte0_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  ✓ gráfico e resultados salvos")
    return results
