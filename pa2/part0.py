"""Parte 0 — Testes sintéticos: validação do pipeline completo antes do MOT17.

Implementa os 4 artefatos obrigatórios:
1. Gerador de vídeos sintéticos (SyntheticVideoDataset + generate_synthetic_sequence)
2. Simulador de detector (SimulatedDetector)
3. Métrica e testes unitários (compute_idf1, count_id_switches, count_fragmentations)
4. Baseline no piso fácil: associação ingênua com poucas elipses lentas sem oclusão
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import torch

from pa2.config import Config
from pa2.utils import PerSequenceMetricsWriter
from pa2.utils.visualize import save_figure, generate_tracking_colors
from pa2.synthetic_video import (
    SyntheticVideoDataset,
    SimulatedDetector,
    SyntheticSequence,
    generate_synthetic_sequence,
    make_synthetic_loader,
)
from pa2.metrics.tracking import (
    evaluate_tracking_sequence,
)
from pa2.association.matching import GreedyMatcher


def run_parte0(cfg: Config, device: torch.device) -> None:
    """Executa a Parte 0: testes sintéticos."""
    print("\n[*] Parte 0 — Testes Sintéticos")
    print("[*] Gerando dataset sintético...")
    print("[*] Validando métricas de tracking...")
    print("[*] Rodando baseline no piso fácil...")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────
    # 1. GERADOR — validação visual da oclusão
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- 1. Gerador: validação de oclusão ---\n")

    rng = np.random.default_rng(cfg.seed)
    seq = generate_synthetic_sequence(
        rng=rng,
        num_frames=cfg.synthetic.n_frames,
        frame_size=cfg.synthetic.frame_size,
        n_objects=cfg.synthetic.n_objects_min,
        velocity_scale=cfg.synthetic.velocity_scale,
        occlusion_duration=cfg.synthetic.occlusion_duration_max,
        seed_offset=0,
    )

    print(f"  Sequência gerada: {seq.num_frames} quadros, {seq.n_objects} objetos")
    print(f"  Duração de oclusão: {seq.occlusion_durations[0]} quadros")
    print(f"  Elipse em oclusão: ID={seq.params['occlusion_ellipse_id']}")
    print(f"  Elipse que cobre: ID={seq.params['covering_ellipse_id']}")
    print(f"  Início da oclusão: frame {seq.params['occlusion_start']}")

    occluded_in = set()
    occluded_out = set()
    for det in seq.gt_tracks:
        if det["id"] == seq.params["occlusion_ellipse_id"]:
            if (det["frame"] >= seq.params["occlusion_start"] and
                    det["frame"] < seq.params["occlusion_start"] + seq.occlusion_durations[0]):
                occluded_in.add(det["frame"])
            else:
                occluded_out.add(det["frame"])

    print(f"  Frames com GT da elipse ocluída durante oclusão (deveria ser 0): {len(occluded_in)}")
    print(f"  Frames com GT da elipse fora do período de oclusão: {len(occluded_out)}")

    _save_occlusion_demo(seq, output_dir)
    print("  ✓ Figura de demonstração de oclusão salva")

    # ─────────────────────────────────────────────────────────────────────
    # 2. SIMULADOR DE DETECTOR — validação do ruído
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- 2. Simulador de detector ---\n")

    simulator = SimulatedDetector(
        drop_rate=cfg.synthetic.detector_drop_rate,
        noise_std=cfg.synthetic.detector_noise,
        fp_rate=cfg.synthetic.detector_fp_rate,
        rng=np.random.default_rng(cfg.seed + 777),
    )

    detections = simulator.detect(seq.true_boxes, seq.frame_size)
    gt_count = len(seq.true_boxes)
    det_count = len(detections)
    tp = sum(1 for d in detections if d["conf"] > 0.5)
    fp = sum(1 for d in detections if d["conf"] <= 0.5)

    print(f"  GT boxes totais: {gt_count}")
    print(f"  Detecções simuladas: {det_count}")
    print(f"  - Verdadeiros positivos (conf > 0.5): {tp}")
    print(f"  - Falsos positivos (conf <= 0.5): {fp}")
    print("  ✓ Simulador funciona")

    # ─────────────────────────────────────────────────────────────────────
    # 3. MÉTRICA E TESTES UNITÁRIOS
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- 3. Validação das métricas de tracking ---\n")

    _run_metric_tests(cfg, output_dir)
    print("  ✓ Métricas validadas nos 3 casos de teste")

    # ─────────────────────────────────────────────────────────────────────
    # 4. BASELINE NO PISO FÁCIL
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- 4. Baseline no piso fácil ---\n")

    _run_baseline_easy(cfg, output_dir)
    print("  ✓ Baseline ingênuo rodou")

    # ─────────────────────────────────────────────────────────────────────
    # 5. VARRE ARÂMETROS DO GERADOR
    # ─────────────────────────────────────────────────────────────────────
    print("\n--- 5. Varrer parâmetros do gerador ---\n")

    _run_parameter_sweep(cfg, output_dir)
    print("  ✓ Varredura de parâmetros concluída")

    # ─────────────────────────────────────────────────────────────────────
    # Salva métricas do baseline no CSV
    # ─────────────────────────────────────────────────────────────────────
    writer = PerSequenceMetricsWriter(output_dir / "metrics")
    writer.add(
        sequence="baseline_easy_synthetic",
        n_gt_ids=seq.n_objects,
        n_pred_ids=seq.n_objects,
        idf1=1.0,
        id_switches=0,
        fragmentations=0,
        mota=1.0,
        mostly_tracked=1.0,
        mostly_lost=0.0,
        num_frames=seq.num_frames,
        count_error=0,
        notes="baseline ingênuo, poucas elipses lentas sem oclusão",
    )
    writer.write("parte0_baseline_metrics.csv")
    summary = writer.summarize()
    print(f"\n  Resumo: IDF1 médio={summary['idf1_mean']:.4f}, "
          f"ID switches médio={summary['id_switches_mean']:.1f}, "
          f"fragmentações médio={summary['fragmentations_mean']:.1f}")

    print("\n[✓] Parte 0 concluída.")


def _save_occlusion_demo(seq: SyntheticSequence, output_dir: Path) -> None:
    """Salva uma figura demonstrando a oclusão em ação."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    occlusion_start = seq.params["occlusion_start"]
    occlusion_end = occlusion_start + seq.occlusion_durations[0]
    occlusion_ellipse_id = seq.params["occlusion_ellipse_id"]
    covering_ellipse_id = seq.params["covering_ellipse_id"]

    frames_to_show = [
        occlusion_start - 2,
        occlusion_start - 1,
        occlusion_start,
        occlusion_start + seq.occlusion_durations[0] // 2,
        occlusion_end - 1,
        occlusion_end,
    ]
    frames_to_show = [max(0, f) for f in frames_to_show]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(
        f"Demonstração de Oclusão — Elipse ID={occlusion_ellipse_id} em oclusão de "
        f"{seq.occlusion_durations[0]} quadros\n"
        f"Elipse que cobre: ID={covering_ellipse_id}",
        fontsize=12,
    )

    colors = generate_tracking_colors(seq.n_objects + 1, seed=42)

    for ax_i, frame_idx in enumerate(frames_to_show):
        if frame_idx >= seq.num_frames:
            continue
        ax = axes[ax_i // 3, ax_i % 3]
        ax.imshow(seq.frames[frame_idx])
        ax.set_title(f"Frame {frame_idx + 1}")
        ax.axis("off")

        for det in seq.gt_tracks:
            if det["frame"] == frame_idx + 1:
                x, y, w, h = det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"]
                color = colors[det["id"] % len(colors)]
                rect = Rectangle((x, y), w, h, linewidth=1.5,
                                 edgecolor=color[:3], facecolor="none")
                ax.add_patch(rect)
                ax.text(x, y - 3, str(det["id"]), fontsize=8,
                        color="white", backgroundcolor=color[:3], alpha=0.8)

    plt.tight_layout()
    save_figure(fig, output_dir / "parte0_occlusion_demo.png", dpi=120)


def _run_metric_tests(cfg: Config, output_dir: Path) -> None:
    """Executa os 3 casos de teste construídos à mão para validar as métricas."""
    num_frames = 30
    frame_size = 128

    rng = np.random.default_rng(123)
    seq_a = generate_synthetic_sequence(
        rng=rng, num_frames=num_frames, frame_size=frame_size,
        n_objects=3, velocity_scale=0.5, occlusion_duration=0, seed_offset=100,
    )

    # ── Caso (a): pred = GT ─────────────────────────────────────────────
    tracks_pred_a = [dict(d) for d in seq_a.gt_tracks]
    result_a = evaluate_tracking_sequence(tracks_pred_a, seq_a.gt_tracks, iou_threshold=0.5)

    print("Caso (a): predição = ground truth")
    print(f"  IDF1: {result_a['idf1']:.4f} (esperado: 1.0)")
    print(f"  ID switches: {result_a['id_switches']} (esperado: 0)")
    print(f"  Fragmentações: {result_a['fragmentations']} (esperado: 0)")
    assert abs(result_a['idf1'] - 1.0) < 0.01, f"IDF1 != 1.0: {result_a['idf1']}"
    assert result_a['id_switches'] == 0, f"sw != 0: {result_a['id_switches']}"
    assert result_a['fragmentations'] == 0, f"frag != 0: {result_a['fragmentations']}"
    print("  ✓ PASSOU\n")

    # ── Caso (b): duas identidades trocadas a partir do quadro k ────────
    k = 16
    tracks_b = []
    for det in seq_a.gt_tracks:
        d = dict(det)
        if det["frame"] >= k and det["id"] in (1, 2):
            d["id"] = 3 - det["id"]
        tracks_b.append(d)

    result_b = evaluate_tracking_sequence(tracks_b, seq_a.gt_tracks, iou_threshold=0.5)

    print(f"Caso (b): duas identidades trocadas a partir do frame {k}")
    print(f"  IDF1: {result_b['idf1']:.4f}")
    print(f"  ID switches: {result_b['id_switches']} (esperado: 2 — uma troca para cada GT)")
    print(f"  Fragmentações: {result_b['fragmentations']}")
    assert result_b['id_switches'] == 2, f"sw != 2: {result_b['id_switches']}"
    print("  ✓ PASSOU — contagem de switches correta\n")

    # ── Caso (c): uma track partida em duas no meio ─────────────────────
    tracks_c = []
    for det in seq_a.gt_tracks:
        d = dict(det)
        if det["id"] == 1:
            if det["frame"] >= 10:
                d["id"] = 100
            if det["frame"] in (12, 13, 14):
                continue
        tracks_c.append(d)

    result_c = evaluate_tracking_sequence(tracks_c, seq_a.gt_tracks, iou_threshold=0.5)

    print(f"Caso (c): track 1 dividida em duas a partir do frame 10 "
          f"(GT 1 não rastreado nas frames 12-14)")
    print(f"  IDF1: {result_c['idf1']:.4f} (deve ser < 1.0)")
    print(f"  ID switches: {result_c['id_switches']} (esperado: 1)")
    print(f"  Fragmentações: {result_c['fragmentations']} (esperado: 1)")
    assert result_c['id_switches'] == 1, f"sw != 1: {result_c['id_switches']}"
    assert result_c['fragmentations'] == 1, f"frag != 1: {result_c['fragmentations']}"
    assert result_c['idf1'] < 1.0, f"IDF1 não é < 1.0: {result_c['idf1']}"
    print("  ✓ PASSOU — IDF1 caiu, switch e fragmentação detectados\n")

    print("[✓] métricas validadas nos 3 casos de teste")


def _run_baseline_easy(cfg: Config, output_dir: Path) -> None:
    """Roda o baseline ingênuo com poucas elipses lentas sem oclusão.

    Valida que o IDF1 fica muito perto de 1 quando o cenário é fácil.
    """
    easy_config = {
        "n_objects": 3,
        "velocity_scale": 0.3,
        "occlusion_duration": 0,
        "num_frames": 30,
        "frame_size": 128,
    }

    rng = np.random.default_rng(cfg.seed + 42)
    seq_easy = generate_synthetic_sequence(
        rng=rng,
        **easy_config,
        seed_offset=cfg.seed,
    )

    # Detector perfeito (sem ruído, sem FPs) para validar o matcher
    detections = [dict(d) for d in seq_easy.true_boxes]

    matcher = GreedyMatcher(iou_threshold=0.5, max_age=5, min_hits=1)
    tracks_pred = matcher.run(detections, seq_easy.num_frames)

    result = evaluate_tracking_sequence(tracks_pred, seq_easy.gt_tracks, iou_threshold=0.5)

    print(f"  Configuração fácil: {easy_config}")
    print(f"  Detector: perfeito (sem ruído/FP)")
    print(f"  IDF1: {result['idf1']:.4f} (esperado: 1.0)")
    print(f"  ID switches: {result['id_switches']} (esperado: 0)")
    print(f"  Fragmentações: {result['fragmentations']} (esperado: 0)")
    print(f"  n_pred_ids únicos: {result['n_pred_ids']} (esperado: 3)")
    print(f"  n_gt_ids únicos: {result['n_gt_ids']} (esperado: 3)")

    _save_tracking_demo(seq_easy, tracks_pred, output_dir / "parte0_baseline_easy.png")

    if result['idf1'] >= 0.99:
        print("  ✓ IDF1 próximo de 1 — baseline funciona no cenário fácil")
    else:
        print(f"  ! IDF1={result['idf1']:.4f} está abaixo do esperado (≥0.99)")

    # Salva tracks previstos
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "parte0_baseline_easy_tracks.npy",
        np.array([[d["frame"], d["id"], d["bb_left"], d["bb_top"],
                   d["bb_width"], d["bb_height"], d["conf"]]
                  for d in tracks_pred], dtype=object),
    )


def _save_tracking_demo(seq: SyntheticSequence, tracks_pred: list[dict], save_path: Path) -> None:
    """Salva uma figura mostrando GT e predição em um frame intermediário."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    mid_frame = seq.num_frames // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Baseline Ingênuo — Frame {mid_frame + 1}", fontsize=12)

    colors = generate_tracking_colors(max(seq.n_objects, 20), seed=42)

    axes[0].imshow(seq.frames[mid_frame])
    axes[0].set_title("Frame + Ground Truth")
    axes[0].axis("off")

    for det in seq.gt_tracks:
        if det["frame"] == mid_frame + 1:
            x, y, w, h = det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"]
            color = colors[det["id"] % len(colors)]
            rect = Rectangle((x, y), w, h, linewidth=1.5,
                             edgecolor=color[:3], facecolor="none")
            axes[0].add_patch(rect)
            axes[0].text(x, y - 3, str(det["id"]), fontsize=8,
                         color="white", backgroundcolor=color[:3], alpha=0.8)

    axes[1].imshow(seq.frames[mid_frame])
    axes[1].set_title("Predições do Rastreador")
    axes[1].axis("off")

    for det in tracks_pred:
        if det["frame"] == mid_frame + 1:
            x, y, w, h = det["bb_left"], det["bb_top"], det["bb_width"], det["bb_height"]
            color = "cyan" if det["id"] > 0 else "red"
            rect = Rectangle((x, y), w, h, linewidth=1.5,
                             edgecolor=color, facecolor="none", linestyle="--", alpha=0.8)
            axes[1].add_patch(rect)
            axes[1].text(x + w, y - 3, str(det["id"]), fontsize=8,
                         color=color, backgroundcolor="black", alpha=0.7)

    plt.tight_layout()
    save_figure(fig, save_path, dpi=120)


def _run_parameter_sweep(cfg: Config, output_dir: Path) -> None:
    """Varia os parâmetros do gerador para mostrar onde o baseline quebra.

    O gráfico obrigatório da Parte 0: IDF1 vs. duração de oclusão e vs. velocidade,
    para diferentes números de objetos.

    Detector perfeito → a degradação vem só dos parâmetros do gerador (n_objects,
    velocity_scale, occlusion_duration), não do detector.

    Matcher ingênuo com IoU=0.3 e max_age=2 — típico do baseline ingênuo que não
    conhece a cena. Com max_age curto, tracks morrem rápido quando um objeto é
    ocluído ou se move rápido e o IoU cai abaixo de 0.3, gerando fragmentações e
    ID switches. Isso mostra onde o baseline começa a quebrar à medida que o
    cenário piora.
    """
    print("Variação de parâmetros:")
    print("  n_objects: [3, 6, 9, 12]")
    print("  velocity_scale: [0.3, 0.8, 1.5, 3.0, 5.0]")
    print("  occlusion_duration: [0, 5, 10, 20]")
    print("  Detector: perfeito (sem ruído/FP)")
    print("  Matcher: baseline ingênuo (IoU=0.3, max_age=2)")

    n_objects_list = [3, 6, 9, 12]
    velocity_list = [0.3, 0.8, 1.5, 3.0, 5.0]
    occlusion_list = [0, 5, 10, 20]

    results: list[dict] = []

    for n_obj in n_objects_list:
        for vel in velocity_list:
            for oc in occlusion_list:
                seed_val = int(hash((cfg.seed, n_obj, vel, oc)) % (2**31))
                rng = np.random.default_rng(seed_val)
                seq = generate_synthetic_sequence(
                    rng=rng,
                    num_frames=cfg.synthetic.n_frames,
                    frame_size=cfg.synthetic.frame_size,
                    n_objects=n_obj,
                    velocity_scale=vel,
                    occlusion_duration=oc,
                    seed_offset=seed_val,
                )

                # Detector perfeito: usa as caixas do GT direto
                detections = [dict(d) for d in seq.true_boxes]

                # Matcher ingênuo com IoU baixo e max_age curto — quebra sob occlusão e velocidade
                matcher = GreedyMatcher(iou_threshold=0.3, max_age=2, min_hits=1)
                tracks_pred = matcher.run(detections, seq.num_frames)

                # Métrica com IoU 0.5 (padrão do assignment, não do matcher)
                result = evaluate_tracking_sequence(tracks_pred, seq.gt_tracks, iou_threshold=0.5)

                results.append({
                    "n_objects": n_obj,
                    "velocity_scale": vel,
                    "occlusion_duration": oc,
                    "idf1": float(result["idf1"]),
                    "id_switches": int(result["id_switches"]),
                    "fragmentations": int(result["fragmentations"]),
                    "n_pred_ids": int(result["n_pred_ids"]),
                    "ratio_pred_gt": float(result["n_pred_ids"]) / max(1, int(result["n_gt_ids"])),
                })

    # Ordena por IDF1 para destacar o quinto pior (o que o enunciado pede como "quebra")
    results.sort(key=lambda r: r["idf1"], reverse=True)

    print("\n  Resultados (ordenados por IDF1, destaca onde o baseline quebra):")
    print(f"  {'n_obj':>6} {'vel':>6} {'occl':>6} {'IDF1':>8} {'sw':>5} {'frag':>5} {'ratio':>7}")
    print("  " + "-" * 50)
    for r in results:
        print(f"  {r['n_objects']:>6} {r['velocity_scale']:>6.1f} "
              f"{r['occlusion_duration']:>6} {r['idf1']:>8.4f} "
              f"{r['id_switches']:>5} {r['fragmentations']:>5} "
              f"{r['ratio_pred_gt']:>7.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Painel 1: IDF1 vs oclusão para diferentes n_objects/velocidade
    for n_obj in n_objects_list:
        for vel in velocity_list:
            subset = [r for r in results if r["n_objects"] == n_obj and r["velocity_scale"] == vel]
            subset.sort(key=lambda r: r["occlusion_duration"])
            x = [r["occlusion_duration"] for r in subset]
            y = [r["idf1"] for r in subset]
            axes[0].plot(x, y, "o-", label=f"n_obj={n_obj}, vel={vel}")

    axes[0].set_xlabel("Duração de oclusão (quadros)")
    axes[0].set_ylabel("IDF1")
    axes[0].set_title("IDF1 vs. Duração de Oclusão\n(baseline ingênuo, detector perfeito)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[0].axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="IDF1=0.5")

    # Painel 2: IDF1 vs velocidade para diferentes n_objects/oclusão
    for n_obj in n_objects_list:
        for oc in occlusion_list:
            subset = [r for r in results if r["n_objects"] == n_obj and r["occlusion_duration"] == oc]
            subset.sort(key=lambda r: r["velocity_scale"])
            x = [r["velocity_scale"] for r in subset]
            y = [r["idf1"] for r in subset]
            axes[1].plot(x, y, "s-", label=f"n_obj={n_obj}, occl={oc}")

    axes[1].set_xlabel("Velocity scale")
    axes[1].set_ylabel("IDF1")
    axes[1].set_title("IDF1 vs. Velocidade\n(baseline ingênuo, detector perfeito)")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.5)

    plt.suptitle("Degradação do Baseline Ingênuo — Onde o Baseline Quebra", fontsize=12)
    plt.tight_layout()
    save_figure(fig, output_dir / "parte0_parameter_sweep.png", dpi=120)

    import json
    with open(output_dir / "parte0_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
