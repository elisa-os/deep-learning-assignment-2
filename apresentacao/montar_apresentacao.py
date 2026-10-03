"""Monta a pasta ``apresentacao/``: copia as figuras de ``outputs/`` (renomeadas na ordem da fala) e gera
as figuras extras que só existem para a apresentação (diagramas, simulador de detector, casos das métricas,
quadros da inferência). Nada aqui treina ou muda resultado: só copia e desenha.

Uso:  uv run python apresentacao/montar_apresentacao.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "apresentacao"
O = ROOT / "outputs"

# (destino, origem em outputs/)
COPIES = {
    "parte0": [
        ("0a_oclusao_por_profundidade.png", "parte0_occlusion_demo.png"),
        ("0e_onde_o_baseline_quebra.png", "parte0_parameter_sweep.png"),
    ],
    "parte1": [
        ("1a_descolamento_mAP_vs_IDF1.png", "parte1_descolamento.png"),
    ],
    "parte2": [
        ("2a_comparacao_parte1_vs_velcte_vs_rnn.png", "final/parte2_comparacao.png"),
        ("2b_iou_as_cegas.png", "final/parte2_gap_rollout.png"),
        ("2c_reconexao_apos_buracos.png", "final/parte2_reconexao.png"),
    ],
    "parte3": [
        ("3a_idf1_por_regime.png", "parte3_ablation/parte3_tracking.png"),
        ("3b_rollout_as_cegas.png", "parte3_ablation/parte3_blind_rollout.png"),
        ("3c_norma_do_gradiente.png", "parte3_ablation/parte3_grad_norms.png"),
    ],
    "parte4": [
        ("4a_gradiente_dLt_dht-k.png", "final_parte4/parte4_gradiente.png"),
        ("4b_sobrevivencia_e_oclusoes_do_dataset.png", "final_parte4/parte4_sobrevivencia.png"),
        ("4c_falha1_oclusao_longa.png", "final_parte4/parte4_falha_oclusao_longa.png"),
        ("4d_falha2_buraco_curto_camera_movel.png", "final_parte4/parte4_falha_buraco_curto_camera_movel.png"),
        ("4e_falha3_troca_entre_pessoas.png", "final_parte4/parte4_falha_troca_entre_pessoas.png"),
        ("4f_correcao_antes_depois.png", "final_parte4/parte4_correcao.png"),
    ],
    "parte5": [
        ("5a_idf1_vs_taxa_de_quadros.png", "final_parte5/parte5_idf1_fixa.png"),
        ("5b_idf1_por_tipo_de_camera.png", "final_parte5/parte5_idf1_camera.png"),
        ("5c_por_que_quebra_diagnosticos.png", "final_parte5/parte5_diagnosticos.png"),
        ("5e_multi_dt_exploratorio.png", "final_parte5/parte5_multidt.png"),
    ],
    "extras": [
        ("E1_eixo1_celula_x_janela.png", "extra_eixo1/eixo1_celula_x_janela.png"),
        ("E2_eixo1_gradiente_por_celula.png", "extra_eixo1/eixo1_gradiente.png"),
        ("E3_detector_degradado.png", "final_parte5/parte5b_detector_val.png"),
    ],
    "inferencia": [
        ("6a_MOT17-09.gif", "inferencia/MOT17-09.gif"),
    ],
}


def _box(ax, xy, w, h, text, fc="#e8f0fe", ec="#3b5bdb", fs: float = 9, bold=False):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04", fc=fc, ec=ec, lw=1.5))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def _arrow(ax, a, b, text=None):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.035, text, ha="center", fontsize=8, color="#333")


def fig_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.set_title("Pipeline completo: de detecções por quadro a identidades ao longo do tempo", fontsize=12)
    _box(ax, (0.2, 3.9), 2.3, 1.0, "det.txt (SDP)\ncaixas por quadro,\nsem identidade", "#fff4e6", "#e8590c", bold=True)
    _box(ax, (3.0, 3.9), 2.0, 1.0, "filtro de score\n≥ 0,4\n(limiar da Parte 1)", "#fff4e6", "#e8590c")
    _box(ax, (5.6, 3.9), 3.0, 1.0, "RNN de movimento\n(1 estado GRU por track)\nprevê a caixa em t+1", "#ffe3e3", "#c92a2a", bold=True)
    _box(ax, (9.2, 3.9), 3.5, 1.0, "associação: IoU(caixa prevista, detecção)\nguloso, limiar 0,3", "#e8f0fe", "#3b5bdb")
    _arrow(ax, (2.5, 4.4), (3.0, 4.4)); _arrow(ax, (5.0, 4.4), (5.6, 4.4))
    _arrow(ax, (8.6, 4.4), (9.2, 4.4))
    _box(ax, (9.2, 2.2), 3.5, 1.1, "casou  → o estado da track recebe a\ndetecção (observada = 1)", "#ebfbee", "#2b8a3e")
    _box(ax, (9.2, 0.7), 3.5, 1.1, "não casou → o estado roda com a própria\nprevisão (observada = 0): oclusão", "#ebfbee", "#2b8a3e")
    _arrow(ax, (10.95, 3.9), (10.95, 3.3)); _arrow(ax, (10.95, 2.2), (10.95, 1.8))
    xs, ys = [10.95, 10.95, 8.9, 8.9, 7.1], [0.7, 0.4, 0.4, 3.55, 3.55]
    ax.plot(xs, ys, color="#2b8a3e", lw=1.4)
    ax.annotate("", xy=(7.1, 3.9), xytext=(7.1, 3.55), arrowprops=dict(arrowstyle="->", lw=1.4, color="#2b8a3e"))
    ax.text(7.2, 3.1, "estado e caixa atualizados\nvoltam à RNN no quadro seguinte", ha="center", fontsize=8.5, color="#2b8a3e")
    _box(ax, (0.2, 0.7), 4.2, 1.9, "gestão de tracks (a mesma da Parte 1)\n• nascimento: detecção sem par vira track\n  "
         "tentativa; ganha id após min_hits = 3\n• morte: sem observação por > max_age = 30\n  quadros; ids nunca reaproveitados",
         "#f3f0ff", "#5f3dc4", fs=8.5)
    _box(ax, (4.8, 0.7), 3.6, 1.9, "saída: tracks MOT (frame, id, caixa)\n→ IDF1, ID switches, fragmentações\n(implementação própria,\n"
         "pa2/metrics/tracking.py)", "#f3f0ff", "#5f3dc4", fs=8.5)
    _arrow(ax, (4.4, 1.65), (4.8, 1.65))
    ax.text(0.2, 0.05, "Mesma associação e gestão em TODAS as comparações; só muda onde a track espera o objeto: "
            "Parte 1 = última caixa vista; velocidade constante = extrapolação linear; Parte 2 = RNN.",
            fontsize=8.5, style="italic")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_arquitetura(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")
    ax.set_title("A RNN de movimento (Trilha A): um passo de um track", fontsize=12)
    _box(ax, (0.1, 1.5), 3.4, 2.4, "entrada (10 números)\n• Δ da caixa vs. entrada anterior (4):\n  Δcx/w, Δcy/h, Δlog w, Δlog h\n"
         "• log(h/H), log(w/h)\n• posição normalizada cx/W, cy/H\n• Δt (= 1 nos treinos)\n• flag observada (1 = detecção,\n  0 = própria previsão)",
         "#fff4e6", "#e8590c", fs=8.5)
    _box(ax, (4.1, 2.0), 2.2, 1.4, "GRUCell\n64 unidades\n(estado h_t da track)", "#ffe3e3", "#c92a2a", bold=True)
    _box(ax, (6.9, 2.0), 2.4, 1.4, "cabeça MLP\n64 → 64 (ReLU) → 4\n(última camada zerada)", "#e8f0fe", "#3b5bdb")
    _box(ax, (9.9, 1.5), 3.0, 2.4, "saída o = (ox, oy, ow, oh)\ncaixa seguinte:\ncx' = cx + w·ox/10\ncy' = cy + h·oy/10\nw' = w·exp(ow/10)\nh' = h·exp(oh/10)",
         "#ebfbee", "#2b8a3e", fs=8.5)
    _arrow(ax, (3.5, 2.7), (4.1, 2.7)); _arrow(ax, (6.3, 2.7), (6.9, 2.7)); _arrow(ax, (9.3, 2.7), (9.9, 2.7))
    ax.plot([6.0, 6.0, 4.4], [2.0, 1.3, 1.3], color="#c92a2a", lw=1.4)
    ax.annotate("", xy=(4.4, 2.0), xytext=(4.4, 1.3), arrowprops=dict(arrowstyle="->", lw=1.4, color="#c92a2a"))
    ax.text(5.2, 0.95, "h_t volta como estado inicial do próximo quadro\n(a memória do track)", ha="center", fontsize=8.5, color="#c92a2a")
    ax.text(0.1, 0.1, "~19 mil parâmetros. Perda: smooth-L1 do erro da caixa prevista vs. GT, em unidades da caixa [dcx/w, dcy/h, log w, log h]·10. "
            "Treino: janelas de T = 32 quadros do GT com ruído de detector.",
            fontsize=8.5, style="italic")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_split(path: Path) -> None:
    df = pd.read_csv(O / "parte1_sequences.csv")
    df["split"] = df["split"].replace({"train": "treino", "val": "VALIDAÇÃO"})
    df = df.sort_values(["split", "sequence"], ascending=[False, True])
    cols = ["sequence", "split", "camera", "resolution", "fps", "frames", "gt_ids", "peds_per_frame", "mean_visibility"]
    head = ["vídeo", "split", "câmera", "resolução", "fps", "quadros", "ids GT", "ped./quadro", "visib. média"]
    cell = [[r["sequence"], r["split"], r["camera"], r["resolution"], int(r["fps"]), int(r["frames"]), int(r["gt_ids"]),
             f"{r['peds_per_frame']:.1f}", f"{r['mean_visibility']:.2f}"] for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(11, 3.4)); ax.axis("off")
    t = ax.table(cellText=cell, colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.5)
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        if r["split"] == "VALIDAÇÃO":
            for j in range(len(head)):
                t[i, j].set_facecolor("#ffe3e3")
    ax.set_title("Split por VÍDEO: validação = 09 (câmera parada, esparso) e 13 (câmera móvel, alta rotatividade de ids)", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_simulador(path: Path) -> None:
    from pa2.synthetic_video import SimulatedDetector, generate_synthetic_sequence
    rng = np.random.default_rng(3)
    seq = generate_synthetic_sequence(rng, num_frames=45, n_objects=7, occlusion_duration=0)
    f = 20
    gt = [b for b in seq.true_boxes if b["frame"] == f]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    configs = [("ground truth", None), ("descarta 30%", dict(drop_rate=0.3, noise_std=0, fp_rate=0)),
               ("ruído 3 px", dict(drop_rate=0, noise_std=3.0, fp_rate=0)),
               ("falsos positivos (2/quadro)", dict(drop_rate=0, noise_std=0, fp_rate=2.0))]
    for ax, (name, cfg) in zip(axes, configs):
        ax.imshow(seq.frames[f - 1]); ax.axis("off"); ax.set_title(name, fontsize=10)
        if cfg is None:
            boxes, col, fp = gt, "#00e676", []
            for b in boxes:
                ax.add_patch(Rectangle((b["bb_left"], b["bb_top"]), b["bb_width"], b["bb_height"], fill=False, ec=col, lw=1.6))
        else:
            ds = SimulatedDetector(rng=np.random.default_rng(11), **cfg).detect(seq.true_boxes, 128, seq.num_frames)
            for b in gt:
                ax.add_patch(Rectangle((b["bb_left"], b["bb_top"]), b["bb_width"], b["bb_height"], fill=False,
                                       ec="#00e676", lw=0.8, ls=":"))
            for d in [d for d in ds if d["frame"] == f]:
                ax.add_patch(Rectangle((d["bb_left"], d["bb_top"]), d["bb_width"], d["bb_height"], fill=False,
                                       ec="#ff1744" if d["is_fp"] else "#ffea00", lw=1.8))
    fig.suptitle("Simulador de detector: verde pontilhado = GT; amarelo = detecção estragada; vermelho = falso positivo", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_casos(path: Path) -> None:
    from pa2.metrics.cases import EXPECTED, case_a, case_b, case_c, make_gt
    from pa2.metrics.tracking import evaluate_tracking_sequence
    desc = {"a": "predição = GT", "b": "ids 1↔2 trocados a partir do quadro 16", "c": "track 1 partida no quadro 10 + sem detecção em 12–14"}
    rows = []
    for k, fn in (("a", case_a), ("b", case_b), ("c", case_c)):
        pred, gt = fn()
        r = evaluate_tracking_sequence(pred, gt)
        e = EXPECTED[k]
        rows.append([f"({k}) {desc[k]}", f"{e['idf1']:.4f}", f"{r['idf1']:.4f}", e["id_switches"], r["id_switches"],
                     e["fragmentations"], r["fragmentations"]])
    head = ["caso", "IDF1 esperado\n(à mão)", "IDF1 obtido", "switches\nesperado", "switches\nobtido", "frag.\nesperado", "frag.\nobtido"]
    fig, ax = plt.subplots(figsize=(12, 2.6)); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.8)
    t.auto_set_column_width(list(range(len(head))))
    ax.set_title("As métricas próprias passam nos 3 casos construídos à mão ((b) e (c) dão IDF1 diferentes, como pede o enunciado)", fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_sdp_vs_torchvision(path: Path) -> None:
    sdp = pd.read_csv(O / "parte1_per_sequence_public.csv").set_index("sequence")
    tv = pd.read_csv(O / "parte1_per_sequence_torchvision.csv").set_index("sequence")
    order = sdp.sort_values("peds_per_frame").index
    labels = [f"{q[-2:]}{'*' if sdp.loc[q, 'split'] == 'val' else ''}" for q in order]
    panels = [("mAP", "mAP (IoU 0,5:0,95)"), ("AP50", "AP50"), ("IDF1", "IDF1"), ("ids_pred/gt", "ids previstos / verdadeiros")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    x = np.arange(len(order))
    for ax, (col, ttl) in zip(axes, panels):
        ax.bar(x - 0.2, sdp.loc[order, col], 0.4, label="SDP (público)", color="#c92a2a")
        ax.bar(x + 0.2, tv.loc[order, col], 0.4, label="torchvision (Faster R-CNN COCO)", color="#3b5bdb")
        ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3, axis="y")
        if col == "ids_pred/gt":
            ax.axhline(1, color="k", ls="--", lw=1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Parte 1 — as duas fontes de detecção, mesma associação (vídeos por densidade; * = validação)", fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)


def figs_inferencia(out_dir: Path) -> None:
    from pa2.inference import run_inference, sample_frames
    result = run_inference("data/MOT17/train/MOT17-09-SDP")
    frames = [max(1, int(round(result.n_frames * q))) for q in (0.1, 0.35, 0.65, 0.9)]
    imgs = sample_frames(result, frames, scale=0.35, show_gt=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 2.9))
    for ax, im, f in zip(axes, imgs, frames):
        ax.imshow(im); ax.set_title(f"quadro {f}", fontsize=9); ax.axis("off")
    fig.suptitle("Inferência no MOT17-09: cada id mantém a cor; cinza = GT", fontsize=10)
    fig.savefig(out_dir / "6c_quadros_inferencia.png", dpi=130, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    for sub, items in COPIES.items():
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        for dst, src in items:
            shutil.copy(O / src, d / dst)
    (OUT / "visao_geral").mkdir(exist_ok=True)
    fig_pipeline(OUT / "visao_geral" / "00a_pipeline.png")
    fig_arquitetura(OUT / "visao_geral" / "00b_arquitetura_rnn.png")
    fig_split(OUT / "visao_geral" / "00c_split_e_videos.png")
    fig_sdp_vs_torchvision(OUT / "parte1" / "1c_sdp_vs_torchvision.png")
    fig_simulador(OUT / "parte0" / "0b_simulador_de_detector.png")
    fig_casos(OUT / "parte0" / "0c_casos_das_metricas.png")
    figs_inferencia(OUT / "inferencia")
    print("pronto:", OUT)


if __name__ == "__main__":
    main()
