"""Visualização e exportação de gráficos de avaliação e trajetórias (PA2)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
if "inline" not in matplotlib.get_backend().lower():
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig: plt.Figure, filepath: str | Path, dpi: int = 100) -> Path:
    dest = Path(filepath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.is_file():
        dest.unlink()
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return dest


def generate_tracking_colors(n: int, seed: int = 42) -> np.ndarray:
    """Gera N cores distintas para identidades/tracks.

    Parâmetros
    ----------
    n : int
        Número de identidades/tracks a colorir.
    seed : int
        Seed para reprodutibilidade das cores.

    Retorna
    -------
    np.ndarray
        Array de shape (n+1, 4) com cores RGBA. Índice 0 é transparente (fundo).
    """
    rng = np.random.default_rng(seed=seed)
    colors = rng.random((n + 1, 4))
    colors[:, 3] = 0.6
    colors[0] = [0, 0, 0, 0]
    return colors


def show_tracking_frame(
    frame_img: np.ndarray,
    gt_tracks: np.ndarray,
    pred_tracks: np.ndarray | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
    gt_colors: np.ndarray | None = None,
    pred_colors: np.ndarray | None = None,
) -> None:
    """Desenha um único quadro com caixas de ground truth e/ou predição coloridas por ID.

    Parâmetros
    ----------
    frame_img : np.ndarray
        Imagem do quadro (H, W, 3) ou (H, W) grayscale.
    gt_tracks : np.ndarray
        Array (N_gt, 5) com colunas [frame, id, bb_left, bb_top, bb_width, bb_height]
        filtrado para este frame, ou caixas shape (N, 4) + ids separados.
        A interface espera caixas + ids como entradas separadas.
    pred_tracks : np.ndarray | None
        Mesma estrutura que gt_tracks, para predições.
    title : str
        Título do subplot.
    ax : plt.Axes | None
        Eixo onde desenhar. Se None, cria novo.
    gt_colors : np.ndarray | None
        Cores RGBA para cada ID de GT.
    pred_colors : np.ndarray | None
        Cores RGBA para cada ID de predição.
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.imshow(frame_img, cmap="gray" if frame_img.ndim == 2 else None)

    # Desenha GT
    if gt_tracks is not None:
        if gt_tracks.ndim == 2 and gt_tracks.shape[1] >= 5:
            # Formato dog: [frame, id, bb_left, bb_top, bb_width, bb_height, ...]
            ids = gt_tracks[:, 1].astype(int)
            boxes = gt_tracks[:, 2:6]  # left, top, w, h
        else:
            # Assume already boxes + ids separados via kwargs — não usar
            boxes = gt_tracks
            ids = np.arange(len(boxes))

        if gt_colors is not None:
            for i, (box, id_) in enumerate(zip(boxes, ids)):
                left, top, w, h = box
                color = gt_colors[min(id_, len(gt_colors) - 1)]
                rect = plt.Rectangle(
                    (left, top), w, h,
                    linewidth=1.5, edgecolor=color[:3], facecolor=color, alpha=color[3],
                )
                ax.add_patch(rect)
                ax.text(left, top - 2, str(id_), fontsize=6,
                        color='white', backgroundcolor=color[:3], alpha=0.8)

    # Desenha predições
    if pred_tracks is not None and pred_colors is not None:
        if pred_tracks.ndim == 2 and pred_tracks.shape[1] >= 5:
            ids = pred_tracks[:, 1].astype(int)
            boxes = pred_tracks[:, 2:6]
        else:
            boxes = pred_tracks
            ids = np.arange(len(boxes))

        for i, (box, id_) in enumerate(zip(boxes, ids)):
            left, top, w, h = box
            color = pred_colors[min(id_, len(pred_colors) - 1)]
            rect = plt.Rectangle(
                (left, top), w, h,
                linewidth=1.5, edgecolor=color[:3], facecolor='none',
                linestyle='--', alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(left + w, top - 2, str(id_), fontsize=6,
                    color='cyan', backgroundcolor='black', alpha=0.7)

    ax.set_title(title)
    ax.axis("off")


def plot_idf1_vs_frames(
    sequences: list[str],
    idf1_values: Sequence[float],
    save_path: str | Path,
    title: str = "IDF1 por Sequência",
) -> Path:
    """Barra horizontal de IDF1 por sequência, ordenada por valor."""
    fig, ax = plt.subplots(figsize=(8, max(3, len(sequences) * 0.4)))

    sorted_idx = np.argsort(idf1_values)
    sorted_seqs = [sequences[i] for i in sorted_idx]
    sorted_vals = [idf1_values[i] for i in sorted_idx]

    y_pos = range(len(sorted_seqs))
    ax.barh(y_pos, sorted_vals, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_seqs, fontsize=9)
    ax.set_xlabel("IDF1")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3, label="IDF1=0.5")
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, save_path)


def plot_comparison_baseline_vs_temporal(
    sequences: list[str],
    baseline_idf1: Sequence[float],
    temporal_idf1: Sequence[float],
    save_path: str | Path,
    title: str = "Comparação: Baseline por Quadro vs. Modelo Temporal",
) -> Path:
    """Gráfico de barras lado a lado: baseline vs modelo temporal por sequência."""
    fig, ax = plt.subplots(figsize=(10, max(3, len(sequences) * 0.4)))

    x = np.arange(len(sequences))
    width = 0.35

    ax.bar(x - width/2, baseline_idf1, width, label='Baseline (Parte 1)',
           color='gray', alpha=0.7, edgecolor='black')
    ax.bar(x + width/2, temporal_idf1, width, label='Modelo Temporal (Parte 2)',
           color='steelblue', alpha=0.8, edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("IDF1")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return save_figure(fig, save_path)


def plot_gradient_vanishing_curve(
    k_values: Sequence[int],
    gradient_norms: Sequence[float],
    save_path: str | Path,
    title: str = "Horizonte de Memória Analítico: ||∂L_t/∂h_{t−k}||",
    model_label: str = "Modelo",
) -> Path:
    """Plota a norma do gradiente em função da distância temporal k.

    Parâmetros
    ----------
    k_values : Sequence[int]
        Distâncias temporais (k = 0, 1, 2, ...).
    gradient_norms : Sequence[float]
        ||∂L_t/∂h_{t−k}|| para cada k.
    save_path : str | Path
        Caminho para salvar a figura.
    title : str
        Título do gráfico.
    model_label : str
        Rótulo para a curva (ex: "RNN simples", "LSTM", "GRU").
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(k_values, gradient_norms, 'o-', label=model_label, linewidth=2, markersize=6)
    ax.set_xlabel("k (passos de tempo remotos)")
    ax.set_ylabel("||∂L_t/∂h_{t−k}||")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.legend()
    ax.axhline(y=1e-6, color='red', linestyle='--', alpha=0.5, label="Threshold de sinal efetivo")
    ax.legend()

    plt.tight_layout()
    return save_figure(fig, save_path)


def plot_occlusion_survival(
    occlusion_durations_gt: Sequence[float],
    occlusion_durations_survived: Sequence[float],
    save_path: str | Path,
    title: str = "Horizonte de Memória Empírico: Sobrevivência a Oclusões",
) -> Path:
    """Compara distribuição de duração de oclusão do dataset com o que o modelo sobreviveu.

    Parâmetros
    ----------
    occlusion_durations_gt : Sequence[float]
        Durações de oclusão no ground truth (todas as oclusões do dataset).
    occlusion_durations_survived : Sequence[float]
        Durações de oclusão que o modelo conseguiu rastrear sem trocar ID ou matar track.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histograma comparativo
    bins = np.linspace(0, max(max(occlusion_durations_gt, default=10),
                               max(occlusion_durations_survived, default=10)) + 1, 30)

    axes[0].hist(occlusion_durations_gt, bins=bins, alpha=0.5, label='Oclusões no dataset',
                 color='gray', edgecolor='black')
    axes[0].hist(occlusion_durations_survived, bins=bins, alpha=0.7, label='Oclusões sobrevividas pelo modelo',
                 color='steelblue', edgecolor='black')
    axes[0].set_xlabel("Duração da oclusão (quadros)")
    axes[0].set_ylabel("Frequência")
    axes[0].set_title("Distribuição de duração de oclusão")
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # CDF
    sorted_gt = np.sort(occlusion_durations_gt)
    sorted_surv = np.sort(occlusion_durations_survived)
    cdf_gt = np.arange(1, len(sorted_gt) + 1) / len(sorted_gt) if len(sorted_gt) else []
    cdf_surv = np.arange(1, len(sorted_surv) + 1) / len(sorted_surv) if len(sorted_surv) else []

    axes[1].plot(sorted_gt, cdf_gt, 'o-', label='Dataset', color='gray', alpha=0.7)
    axes[1].plot(sorted_surv, cdf_surv, 'o-', label='Modelo (sobreviveu)', color='steelblue')
    axes[1].set_xlabel("Duração da oclusão (quadros)")
    axes[1].set_ylabel("CDF")
    axes[1].set_title("Função de distribuição acumulada")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    return save_figure(fig, save_path)
