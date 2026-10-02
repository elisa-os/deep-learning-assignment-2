"""Visualização e exportação de gráficos de avaliação e trajetórias (PA2)."""

from __future__ import annotations

from pathlib import Path

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
