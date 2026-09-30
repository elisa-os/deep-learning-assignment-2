"""Modelos de memória temporal para o PA2 — Trilha A (RNN de movimento)."""

from .motion_rnn import (
    MotionRNN,
    MotionRNNPredictor,
    train_motion_rnn,
)

__all__ = [
    "MotionRNN",
    "MotionRNNPredictor",
    "train_motion_rnn",
]
