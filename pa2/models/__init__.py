"""Modelos de memória temporal para o PA2 — Trilha A (RNN de movimento)."""

from .motion_rnn import (
    MotionRNN,
    TrainSettings,
    load_checkpoint,
    save_checkpoint,
    train_motion_rnn,
)

__all__ = ["MotionRNN", "TrainSettings", "load_checkpoint", "save_checkpoint", "train_motion_rnn"]
