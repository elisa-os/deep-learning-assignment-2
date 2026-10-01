"""Modelos de memória temporal para o PA2 — Trilha A (RNN de movimento)."""

from .motion_rnn import (
    MotionRNN,
    TrainSettings,
    eval_shift,
    eval_windows,
    params_finite,
    tf_ratio_at,
    load_checkpoint,
    save_checkpoint,
    train_motion_rnn,
)

__all__ = ["MotionRNN", "TrainSettings", "eval_shift", "eval_windows", "load_checkpoint",
           "params_finite", "save_checkpoint", "tf_ratio_at", "train_motion_rnn"]
