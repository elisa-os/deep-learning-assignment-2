"""Dados sintéticos para o PA2 — gerador de vídeos e simulador de detector."""

from .synthetic import (
    SyntheticSequence,
    SyntheticVideoDataset,
    SimulatedDetector,
    generate_synthetic_sequence,
    make_synthetic_loader,
)

__all__ = [
    "SyntheticSequence",
    "SyntheticVideoDataset",
    "SimulatedDetector",
    "generate_synthetic_sequence",
    "make_synthetic_loader",
]
