"""Teste de estresse da Parte 5: queda de taxa de quadros (subamostragem do vídeo)."""

from .degrade import degrade_detections
from .subsample import SubsampledSequence, cut_frames, phases, subsample_dets

__all__ = ["degrade_detections", "SubsampledSequence", "cut_frames", "phases", "subsample_dets"]
