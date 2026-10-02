"""Associação e gestão de tracks para o PA2.

- ``matching.GreedyMatcher``: baseline da Parte 0 (vídeos sintéticos).
- ``tracker.IoUTracker``: baseline ingênuo das Partes 1+ (caixa observada, IoU).
- ``motion.*`` e ``motion_tracker.MotionTracker``: modelos de movimento (caixa parada, velocidade
  constante, RNN) sobre a mesma gestão de tracks.
"""

from .matching import GreedyMatcher

__all__ = ["GreedyMatcher"]
