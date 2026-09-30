"""Associação e gestão de tracks para o PA2 — baseline por quadro e modelo temporal."""

from .matching import (
    GreedyMatcher,
    HungarianMatcher,
    TrackManager,
    track_matches_iou,
    simple_greedy_match,
)

__all__ = [
    "BoxAssociation",
    "GreedyMatcher",
    "HungarianMatcher",
    "TrackManager",
    "track_matches_iou",
]