"""Métricas de tracking para o PA2 — implementação própria (não usa bibliotecas prontas)."""

from .tracking import compute_idf1, count_id_switches, count_fragmentations, match_global

__all__ = [
    "compute_idf1",
    "count_id_switches",
    "count_fragmentations",
    "match_global",
]
