"""Entregável `metrics.py` — IDF1, ID switches e fragmentações (implementação própria).

O código vive em ``pa2/metrics/tracking.py``; este arquivo apenas o expõe na raiz
do repositório, como pede o enunciado.
"""

from pa2.metrics.tracking import (  # noqa: F401
    compute_idf1,
    compute_iou,
    count_fragmentations,
    count_id_switches,
    evaluate_tracking_sequence,
    iou_matrix,
    match_global,
)
