"""Casos construídos à mão para validar as métricas de tracking (Parte 0, item 3).

Três objetos parados e bem separados, 30 quadros (ids GT 1, 2, 3). Como as caixas
não se sobrepõem, os valores esperados podem ser derivados à mão:

(a) predição = GT                      -> IDF1 = 1, 0 switches, 0 fragmentações
(b) ids 1 e 2 trocados do quadro k=16  -> pred 1 casa com GT1 em 15 quadros e com GT2
    em 15; idem pred 2. A melhor atribuição global acerta só 15 quadros de cada
    um: IDTP = 15+15+30 = 60 de Ng = Np = 90  =>  IDF1 = 120/180 = 2/3, e 2 switches.
(c) track 1 partida no quadro 10 (id 100) e sem detecção nos quadros 12-14
                                        -> pred 100 cobre 18 quadros de GT1 (10, 11, 15..30)
    e pred 1 fica sem par: IDTP = 18+30+30 = 78, Np = 87, Ng = 90, IDFN = 12, IDFP = 9
    =>  IDF1 = 156/177 ≈ 0.8814, 1 switch, 1 fragmentação.

(b) e (c) têm IDF1 diferentes: em (b) toda caixa é "detectada" mas com a identidade
errada em metade do tempo; em (c) perde-se identidade mas também caixas (FN).
"""

from __future__ import annotations

from typing import Any

NUM_FRAMES = 30
SWAP_FRAME = 16          # (b): troca a partir deste quadro (inclusive)
SPLIT_FRAME = 10         # (c): track 1 vira id 100 a partir daqui
DROP_FRAMES = (12, 13, 14)
SPLIT_ID = 100

# Valores esperados, derivados à mão (ver docstring do módulo)
EXPECTED = {
    "a": {"idf1": 1.0, "id_switches": 0, "fragmentations": 0},
    "b": {"idf1": 120 / 180, "id_switches": 2, "fragmentations": 0},
    "c": {"idf1": 156 / 177, "id_switches": 1, "fragmentations": 1},
}


def make_gt() -> list[dict[str, Any]]:
    """3 objetos 20x20 parados em posições bem separadas."""
    anchors = {1: (10, 10), 2: (50, 10), 3: (90, 10)}
    return [
        {"frame": f, "id": i, "bb_left": x, "bb_top": y, "bb_width": 20, "bb_height": 20, "conf": 1.0}
        for f in range(1, NUM_FRAMES + 1)
        for i, (x, y) in anchors.items()
    ]


def case_a() -> tuple[list[dict], list[dict]]:
    gt = make_gt()
    return [dict(d) for d in gt], gt


def case_b() -> tuple[list[dict], list[dict]]:
    gt = make_gt()
    pred = []
    for d in gt:
        d = dict(d)
        if d["frame"] >= SWAP_FRAME and d["id"] in (1, 2):
            d["id"] = 3 - d["id"]
        pred.append(d)
    return pred, gt


def case_c() -> tuple[list[dict], list[dict]]:
    gt = make_gt()
    pred = []
    for d in gt:
        d = dict(d)
        if d["id"] == 1:
            if d["frame"] in DROP_FRAMES:
                continue
            if d["frame"] >= SPLIT_FRAME:
                d["id"] = SPLIT_ID
        pred.append(d)
    return pred, gt


CASES = {"a": case_a, "b": case_b, "c": case_c}
