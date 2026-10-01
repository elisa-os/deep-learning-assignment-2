"""Testes das métricas de tracking com os casos feitos à mão (valores derivados em cases.py)."""

import pytest

from pa2.metrics.cases import CASES, EXPECTED, make_gt
from pa2.metrics.tracking import evaluate_tracking_sequence


@pytest.mark.parametrize("key", ["a", "b", "c"])
def test_handmade_cases(key):
    pred, gt = CASES[key]()
    r = evaluate_tracking_sequence(pred, gt, iou_threshold=0.5)
    exp = EXPECTED[key]
    assert r["idf1"] == pytest.approx(exp["idf1"])
    assert r["id_switches"] == exp["id_switches"]
    assert r["fragmentations"] == exp["fragmentations"]


def test_swap_and_split_have_different_idf1():
    rb = evaluate_tracking_sequence(*CASES["b"](), iou_threshold=0.5)
    rc = evaluate_tracking_sequence(*CASES["c"](), iou_threshold=0.5)
    assert abs(rb["idf1"] - rc["idf1"]) > 0.1


def test_gt_gap_then_new_id_counts_switch_and_fragmentation():
    gt = make_gt()
    pred = []
    for d in gt:
        d = dict(d)
        if d["id"] == 3:
            if 10 <= d["frame"] <= 14:
                continue                       # perde o objeto por 5 quadros
            if d["frame"] >= 15:
                d["id"] = 7                    # e ele volta com outro id
        pred.append(d)
    r = evaluate_tracking_sequence(pred, gt, iou_threshold=0.5)
    assert r["id_switches"] == 1 and r["fragmentations"] == 1


def test_all_predictions_wrong_gives_zero_idf1():
    gt = make_gt()
    pred = [dict(d, bb_left=d["bb_left"] + 500, id=d["id"] + 50) for d in gt]
    r = evaluate_tracking_sequence(pred, gt, iou_threshold=0.5)
    assert r["idf1"] == 0.0 and r["IDTP"] == 0


def test_gt_with_zero_conf_is_ignored():
    gt = make_gt()
    pred = [dict(d) for d in gt if d["id"] != 3]       # não detecta o objeto 3
    for d in gt:
        if d["id"] == 3:
            d["conf"] = 0.0                            # ...que está marcado como "ignorar"
    r = evaluate_tracking_sequence(pred, gt, iou_threshold=0.5)
    assert r["idf1"] == pytest.approx(1.0) and r["n_gt_ids"] == 2


def test_extra_false_positive_track_lowers_idf1_and_inflates_ids():
    gt = make_gt()
    pred = [dict(d) for d in gt]
    pred += [{"frame": f, "id": 99, "bb_left": 60, "bb_top": 80, "bb_width": 10, "bb_height": 10,
              "conf": 1.0} for f in range(1, 31)]
    r = evaluate_tracking_sequence(pred, gt, iou_threshold=0.5)
    assert r["IDFP"] == 30 and r["idf1"] == pytest.approx(180 / 210)
    assert r["count_error_signed"] == 1
