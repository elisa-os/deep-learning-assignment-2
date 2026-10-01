"""Rastreador ingênuo por IoU: nascimento, morte, associação e limiar de confiança."""

import numpy as np

from pa2.association.tracker import IoUTracker, track_sequence


def _box(x, y=10, w=20, h=40):
    return [x, y, w, h]


def _run(tracker, frames):
    """frames: lista (por quadro) de listas de caixas. Devolve ids por quadro."""
    return [sorted(tid for tid, _ in tracker.update(np.array(f, float).reshape(-1, 4)))
            for f in frames]


def test_static_objects_keep_their_ids():
    t = IoUTracker(iou_threshold=0.3, max_age=5)
    ids = _run(t, [[_box(10), _box(100)]] * 5)
    assert all(i == [1, 2] for i in ids)


def test_moving_object_is_followed_when_iou_stays_above_threshold():
    t = IoUTracker(iou_threshold=0.3)
    ids = _run(t, [[_box(10 + 3 * k)] for k in range(10)])
    assert all(i == [1] for i in ids)


def test_gap_longer_than_max_age_creates_new_id():
    t = IoUTracker(max_age=2)
    ids = _run(t, [[_box(10)], [], [], [], [_box(10)]])
    assert ids[0] == [1] and ids[-1] == [2]


def test_gap_within_max_age_keeps_id():
    t = IoUTracker(max_age=3)
    ids = _run(t, [[_box(10)], [], [], [_box(10)]])
    assert ids[-1] == [1]


def test_tracker_does_not_extrapolate_motion_during_gap():
    # objeto anda 40 px/quadro; reaparece depois de 2 quadros longe da última caixa -> id novo
    t = IoUTracker(max_age=5, iou_threshold=0.3)
    ids = _run(t, [[_box(0)], [], [], [_box(120)]])
    assert ids[-1] == [2]


def test_min_hits_delays_birth_and_drops_one_off_detections():
    t = IoUTracker(min_hits=3)
    ids = _run(t, [[_box(10)], [_box(10)], [_box(10)], [_box(300)], [_box(10)]])
    assert ids[0] == [] and ids[1] == [] and ids[2] == [1]
    assert ids[3] == []          # ruído isolado não ganha id
    assert ids[4] == [1]


def test_min_conf_filters_detections():
    t = IoUTracker(min_conf=0.5)
    out = t.update(np.array([_box(10), _box(100)], float), np.array([0.9, 0.2]))
    assert out == [(1, 0)]


def test_ids_are_never_reused():
    t = IoUTracker(max_age=0)
    ids = _run(t, [[_box(10)], [], [_box(10)], [], [_box(10)]])
    assert ids == [[1], [], [2], [], [3]]


def test_greedy_and_hungarian_differ_on_a_conflict():
    # IoU (tracks x detecções): guloso pega o maior valor (0.6) e deixa a track 2 sem par;
    # Hungarian maximiza a soma (0.5 + 0.55 > 0.6) e casa as duas.
    iou = np.array([[0.6, 0.5],
                    [0.55, 0.0]])
    greedy = IoUTracker(iou_threshold=0.3, method="greedy")._assign(iou)
    hungarian = IoUTracker(iou_threshold=0.3, method="hungarian")._assign(iou)
    assert greedy == [(0, 0)]
    assert sorted(hungarian) == [(0, 1), (1, 0)]


def test_track_sequence_output_format():
    dets = np.array([[1, 10, 10, 20, 40, 0.9], [2, 11, 10, 20, 40, 0.8]], float)
    tr = track_sequence(dets, 3, iou_threshold=0.3, max_age=5, min_hits=1)
    assert [t["id"] for t in tr] == [1, 1] and tr[0]["frame"] == 1 and tr[1]["frame"] == 2
    assert set(tr[0]) == {"frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"}
