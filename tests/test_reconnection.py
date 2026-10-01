"""Análise de reconexão depois de buracos de rastreamento."""

from pathlib import Path

import numpy as np
import pytest

from pa2.association.motion import StaticMotion
from pa2.association.motion_tracker import track_sequence_motion
from pa2.metrics.reconnection import bucket_of, gap_episodes
from pa2.mot17 import Sequence
from pa2.mot17.evaluate import evaluate_tracks

ROOT = Path("data/MOT17")


class _FakeSeq:
    """GT de uma identidade em 12 quadros, parada no lugar."""
    def __init__(self):
        self._gt = np.array([[f, 1, 100, 100, 20, 40, 0.3] for f in range(1, 13)], float)

    def gt_pedestrians(self):
        return self._gt

    def gt_distractors(self):
        return np.zeros((0, 6))


def _pred(frames_ids):
    return [{"frame": f, "id": i, "bb_left": 100., "bb_top": 100., "bb_width": 20., "bb_height": 40.,
             "conf": 1.0} for f, i in frames_ids]


def test_buckets():
    assert [bucket_of(k) for k in (1, 5, 6, 15, 16, 30, 31, 200)] == \
        ["1-5", "1-5", "6-15", "6-15", "16-30", "16-30", "31+", "31+"]


def test_kept_switched_and_lost_outcomes():
    # id 7 nos quadros 1-3, buraco de 2 (4-5), volta com 7 (6-7), buraco de 3 (8-10), volta com 9 (11),
    # e o quadro 12 fica sem casamento até o fim da vida -> lost
    tr = _pred([(f, 7) for f in (1, 2, 3, 6, 7)] + [(11, 9)])
    e = gap_episodes(_FakeSeq(), tr)
    assert list(e.outcome) == ["kept", "switched", "lost"]
    assert list(e.length) == [2, 3, 1]
    assert np.allclose(e.mean_visibility, 0.3)


def test_perfect_tracking_has_no_gaps():
    assert len(gap_episodes(_FakeSeq(), _pred([(f, 1) for f in range(1, 13)]))) == 0


@pytest.mark.skipif(not (ROOT / "train/MOT17-09-SDP/gt/gt.txt").exists(), reason="sem dados")
def test_kept_plus_switched_equals_fragmentations_on_real_data():
    s = Sequence(ROOT, "09", "SDP")
    d = s.det[s.det[:, 5] >= 0.4]
    tr = track_sequence_motion(d, s.info.seq_length, StaticMotion(), iou_threshold=0.3, max_age=30,
                               min_hits=3, method="greedy")
    e = gap_episodes(s, tr)
    assert (e.outcome != "lost").sum() == evaluate_tracks(s, tr)["fragmentations"]
