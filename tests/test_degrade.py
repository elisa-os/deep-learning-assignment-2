"""Degradação das detecções (Parte 5, qualidade do detector)."""

import numpy as np

from pa2.stress import degrade_detections


def _dets(n_frames=50, per_frame=10):
    rng = np.random.default_rng(0)
    rows = [[f, rng.uniform(0, 900), rng.uniform(0, 500), rng.uniform(20, 80), rng.uniform(50, 200), 0.9]
            for f in range(1, n_frames + 1) for _ in range(per_frame)]
    return np.array(rows)


def _go(d, **kw):
    return degrade_detections(d, n_frames=50, image_size=(1000, 600), score_min=0.4,
                              rng=np.random.default_rng(1), **kw)


def test_identity_when_no_degradation():
    d = _dets()
    assert np.array_equal(_go(d), d)


def test_drop_rate():
    d = _dets()
    assert abs(len(_go(d, drop=0.3)) / len(d) - 0.7) < 0.05


def test_noise_moves_boxes_relative_to_size():
    d = _dets()
    out = _go(d, noise=0.1)
    assert len(out) == len(d)
    rel = np.abs(out[:, 1] - d[:, 1]) / d[:, 3]
    assert 0.03 < rel.mean() < 0.15
    assert np.array_equal(out[:, 0], d[:, 0]) and np.allclose(out[:, 5], d[:, 5])


def test_false_positives_rate_scores_and_bounds():
    d = _dets()
    out = _go(d, fp=2.0)
    extra = len(out) - len(d)
    assert abs(extra / 50 - 2.0) < 0.6
    fp = out[len(out) - 1:]
    assert out[:, 5].min() >= 0.4 and out[:, 0].min() >= 1 and out[:, 0].max() <= 50
    assert np.all(out[:, 1] >= 0) and np.all(np.diff(out[:, 0]) >= 0)
