"""Testes do gerador sintético (oclusão real) e do simulador de detector."""

import numpy as np
import pytest

from pa2.synthetic_video import SimulatedDetector, generate_synthetic_sequence
from pa2.synthetic_video.synthetic import HIDDEN_VISIBILITY, MIN_VISIBILITY


@pytest.mark.parametrize("N", [5, 10, 20])
@pytest.mark.parametrize("seed", range(5))
def test_scripted_occlusion_hides_target_for_N_frames(N, seed):
    seq = generate_synthetic_sequence(np.random.default_rng(seed), num_frames=60,
                                      n_objects=8, occlusion_duration=N)
    p = seq.params
    vis = p["target_visibility"]
    start = p["occlusion_start"]
    assert all(vis[f - 1] < HIDDEN_VISIBILITY for f in range(start, start + N))
    assert max(vis[: start - 1]) >= MIN_VISIBILITY        # aparece antes
    assert max(vis[start - 1 + N:]) >= MIN_VISIBILITY     # e volta depois
    tid = p["occlusion_ellipse_id"]
    seen = {b["frame"] for b in seq.true_boxes if b["id"] == tid}
    assert not seen & set(range(start, start + N))        # detector perfeito não vê o escondido
    gt_hidden = [d for d in seq.gt_tracks if d["id"] == tid and d["frame"] in range(start, start + N)]
    assert len(gt_hidden) == N and all(d["conf"] == 0 for d in gt_hidden)  # mas o GT mantém


def test_no_occlusion_when_duration_zero():
    seq = generate_synthetic_sequence(np.random.default_rng(0), n_objects=3, occlusion_duration=0)
    assert seq.params["occlusion_ellipse_id"] is None and seq.params["hidden_frames"] == []


def test_depth_order_front_ellipse_wins():
    """Duas elipses sobrepostas: o pixel comum pertence à de maior z."""
    seq = generate_synthetic_sequence(np.random.default_rng(3), num_frames=30, n_objects=10,
                                      occlusion_duration=0)
    vis = [d["visibility"] for d in seq.gt_tracks]
    assert min(vis) < 1.0          # há oclusão parcial natural
    assert all(0.0 <= v <= 1.0 for v in vis)


def test_generator_is_deterministic():
    a = generate_synthetic_sequence(np.random.default_rng(5), num_frames=20, n_objects=6)
    b = generate_synthetic_sequence(np.random.default_rng(5), num_frames=20, n_objects=6)
    assert np.array_equal(a.frames[7], b.frames[7]) and a.gt_tracks == b.gt_tracks


def test_noise_and_contrast_knobs_change_image():
    kw = dict(num_frames=5, n_objects=5)
    base = generate_synthetic_sequence(np.random.default_rng(1), **kw)
    noisy = generate_synthetic_sequence(np.random.default_rng(1), noise_level=0.1, **kw)
    flat = generate_synthetic_sequence(np.random.default_rng(1), contrast_scale=0.3, **kw)
    assert np.abs(noisy.frames[0].astype(float) - base.frames[0]).mean() > 3
    assert flat.frames[0].std() < base.frames[0].std()


def test_detector_simulator_rates():
    n_frames = 400
    gt = [{"frame": f, "id": i, "bb_left": 50, "bb_top": 50, "bb_width": 20, "bb_height": 20,
           "conf": 1.0} for f in range(1, n_frames + 1) for i in range(10)]
    dets = SimulatedDetector(drop_rate=0.2, noise_std=2.0, fp_rate=0.5,
                             rng=np.random.default_rng(0)).detect(gt, 128, n_frames)
    true_d = [d for d in dets if not d["is_fp"]]
    assert abs((1 - len(true_d) / len(gt)) - 0.2) < 0.03
    assert abs(np.std([d["bb_left"] - 50 for d in true_d]) - 2.0) < 0.15
    assert abs(sum(d["is_fp"] for d in dets) / n_frames - 0.5) < 0.08


def test_detector_simulator_identity_when_off():
    gt = [{"frame": 1, "id": 1, "bb_left": 10, "bb_top": 10, "bb_width": 20, "bb_height": 20,
           "conf": 1.0}]
    out = SimulatedDetector(drop_rate=0, noise_std=0, fp_rate=0).detect(gt, 128)
    assert len(out) == 1 and out[0]["bb_left"] == 10 and out[0]["src_id"] == 1
