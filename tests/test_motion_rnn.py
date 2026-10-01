"""Parte 2: rastreador com modelo de movimento, trajetórias do GT e a RNN de movimento."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.association.motion_tracker import MotionTracker, track_sequence_motion
from pa2.association.tracker import IoUTracker, track_sequence
from pa2.models import MotionRNN, TrainSettings, load_checkpoint, save_checkpoint, train_motion_rnn
from pa2.models.motion_rnn import box_error, decode
from pa2.mot17 import Sequence
from pa2.mot17.trajectories import (Segment, WindowSampler, cxcywh_to_ltwh, gt_segments,
                                    ltwh_to_cxcywh)

ROOT = Path("data/MOT17")
needs_data = pytest.mark.skipif(not (ROOT / "train/MOT17-09-SDP/gt/gt.txt").exists(),
                                reason="MOT17 (anotações) não encontrado em data/MOT17")


def _run(tracker, frames):
    return [sorted(tid for tid, _ in tracker.update(np.array(f, float).reshape(-1, 4)))
            for f in frames]


def _fast_object_with_gap(gap=4, speed=8, n=14):
    """Um objeto 20x40 andando `speed` px/quadro, sumindo por `gap` quadros no meio."""
    frames = []
    for k in range(n):
        frames.append([] if 4 <= k < 4 + gap else [[10 + speed * k, 50, 20, 40]])
    return frames


# ── rastreador com movimento ────────────────────────────────────────────────
def test_static_motion_tracker_is_the_part1_tracker():
    rng = np.random.default_rng(0)
    frames = []
    for k in range(30):
        f = np.array([[10 + 3 * k, 20, 30, 60], [200, 100 - k, 40, 80]], float) + rng.normal(0, 1, (2, 4)) * [2, 2, 1, 1]
        if k % 2:
            f = np.vstack([f, [rng.uniform(0, 300), 0, 30, 50]])
        frames.append(f)
    a, b = IoUTracker(max_age=3, min_hits=2), MotionTracker(StaticMotion(), max_age=3, min_hits=2)
    for f in frames:
        assert a.update(f) == b.update(f)


@needs_data
def test_static_motion_matches_part1_on_real_sequence():
    s = Sequence(ROOT, "09", "SDP")
    d = s.det[s.det[:, 5] >= 0.4]
    kw = dict(iou_threshold=0.3, max_age=30, min_hits=3, method="greedy")
    assert track_sequence(d, s.info.seq_length, **kw) == \
        track_sequence_motion(d, s.info.seq_length, StaticMotion(), **kw)


def test_constant_velocity_survives_a_gap_that_breaks_the_static_tracker():
    frames = _fast_object_with_gap(gap=4, speed=8)
    static = _run(MotionTracker(StaticMotion(), iou_threshold=0.3, max_age=10, min_hits=1), frames)
    cv = _run(MotionTracker(ConstantVelocityMotion(), iou_threshold=0.3, max_age=10, min_hits=1), frames)
    assert static[0] == [1] and static[-1] != [1]       # a caixa parada não alcança o objeto de volta
    assert all(i in ([], [1]) for i in cv) and cv[-1] == [1]


def test_unobserved_track_dies_after_max_age_even_with_motion_model():
    t = MotionTracker(ConstantVelocityMotion(), max_age=2, min_hits=1)
    ids = _run(t, [[[10, 10, 20, 40]], [], [], [], [[10, 10, 20, 40]]])
    assert ids[0] == [1] and ids[-1] == [2]


# ── trajetórias do GT ───────────────────────────────────────────────────────
def test_box_conversions_roundtrip():
    b = np.array([[10., 20., 30., 40.]])
    assert np.allclose(cxcywh_to_ltwh(ltwh_to_cxcywh(b)), b)


@needs_data
def test_gt_segments_are_contiguous_and_class_filtered():
    s = Sequence(ROOT, "09", "SDP")
    segs = gt_segments(s)
    assert segs and all((np.diff(g.frames) == 1).all() for g in segs)
    assert {g.track_id for g in segs} <= set(np.unique(s.gt_pedestrians()[:, 1]).astype(int))


def test_window_sampler_shapes_and_padding():
    boxes = np.tile([100., 100., 20., 40.], (10, 1)) + np.arange(10)[:, None] * [3, 0, 0, 0]
    seg = Segment("x", 1, np.arange(10), boxes, np.ones(10), (640., 480.))
    smp = WindowSampler([seg], L=16, strides=(1, 2), seed=0)
    b, valid, size, dt = smp.sample(8)
    assert b.shape == (8, 17, 4) and valid.shape == (8, 17) and dt.shape == (8, 16)
    assert valid[:, :3].all() and not valid.all()        # janela maior que o segmento: tem padding
    assert (size == [640, 480]).all()


# ── modelo ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["RNN", "GRU", "LSTM"])
def test_untrained_model_predicts_staying_put(kind):
    m = MotionRNN(kind, 16)
    box = torch.tensor([[100., 100., 20., 40.]])
    pred, state, _ = m.step(box, None, torch.ones(1), torch.ones(1), torch.tensor([[640., 480.]]),
                            m.init_state(1))
    assert torch.allclose(pred, box) and state.shape == (1, m.n_state * 16)


def test_decode_and_box_error_are_inverse_in_box_units():
    box = torch.tensor([[100., 100., 20., 40.]])
    nxt = decode(box, torch.tensor([[5., -5., 1., 0.]]))   # 50% da largura... (S = 10)
    err = box_error(nxt, box)
    assert torch.allclose(err[0, 0], torch.tensor(0.5 * 20 / 20)) and err[0, 1] < 0


def test_rollout_gradients_flow_through_gaps_and_own_predictions():
    m = MotionRNN("GRU", 16)
    obs = torch.rand(4, 9, 4) * 10 + torch.tensor([100., 100., 20., 40.])
    can = torch.ones(4, 9, dtype=torch.bool)
    can[:, 3:6] = False
    for p in m.parameters():
        p.grad = None
    preds, flags = m.rollout(obs, can, torch.tensor([[640., 480.]]).repeat(4, 1), torch.ones(4, 8),
                             tf_ratio=0.5, generator=torch.Generator().manual_seed(0))
    assert preds.shape == (4, 8, 4) and (flags[:, 3:6] == 0).all() and (flags[:, 0] == 1).all()
    preds.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_model_learns_constant_velocity_and_beats_standing_still():
    """Trajetórias lineares sintéticas: a RNN deve aprender a velocidade (sem ela o erro é ~v)."""
    rng = np.random.default_rng(0)
    segs = []
    for i in range(40):
        v = rng.uniform(-6, 6, 2)
        c = np.array([300., 200.]) + np.arange(60)[:, None] * v
        boxes = np.concatenate([c, np.tile([30., 80.], (60, 1))], 1)
        segs.append(Segment("syn", i, np.arange(60), boxes, np.ones(60), (640., 480.)))
    # a validação agora tem protocolo próprio (val_*); aqui a validação também é sem buracos
    cfg = TrainSettings(epochs=15, steps_per_epoch=40, batch_size=32, window_T=16, gap_prob=0.0,
                        val_gap_prob=0.0, obs_noise=(0.0, 0.0, 0.0, 0.0), lr=3e-3)
    torch.manual_seed(0)
    m = MotionRNN("GRU", 32)
    val = WindowSampler(segs[:10], 16).fixed_windows()
    hist = train_motion_rnn(m, WindowSampler(segs, 16, seed=0), val, cfg, log=lambda *_: None)
    assert hist["val_loss"][-1] < 0.5 * hist["val_loss"][0]
    assert hist["val_iou"][-1] > 0.9


def test_checkpoint_roundtrip(tmp_path):
    m = MotionRNN("LSTM", 24, use_delta_t=False)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.1)
    save_checkpoint(m, tmp_path / "m.pt", {"epoch": 3})
    m2, meta = load_checkpoint(tmp_path / "m.pt")
    assert m2.config == m.config and meta["epoch"] == 3
    x = (torch.tensor([[50., 50., 20., 40.]]), None, torch.ones(1), torch.ones(1),
         torch.tensor([[640., 480.]]), m.init_state(1))
    assert torch.allclose(m.step(*x)[0], m2.step(*x)[0])


def test_rnn_motion_adapter_runs_in_tracker():
    m = MotionRNN("GRU", 16)
    t = MotionTracker(RNNMotion(m, (640, 480)), max_age=3, min_hits=1)
    ids = _run(t, [[[10 + 2 * k, 50, 20, 40], [300, 200 - k, 30, 60]] for k in range(6)])
    assert all(i == [1, 2] for i in ids)
