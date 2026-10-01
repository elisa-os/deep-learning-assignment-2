"""Parte 4: horizonte de memória (gradiente e oclusões injetadas), galeria e amortecimento."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pa2.analysis.gallery import pick_failures
from pa2.analysis.memory import (gradient_horizon, horizon_summary, sample_trials, survival_experiment,
                                 survival_table, wilson)
from pa2.analysis.runner import match_gt_to_dets, run_trace, sorted_dets
from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.models import MotionRNN
from pa2.mot17.trajectories import Segment
from pa2.part4 import _n50


class _Seq:
    """Dois pedestres 20x40 andando 8 px/quadro, 90 quadros; detecções = GT."""
    video = "01"

    def __init__(self, T=90):
        self.info = SimpleNamespace(seq_length=T, im_width=1920, im_height=1080)
        rows = []
        for f in range(1, T + 1):
            rows.append([f, 1, 100 + 8 * f, 100, 20, 40, 1.0])
            rows.append([f, 2, 100 + 8 * f, 600, 20, 40, 1.0])
        self._gt = np.array(rows, float)

    def gt_pedestrians(self):
        return self._gt

    def dets(self):
        d = self._gt[:, [0, 2, 3, 4, 5]]
        return np.c_[d, np.ones(len(d))]


def _model_with_motion():
    m = MotionRNN("GRU", 16)
    torch.manual_seed(0)
    torch.nn.init.normal_(m.head[-1].weight, std=0.5)
    return m


def test_wilson_interval_contains_proportion_and_shrinks_with_n():
    lo, hi = wilson(30, 60)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson(300, 600)
    assert hi2 - lo2 < hi - lo
    assert wilson(0, 0)[0] != wilson(0, 0)[0]          # nan


def test_gradient_horizon_shapes_and_signs():
    m = _model_with_motion()          # a cabeça nasce zerada: sem isso o gradiente no estado é 0
    boxes = np.tile([300., 300., 30., 80.], (60, 1)) + np.arange(60)[:, None] * [3, 0, 0, 0]
    seg = [Segment("x", 1, np.arange(60), boxes, np.ones(60), (1920., 1080.))]
    for mode in ("observed", "blind"):
        n = gradient_horizon(m, seg, (0.02,) * 4, L=24, context=4, n=20, mode=mode)
        assert n.shape == (20, 24) and np.isfinite(n).all() and (n[:, 0] > 0).all()
    s = horizon_summary(n)
    assert len(s["relative"]) == 24 and abs(s["relative"][0] - 1.0) < 1e-9


def test_perfect_detections_match_gt_one_to_one():
    seq = _Seq()
    D = sorted_dets(seq.dets(), 0.0)
    match = match_gt_to_dets(seq, D)
    assert len(match) == len(D) and len(set(match.values())) == len(D)


def test_sample_trials_respect_margins_and_do_not_overlap():
    seq = _Seq()
    D = sorted_dets(seq.dets(), 0.0)
    match = match_gt_to_dets(seq, D)
    t = sample_trials(match, seq, 10, np.random.default_rng(0), per_run=3, ids_pool=[1, 2])
    assert 1 <= len(t) <= 2
    for gid, s in t:
        assert all((gid, f) in match for f in range(s - 5, s)) and (gid, s + 10) in match
    if len(t) == 2:
        (_, a), (_, b) = sorted(t, key=lambda x: x[1])
        assert b - a > 10 + 5


def test_injected_occlusion_separates_motion_models():
    """Objetos andando 8 px/quadro (40% da largura): com uma oclusão de 6 quadros a caixa parada
    não alcança o objeto de volta, a de velocidade constante sim."""
    seq = _Seq()
    D = sorted_dets(seq.dets(), 0.0)
    motions = {"static": lambda s: StaticMotion(), "const_vel": lambda s: ConstantVelocityMotion()}
    assoc = dict(method="greedy", iou_threshold=0.3, max_age=30, min_hits=1)
    tr = survival_experiment(seq, D, motions, assoc, (0, 6), reps=3, per_run=1)
    tab = survival_table(tr).set_index(["method", "N"])
    assert tab.loc[("static", 0), "kept"] == 1.0 and tab.loc[("const_vel", 0), "kept"] == 1.0
    assert tab.loc[("const_vel", 6), "kept"] == 1.0
    assert tab.loc[("static", 6), "kept"] == 0.0
    assert set(tr.outcome) <= {"kept", "swapped", "reborn"}


def test_n50_interpolates_between_points():
    tab = pd.DataFrame({"method": "a", "N": [0, 10, 20], "kept": [1.0, 0.8, 0.2]})
    assert abs(_n50(tab, "a") - 15.0) < 1e-9


def test_blind_damping_only_affects_unobserved_steps():
    m = _model_with_motion()
    size = (1920, 1080)
    fed = np.array([[100., 100., 20., 40.], [300., 200., 30., 80.]])
    prev = [fed[0] - [3, 0, 0, 0], fed[1] - [3, 0, 0, 0]]
    obs = np.array([True, False])
    _, p1 = RNNMotion(m, size).step([None, None], fed, prev, obs)
    _, p0 = RNNMotion(m, size, blind_damping=0.0).step([None, None], fed, prev, obs)
    _, ph = RNNMotion(m, size, blind_damping=0.5).step([None, None], fed, prev, obs)
    assert np.allclose(p0[0], p1[0]) and np.allclose(ph[0], p1[0])       # observada: igual
    assert np.allclose(p0[1], fed[1], atol=1e-4)                                   # amortecimento total: fica parada
    assert np.allclose(ph[1] - fed[1], 0.5 * (p1[1] - fed[1]), atol=1e-4)


def test_run_trace_gates_follow_the_motion_model():
    seq = _Seq(30)
    D = sorted_dets(seq.dets(), 0.0)
    assign, gates = run_trace(D, 30, ConstantVelocityMotion(), method="greedy", iou_threshold=0.3,
                              max_age=5, min_hits=1)
    tid, row = assign[19][0]
    box = D[row, 1:5]
    g = gates[19][tid]
    assert abs(g[0] - box[0]) < 2.0                                      # a previsão chega perto da detecção


def test_pick_failures_rules():
    base = dict(outcome="switched", mean_visibility=0.1, track_alive_at_end=True, gate_iou_at_end=0.1,
                swap_partner=np.nan, video="09", camera="parada", length=20)
    meas = pd.DataFrame([
        dict(base),                                                       # 0: oclusão longa, câmera parada
        dict(base, camera="móvel", video="13", length=4, mean_visibility=0.9, gate_iou_at_end=0.0),  # 1
        dict(base, length=6, swap_partner=5.0, gate_iou_at_end=0.5),     # 2: troca entre pessoas
        dict(base, outcome="kept"),                                       # 3: não é falha
    ])
    p = pick_failures(None, meas)
    assert p == {"oclusao_longa": 0, "buraco_curto_camera_movel": 1, "troca_entre_pessoas": 2}
