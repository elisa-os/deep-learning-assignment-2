"""Parte 5: subamostragem de vídeo, diagnósticos de Δt e resumo da curva de degradação."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pa2.association.motion import ConstantVelocityMotion, RNNMotion, StaticMotion
from pa2.config import load_config
from pa2.models import MotionRNN
from pa2.mot17.loader import SeqInfo, Sequence
from pa2.mot17.trajectories import Segment
from pa2.part5 import expand, summarize
from pa2.stress import SubsampledSequence, cut_frames, phases, subsample_dets
from pa2.stress.diagnostics import consecutive_iou, dt_sensitivity, onestep_displacement_regression

ROOT = Path("data/MOT17")
needs_data = pytest.mark.skipif(not (ROOT / "train/MOT17-09-SDP/gt/gt.txt").exists(),
                                reason="MOT17 (anotações) não encontrado em data/MOT17")


class _FakeSeq:
    """30 quadros, 2 pedestres e 1 distrator andando; GT e detecções no formato do loader."""
    video, name = "99", "MOT17-99"
    info = SeqInfo("MOT17-99-SDP", "img1", 30, 30, 1920, 1080, ".jpg")

    def __init__(self):
        self.ped = np.array([[f, i, 100 + 5 * f, 100 * i, 20, 40, 0.8] for f in range(1, 31) for i in (1, 2)], float)
        self.dis = np.array([[f, 9, 500, 500, 20, 40] for f in range(1, 31)], float)

    def gt_pedestrians(self):
        return self.ped

    def gt_distractors(self):
        return self.dis


# ── subamostragem ───────────────────────────────────────────────────────────
def test_phases_and_validation():
    assert list(phases(5)) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        cut_frames(np.zeros((1, 6)), 2, 2)
    with pytest.raises(ValueError):
        cut_frames(np.zeros((1, 6)), 0, 0)


@pytest.mark.parametrize("k,phase,expected", [(1, 0, 30), (2, 0, 15), (2, 1, 15), (5, 0, 6), (5, 4, 6), (4, 3, 7)])
def test_cut_frames_keeps_the_right_frames_and_renumbers(k, phase, expected):
    a = np.array([[f, 1, 0, 0, 1, 1, 1] for f in range(1, 31)], float)
    out = cut_frames(a, k, phase)
    assert len(out) == expected
    assert out[:, 0].tolist() == list(range(1, expected + 1))
    # o i-ésimo quadro mantido é o original 1 + phase + i*k: confere pelo id codificado em outra coluna
    b = a.copy()
    b[:, 1] = np.arange(1, 31)
    kept = cut_frames(b, k, phase)[:, 1].astype(int)
    assert kept.tolist() == [1 + phase + i * k for i in range(expected)]


def test_cut_frames_does_not_mutate_input_and_handles_empty():
    a = np.array([[3, 1, 0, 0, 1, 1, 1]], float)
    cut_frames(a, 2, 0)
    assert a[0, 0] == 3
    assert len(cut_frames(np.zeros((0, 6)), 2, 0)) == 0


def test_subsampled_sequence_is_consistent_with_detections():
    seq = _FakeSeq()
    sub = SubsampledSequence(seq, 5, 2)
    assert sub.info.seq_length == 6 and sub.info.frame_rate == 6.0
    assert sub.video == "99" and sub.name == "MOT17-99"
    ped, dis = sub.gt_pedestrians(), sub.gt_distractors()
    assert sorted(set(ped[:, 0])) == list(range(1, 7)) and sorted(set(dis[:, 0])) == list(range(1, 7))
    dets = np.array([[f, 100 + 5 * f, 100, 20, 40, 0.9] for f in range(1, 31)], float)
    d = subsample_dets(dets, 5, 2)
    assert sorted(set(d[:, 0])) == list(range(1, 7))
    # mesma caixa no mesmo quadro novo: o quadro original 3 vira o 1; o GT do obj. 1 e a detecção coincidem em x
    assert d[0, 1] == ped[ped[:, 1] == 1][0, 2]


def test_k1_is_the_identity():
    seq = _FakeSeq()
    sub = SubsampledSequence(seq, 1, 0)
    assert np.array_equal(sub.gt_pedestrians(), seq.ped) and sub.info.seq_length == 30


@needs_data
def test_k1_reproduces_the_original_evaluation_on_real_data():
    from pa2.association.motion_tracker import track_sequence_motion
    from pa2.mot17.evaluate import evaluate_tracks
    s = Sequence(ROOT, "09", "SDP")
    d = s.det[s.det[:, 5] >= 0.4]
    kw = dict(iou_threshold=0.3, max_age=30, min_hits=3, method="greedy")
    t0 = track_sequence_motion(d, s.info.seq_length, StaticMotion(), **kw)
    t1 = track_sequence_motion(subsample_dets(d, 1, 0), s.info.seq_length, StaticMotion(), **kw)
    assert t0 == t1
    a = evaluate_tracks(s, t0)
    b = evaluate_tracks(SubsampledSequence(s, 1, 0), t1)
    assert a["idf1"] == b["idf1"] and a["id_switches"] == b["id_switches"]


@needs_data
def test_subsampled_real_sequence_has_fewer_frames_and_same_ids():
    s = Sequence(ROOT, "09", "SDP")
    sub = SubsampledSequence(s, 5, 0)
    assert sub.info.seq_length == len(range(1, s.info.seq_length + 1, 5))
    assert set(sub.gt_pedestrians()[:, 1]) <= set(s.gt_pedestrians()[:, 1])
    assert sub.gt_pedestrians()[:, 0].max() == sub.info.seq_length


# ── diagnósticos ────────────────────────────────────────────────────────────
def _segments(n=30, length=80, speed=3.0, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        v = rng.uniform(-speed, speed, 2)
        c = np.array([500., 400.]) + np.arange(length)[:, None] * v
        out.append(Segment("syn", i, np.arange(length), np.concatenate([c, np.tile([30., 80.], (length, 1))], 1),
                           np.ones(length), (1920., 1080.)))
    return out


def test_consecutive_iou_falls_with_stride():
    d = consecutive_iou(_segments(speed=6.0), (1, 2, 5)).set_index("stride")
    assert d.loc[1, "iou_medio"] > d.loc[2, "iou_medio"] > d.loc[5, "iou_medio"]
    assert d.loc[1, "frac_iou_abaixo_0.3"] <= d.loc[5, "frac_iou_abaixo_0.3"]


def test_displacement_slope_is_one_for_perfect_extrapolation_and_zero_for_standing_still():
    segs = _segments(speed=6.0)
    zero = (0.0, 0.0, 0.0, 0.0)
    cv = onestep_displacement_regression(lambda size, k: ConstantVelocityMotion(beta=1.0), segs, zero, (1, 2, 5),
                                         n=400)
    assert np.allclose(cv.slope, 1.0, atol=1e-6) and np.allclose(cv["corr"], 1.0, atol=1e-6)
    st = onestep_displacement_regression(lambda size, k: StaticMotion(), segs, zero, (1, 2), n=400)
    assert np.allclose(st.slope, 0.0, atol=1e-9)


def _random_model(use_delta_t=True):
    torch.manual_seed(0)
    m = MotionRNN("GRU", 16, use_delta_t=use_delta_t)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return m


def test_rnn_motion_passes_dt_to_the_model_only_when_the_model_uses_it():
    fed = np.array([[100., 100., 20., 40.]])
    prev = [fed[0] - [3, 0, 0, 0]]
    for use_dt, should_change in ((True, True), (False, False)):
        m = _random_model(use_dt)
        _, a = RNNMotion(m, (1920, 1080), dt=1.0).step([None], fed, prev, np.array([True]))
        _, b = RNNMotion(m, (1920, 1080), dt=5.0).step([None], fed, prev, np.array([True]))
        assert (not np.allclose(a, b)) == should_change


def test_dt_sensitivity_zero_without_dt_feature_and_positive_with_it():
    segs = _segments(speed=6.0)
    sig = (0.02,) * 4
    off = dt_sensitivity(lambda size, dt: RNNMotion(_random_model(False), size, dt=dt), segs, sig, (2,), n=300)
    on = dt_sensitivity(lambda size, dt: RNNMotion(_random_model(True), size, dt=dt), segs, sig, (2,), n=300)
    assert off.rel_change_mediana.iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert on.rel_change_mediana.iloc[0] > 1e-3


# ── resumo da curva ─────────────────────────────────────────────────────────
def _raw():
    rows = []
    for variant in ("fixa", "casada"):
        for method in ("static", "rnn_dt1", "rnn_dtk"):
            for k in (1, 2):
                if variant == "casada" and k == 1:
                    continue
                if method == "rnn_dtk" and k == 1:
                    continue
                for phase in range(k):
                    for video, split in (("09", "val"), ("02", "train")):
                        base = {"static": 0.4, "rnn_dt1": 0.6, "rnn_dtk": 0.6}[method]
                        rows.append({"variant": variant, "method": method, "k": k, "phase": phase, "video": video,
                                     "split": split, "IDF1": base - 0.1 * (k - 1) + 0.01 * phase, "ID_switches": 5,
                                     "fragmentations": 3, "n_gt_ids": 10, "n_pred_ids": 12, "MOTA": 0.5,
                                     "ids_pred/gt": 1.2, "IDsw/id": 0.5})
    return pd.DataFrame(rows)


def test_expand_fills_the_cells_that_are_equal_by_construction():
    e = expand(_raw())
    assert ((e.variant == "casada") & (e.k == 1)).any() and ((e.method == "rnn_dtk") & (e.k == 1)).any()
    a = e[(e.variant == "fixa") & (e.method == "rnn_dt1") & (e.k == 1)].IDF1.to_numpy()
    b = e[(e.variant == "fixa") & (e.method == "rnn_dtk") & (e.k == 1)].IDF1.to_numpy()
    assert np.array_equal(a, b)


def test_summarize_means_phases_and_relative_idf1():
    s = summarize(_raw())
    r = s[(s.variant == "fixa") & (s.method == "rnn_dt1") & (s.subset == "val") & (s.k == 2)].iloc[0]
    assert r.IDF1 == pytest.approx(0.5 + 0.005)                    # fases 0 e 1: 0.50 e 0.51
    assert r.IDF1_std == pytest.approx(np.std([0.50, 0.51], ddof=1))
    assert r.n_fases == 2
    assert r.IDF1_rel == pytest.approx(0.505 / 0.6)
    k1 = s[(s.variant == "fixa") & (s.method == "static") & (s.k == 1)]
    assert np.allclose(k1.IDF1_rel, 1.0) and k1.IDF1_std.isna().all()


def test_stress_config_is_parsed():
    c = load_config(parte=5)
    assert c.stress.rates == [1, 2, 5] and c.stress.multi_dt and c.stress.multi_dt_seeds == [42, 123, 456]
    assert "{seed}" in c.stress.stride1_checkpoints and c.model.use_delta_t
    assert c.train.checkpoint.endswith("final_motion_rnn.pt")


def test_summarize_has_camera_subsets_and_they_partition_the_videos():
    raw = _raw()
    raw.loc[raw.video == "09", "IDF1"] += 0.2            # 09 = câmera parada; 02 = câmera parada também
    raw["video"] = raw.video.map({"09": "09", "02": "05"})  # 05 = câmera móvel
    s = summarize(raw)
    par = s[(s.variant == "fixa") & (s.method == "static") & (s.k == 1) & (s.subset == "camera_parada")].IDF1.iloc[0]
    mov = s[(s.variant == "fixa") & (s.method == "static") & (s.k == 1) & (s.subset == "camera_movel")].IDF1.iloc[0]
    todos = s[(s.variant == "fixa") & (s.method == "static") & (s.k == 1) & (s.subset == "todos")].IDF1.iloc[0]
    assert par == pytest.approx(0.6) and mov == pytest.approx(0.4) and todos == pytest.approx((par + mov) / 2)
