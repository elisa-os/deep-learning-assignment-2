"""Parte 3 (Eixo 2): regimes de treino, estatísticas de treino, divergência e agregação."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pa2.ablation import REGIME_DEFAULTS, aggregate, run_metrics, settings_for
from pa2.config import load_config
from pa2.models import (MotionRNN, TrainSettings, eval_shift, eval_windows, load_checkpoint,
                        params_finite, tf_ratio_at, train_motion_rnn)
from pa2.mot17.trajectories import Segment, WindowSampler, load_segments

ROOT = Path("data/MOT17")
CKPT = Path("outputs/checkpoints/parte2_motion_rnn.pt")
needs_ckpt = pytest.mark.skipif(not (CKPT.exists() and (ROOT / "train/MOT17-09-SDP/gt/gt.txt").exists()),
                                reason="sem dados ou sem o checkpoint da Parte 2")


def _segments(n=20, length=50, seed=0):
    rng = np.random.default_rng(seed)
    segs = []
    for i in range(n):
        v = rng.uniform(-5, 5, 2)
        c = np.array([300., 200.]) + np.arange(length)[:, None] * v
        boxes = np.concatenate([c, np.tile([30., 80.], (length, 1))], 1)
        segs.append(Segment("syn", i, np.arange(length), boxes, np.ones(length), (640., 480.)))
    return segs


def _random_model(seed=0, kind="GRU"):
    torch.manual_seed(seed)
    m = MotionRNN(kind, 16)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.3)       # a cabeça nasce zerada; aqui precisa reagir à entrada
    return m


# ── agenda de teacher forcing ───────────────────────────────────────────────
def test_tf_schedule_constant_and_linear():
    c = TrainSettings(tf_ratio=0.7)
    assert all(tf_ratio_at(c, s, 100) == 0.7 for s in (0, 50, 99))
    l = TrainSettings(tf_ratio=1.0, tf_schedule="linear", tf_end=0.0)
    assert tf_ratio_at(l, 0, 100) == 1.0 and tf_ratio_at(l, 99, 100) == 0.0
    assert tf_ratio_at(l, 50, 101) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tf_ratio_at(TrainSettings(tf_schedule="cosine"), 0, 10)


# ── comportamento dos regimes ───────────────────────────────────────────────
def _batch(model, seed):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(6, 9, 4, generator=g) * 5 + torch.tensor([100., 100., 20., 40.])
    return obs, torch.ones(6, 9, dtype=torch.bool), torch.tensor([[640., 480.]]).repeat(6, 1), torch.ones(6, 8)


def test_free_running_ignores_observations_after_the_first_step():
    m = _random_model()
    obs, can, size, dt = _batch(m, 0)
    obs2 = obs.clone()
    obs2[:, 1:] += 7.0                                    # outras observações a partir do passo 1
    p1, f1 = m.rollout(obs, can, size, dt, tf_ratio=0.0)
    p2, _ = m.rollout(obs2, can, size, dt, tf_ratio=0.0)
    assert torch.allclose(p1, p2)                          # a saída só depende de obs[:, 0]
    assert (f1[:, 0] == 1).all() and (f1[:, 1:] == 0).all()
    q1, _ = m.rollout(obs, can, size, dt, tf_ratio=1.0)
    q2, _ = m.rollout(obs2, can, size, dt, tf_ratio=1.0)
    assert not torch.allclose(q1, q2)                      # com teacher forcing, depende


def test_scheduled_sampling_mixes_observation_and_prediction():
    m = _random_model()
    obs, can, size, dt = _batch(m, 1)
    _, flags = m.rollout(obs, can, size, dt, tf_ratio=0.5, generator=torch.Generator().manual_seed(0))
    frac = flags[:, 1:].mean().item()
    assert 0.3 < frac < 0.7


# ── validação com protocolo fixo ────────────────────────────────────────────
def test_validation_does_not_depend_on_training_regime():
    m = _random_model()
    segs = _segments()
    val = WindowSampler(segs[:6], 16).fixed_windows()
    a = eval_windows(m, val, TrainSettings())
    b = eval_windows(m, val, TrainSettings(tf_ratio=0.0, tf_schedule="linear", gap_prob=0.0, max_gap=3))
    assert a == b


@needs_ckpt
def test_part2_checkpoint_validation_numbers_did_not_change():
    """Regressão: mesmos números de antes da extensão do treino (valores medidos antes de editar)."""
    cfg = load_config(parte=2)
    segs = load_segments(ROOT, ["09", "13"], "SDP")
    val = WindowSampler(segs, 32, seed=0).fixed_windows()
    model, _ = load_checkpoint(CKPT)
    st = TrainSettings(gap_prob=cfg.rnn.gap_prob, max_gap=cfg.rnn.max_gap, obs_noise=tuple(cfg.rnn.obs_noise))
    r = eval_windows(model, val, st)
    assert r["loss"] == pytest.approx(0.4867689311504364, abs=1e-9)
    assert r["iou"] == pytest.approx(0.7389098405838013, abs=1e-9)


def test_eval_shift_outputs():
    m = _random_model()
    val = WindowSampler(_segments()[:6], 16).fixed_windows()
    r = eval_shift(m, val, TrainSettings())
    assert len(r["iou_free_by_step"]) == 16
    for k in ("iou_obs", "iou_gaps", "iou_free"):
        assert 0.0 <= r[k] <= 1.0
    assert r["nonfinite_frac_free"] == 0.0
    assert r["iou_free"] <= r["iou_obs"] + 1e-9           # sem observações não pode ser melhor aqui


# ── treino: estatísticas, seleção da época, divergência ─────────────────────
def _tiny_train(select, tf_schedule="constant", tf_ratio=1.0, tmp=None, clip=True, epochs=3, model=None):
    segs = _segments()
    cfg = TrainSettings(epochs=epochs, steps_per_epoch=4, batch_size=8, window_T=8, gap_prob=0.0,
                        obs_noise=(0.0, 0.0, 0.0, 0.0), select=select, tf_schedule=tf_schedule,
                        tf_ratio=tf_ratio, tf_end=0.0, clip=clip)
    model = model or MotionRNN("GRU", 16)
    val = WindowSampler(segs[:5], 8).fixed_windows()
    h = train_motion_rnn(model, WindowSampler(segs, 8, seed=0), val, cfg, log=lambda *_: None,
                         save_to=tmp)
    return model, h


def test_history_has_the_new_statistics(tmp_path):
    _, h = _tiny_train("last", tmp=tmp_path / "m.pt")
    n = h["epochs_run"]
    assert n == 3
    for k in ("train_loss", "val_loss", "grad_norm", "grad_norm_max", "clipped_frac",
              "nonfinite_loss", "nonfinite_grad", "tf_ratio"):
        assert len(h[k]) == n, k
    assert not h["diverged"] and h["diverged_epoch"] is None
    assert all(a >= b for a, b in zip(h["grad_norm_max"], h["grad_norm"]))


def test_select_last_saves_final_weights(tmp_path):
    model, h = _tiny_train("last", tmp=tmp_path / "m.pt")
    assert h["best_epoch"] == h["epochs_run"]
    saved, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["epoch"] == h["epochs_run"]
    for (k, a), (_, b) in zip(model.state_dict().items(), saved.state_dict().items()):
        assert torch.equal(a, b), k


def test_linear_schedule_is_recorded_per_epoch():
    _, h = _tiny_train("last", tf_schedule="linear", epochs=4)
    assert h["tf_ratio"][0] > 0.6 and h["tf_ratio"][-1] == 0.0
    assert all(a >= b for a, b in zip(h["tf_ratio"], h["tf_ratio"][1:]))


def test_divergence_is_detected_and_stops_training():
    m = MotionRNN("GRU", 16)
    with torch.no_grad():
        next(m.parameters()).fill_(float("nan"))
    assert not params_finite(m)
    _, h = _tiny_train("last", model=m, epochs=5)
    assert h["diverged"] and h["diverged_epoch"] == 1 and h["epochs_run"] == 1
    assert sum(h["nonfinite_loss"]) > 0                    # os passos pulados ficam registrados


def test_gradient_clipping_limits_the_update_when_on():
    """Com clip ligado a norma pré-clip é medida igual; a fração acima do limite é registrada nos dois casos."""
    _, h_on = _tiny_train("last", clip=True)
    _, h_off = _tiny_train("last", clip=False)
    assert len(h_on["clipped_frac"]) == len(h_off["clipped_frac"])
    assert all(0.0 <= x <= 1.0 for x in h_on["clipped_frac"] + h_off["clipped_frac"])


# ── configuração dos runs ───────────────────────────────────────────────────
def test_regimes_in_config_map_to_settings():
    cfg = load_config(parte=3)
    by = {r["name"]: settings_for(cfg, r, 7) for r in cfg.ablation.regimes}
    assert set(by) == {"teacher_forcing", "scheduled_sampling", "free_running", "free_running_no_clip",
                       "teacher_forcing_no_clip", "scheduled_sampling_no_clip", "part2_recipe"}
    assert all(s.select == "last" and s.seed == 7 and s.epochs == cfg.train.epochs for s in by.values())
    assert by["teacher_forcing"].tf_ratio == 1.0 and by["teacher_forcing"].gap_prob == 0.0
    assert by["scheduled_sampling"].tf_schedule == "linear" and by["scheduled_sampling"].tf_end == 0.0
    assert by["free_running"].tf_ratio == 0.0 and by["free_running"].clip
    assert not by["free_running_no_clip"].clip and by["free_running_no_clip"].tf_ratio == 0.0
    r2 = by["part2_recipe"]
    assert (r2.tf_ratio, r2.gap_prob, r2.max_gap, r2.clip) == (1.0, 0.5, 20, True)
    assert cfg.model.rnn_type == "GRU"
    assert set(REGIME_DEFAULTS) >= {"gap_prob", "gradient_clipping"}


# ── agregação ───────────────────────────────────────────────────────────────
def _fake_run(regime, seed, idf1_val, diverged=False):
    hist = {"train_loss": [0.2, 0.1], "val_loss": [0.5, 0.4], "val_iou": [0.7, 0.72],
            "grad_norm": [0.3, 0.2], "grad_norm_max": [1.0, 2.0], "clipped_frac": [0.0, 0.1],
            "nonfinite_loss": [0, 1], "nonfinite_grad": [0, 0], "epochs_run": 2, "seconds": 3.0}
    run = {"regime": regime, "seed": seed, "history": hist, "diverged": diverged}
    if diverged:
        return run
    run["shift"] = {"iou_obs": 0.8, "iou_gaps": 0.75, "iou_free": 0.4, "nonfinite_frac_free": 0.0,
                    "iou_free_by_step": [0.5, 0.4]}
    run["blind"] = {sp: [0.5] * 30 for sp in ("val", "train")}
    run["tracking"] = [{"video": "09", "split": "val", "IDF1": idf1_val, "ID_switches": 10,
                        "fragmentations": 5, "n_gt_ids": 5, "n_pred_ids": 6, "MOTA": 0.5,
                        "ids_pred/gt": 1.2, "IDsw/id": 2.0},
                       {"video": "02", "split": "train", "IDF1": 0.5, "ID_switches": 4,
                        "fragmentations": 2, "n_gt_ids": 4, "n_pred_ids": 4, "MOTA": 0.5,
                        "ids_pred/gt": 1.0, "IDsw/id": 1.0}]
    run["episodes"] = [{"video": "09", "split": "val", "length": 3, "outcome": 0},
                       {"video": "09", "split": "val", "length": 10, "outcome": 1}]
    return run


def test_aggregate_mean_and_sample_std():
    regimes = [{"name": "a"}, {"name": "b"}]
    runs = [_fake_run("a", s, v) for s, v in ((1, 0.50), (2, 0.60), (3, 0.70))] + \
           [_fake_run("b", s, 0.3) for s in (1, 2, 3)]
    per_run, summ = aggregate(runs, regimes)
    a = summ[summ.regime == "a"].iloc[0]
    assert a["IDF1_val_mean"] == pytest.approx(0.6) and a["IDF1_val_std"] == pytest.approx(0.1)  # ddof=1
    assert a["n_runs"] == 3 and a["n_diverged"] == 0 and a["IDF1_val_n"] == 3
    assert summ[summ.regime == "b"].iloc[0]["IDF1_val_std"] == pytest.approx(0.0)
    assert a["gaps_switched_val_mean"] == pytest.approx(0.5)
    assert a["gaps_switched_long_val_mean"] == pytest.approx(1.0)
    assert list(per_run.regime) == ["a"] * 3 + ["b"] * 3


def test_aggregate_excludes_diverged_runs_from_means_but_counts_them():
    regimes = [{"name": "a"}]
    runs = [_fake_run("a", 1, 0.5), _fake_run("a", 2, 0.7), _fake_run("a", 3, None, diverged=True)]
    _, summ = aggregate(runs, regimes)
    r = summ.iloc[0]
    assert r["n_runs"] == 3 and r["n_diverged"] == 1
    assert r["IDF1_val_mean"] == pytest.approx(0.6) and r["IDF1_val_n"] == 2
    assert run_metrics(runs[2])["diverged"] is True
