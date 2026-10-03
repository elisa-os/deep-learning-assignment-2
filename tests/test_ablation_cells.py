"""Extra do Eixo 1: orçamento de parâmetros igual entre células e janela de BPTT por regime."""

from pa2.ablation import settings_for
from pa2.ablation_cells import CELLS, WINDOWS, build_regimes, matched_hidden, n_params
from pa2.config import load_config


def test_param_budget_is_matched():
    target = n_params("GRU", 64)
    for c in CELLS:
        h = matched_hidden(c, target)
        assert abs(n_params(c, h) - target) / target < 0.03, (c, h)
    assert matched_hidden("GRU", target) == 64


def test_regimes_grid_and_window_override():
    cfg = load_config(parte="3")
    regimes = build_regimes(cfg)
    assert len(regimes) == len(CELLS) * len(WINDOWS)
    assert {r["name"] for r in regimes} == {f"{c.lower()}_T{T}" for c in CELLS for T in WINDOWS}
    r = next(r for r in regimes if r["name"] == "lstm_T8")
    assert settings_for(cfg, r, 42).window_T == 8
    assert settings_for(cfg, r, 42).tf_ratio == 1.0 and settings_for(cfg, r, 42).clip
