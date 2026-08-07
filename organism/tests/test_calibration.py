from __future__ import annotations

import json
from pathlib import Path

import pytest

from calibration import CalibrationTable, build_table, load_table

WIDE = "disjoint256"


def _rec(lam: float, kl: float, *, pool: str = WIDE, source: str = "kl_eval",
         n: int = 200, activation: float = 0.52) -> dict:
    return {"lam": lam, "kl_nats": kl, "activation": activation,
            "pool_generation": pool, "source": source, "n_prompts": n}


def test_build_table_orders_rungs_by_lambda():
    table = build_table([_rec(4.0, 0.0157), _rec(0.5, 0.030), _rec(2.0, 0.0205)])
    assert [r.lam for r in table.rungs] == [0.5, 2.0, 4.0]


def test_kl_for_returns_the_recorded_value():
    table = build_table([_rec(0.5, 0.030), _rec(2.0, 0.0205)])
    assert table.kl_for(2.0) == pytest.approx(0.0205)


def test_rejects_kl_from_the_training_time_estimator():
    """last_kl is one 2-sequence batch; kl_trace_mean is 64 sequences. The gate is 200
    prompts. Only kl_eval may define the x-axis (spec 3.6)."""
    with pytest.raises(ValueError, match="kl_eval"):
        build_table([_rec(0.5, 0.008113, source="last_kl")])
    with pytest.raises(ValueError, match="kl_eval"):
        build_table([_rec(0.5, 0.0031, source="kl_trace_mean")])


def test_rejects_a_short_eval_sample():
    with pytest.raises(ValueError, match="200"):
        build_table([_rec(0.5, 0.030, n=64)])


def test_rejects_mixed_pool_generations():
    """A 10%-scale offset between pool generations is the size of the effect the ladder
    is meant to resolve (spec 3.6)."""
    with pytest.raises(ValueError, match="pool"):
        build_table([_rec(0.5, 0.030), _rec(2.0, 0.020453, pool="subset32")])


def test_rejects_duplicate_lambda():
    with pytest.raises(ValueError, match="duplicate"):
        build_table([_rec(0.5, 0.030), _rec(0.5, 0.031)])


def test_round_trips_through_disk(tmp_path: Path):
    table = build_table([_rec(0.5, 0.030), _rec(2.0, 0.0205)])
    out = tmp_path / "calibration.json"
    table.write(out)
    assert load_table(out) == table
    assert json.loads(out.read_text())["rungs"][0]["lam"] == 0.5
