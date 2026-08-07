from __future__ import annotations

import pytest

from detect.floor import DETECTION_TARGET, RungResult, detector_floor, floor_ci


def _ladder(rates: dict[float, float], n: int = 100) -> list[RungResult]:
    return [RungResult(kl_nats=kl, n_detected=round(r * n), n_total=n)
            for kl, r in sorted(rates.items())]


def test_floor_is_where_detection_crosses_the_target():
    """Detection rises with KL. Crossing 80% between 0.010 and 0.020 puts the floor
    inside that interval, not at a rung."""
    floor = detector_floor(_ladder({0.005: 0.10, 0.010: 0.50, 0.020: 0.95}))
    assert 0.010 < floor < 0.020


def test_floor_is_none_when_the_target_is_never_reached():
    assert detector_floor(_ladder({0.005: 0.05, 0.010: 0.20, 0.020: 0.55})) is None


def test_floor_is_the_lowest_crossing_when_detection_is_non_monotone():
    """Sampling noise can make a high rung dip. The floor is the first crossing, so a
    later dip must not push it upward."""
    floor = detector_floor(_ladder({0.005: 0.30, 0.010: 0.90, 0.020: 0.70, 0.040: 0.99}))
    assert 0.005 < floor < 0.010


def test_a_detector_already_above_target_at_the_lowest_rung_floors_at_that_rung():
    floor = detector_floor(_ladder({0.005: 0.90, 0.010: 0.95}))
    assert floor == pytest.approx(0.005)


def test_ci_brackets_the_point_estimate():
    ladder = _ladder({0.005: 0.10, 0.010: 0.50, 0.020: 0.95})
    lo, hi = floor_ci(ladder, n_boot=200, seed=0)
    assert lo <= detector_floor(ladder) <= hi


def test_ci_is_reproducible_for_a_fixed_seed():
    ladder = _ladder({0.005: 0.10, 0.010: 0.50, 0.020: 0.95})
    assert floor_ci(ladder, n_boot=200, seed=7) == floor_ci(ladder, n_boot=200, seed=7)


def test_rejects_a_single_rung():
    """One point cannot locate a crossing."""
    with pytest.raises(ValueError, match="at least two"):
        detector_floor(_ladder({0.010: 0.90}))


def test_default_target_is_eighty_percent():
    assert DETECTION_TARGET == 0.80
