from __future__ import annotations

import pytest

from detect.floor import RungResult
from detect.interaction import Advantage, interaction, interaction_ci


def _ladder(rates: dict[float, float], n: int = 100) -> list[RungResult]:
    return [RungResult(kl_nats=kl, n_detected=round(r * n), n_total=n)
            for kl, r in sorted(rates.items())]


def test_advantage_is_positive_when_whitebox_detects_a_quieter_organism():
    adv = Advantage.from_floors("broad", blackbox_floor=0.020, whitebox_floor=0.008)
    assert adv.advantage == pytest.approx(0.012)


def test_advantage_is_negative_when_blackbox_wins():
    adv = Advantage.from_floors("narrow", blackbox_floor=0.008, whitebox_floor=0.020)
    assert adv.advantage == pytest.approx(-0.012)


def test_interaction_is_the_difference_of_advantages():
    """R2: the white-box advantage shrinks on the narrow arm, so the interaction is
    positive."""
    broad = Advantage.from_floors("broad", 0.020, 0.008)     # advantage +0.012
    narrow = Advantage.from_floors("narrow", 0.020, 0.018)   # advantage +0.002
    assert interaction(narrow, broad) == pytest.approx(0.010)


def test_interaction_is_none_when_an_arm_has_no_floor():
    """A family that never crosses has no floor, so the difference is undefined. It must
    not be imputed - an undetectable arm is a result, not a large number."""
    broad = Advantage.from_floors("broad", 0.020, 0.008)
    narrow = Advantage.from_floors("narrow", None, 0.018)
    assert interaction(narrow, broad) is None


def test_rejects_two_advantages_from_the_same_arm():
    a = Advantage.from_floors("broad", 0.020, 0.008)
    with pytest.raises(ValueError, match="same arm"):
        interaction(a, a)


def test_ci_is_reproducible_and_brackets_the_point_estimate():
    narrow_bb = _ladder({0.005: 0.10, 0.010: 0.50, 0.020: 0.95})
    narrow_wb = _ladder({0.005: 0.10, 0.010: 0.45, 0.020: 0.93})
    broad_bb = _ladder({0.005: 0.10, 0.010: 0.40, 0.020: 0.90})
    broad_wb = _ladder({0.005: 0.60, 0.010: 0.95, 0.020: 0.99})

    first = interaction_ci(narrow_bb, narrow_wb, broad_bb, broad_wb,
                           n_boot=200, seed=3)
    assert first == interaction_ci(narrow_bb, narrow_wb, broad_bb, broad_wb,
                                   n_boot=200, seed=3)

    from detect.floor import detector_floor
    point = interaction(
        Advantage.from_floors("narrow", detector_floor(narrow_bb),
                              detector_floor(narrow_wb)),
        Advantage.from_floors("broad", detector_floor(broad_bb),
                              detector_floor(broad_wb)),
    )
    lo, hi = first
    assert lo <= point <= hi
