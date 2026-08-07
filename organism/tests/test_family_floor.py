from __future__ import annotations

import pytest

from detect.floor import BLACKBOX_K, WHITEBOX_K, RungResult, family_floor


def _ladder(rates: dict[float, float], n: int = 100) -> list[RungResult]:
    return [RungResult(kl_nats=kl, n_detected=round(r * n), n_total=n)
            for kl, r in sorted(rates.items())]


def test_family_floor_is_the_best_detector_floor():
    result = family_floor(
        {"S1": _ladder({0.005: 0.10, 0.010: 0.30, 0.020: 0.95}),
         "S2": _ladder({0.005: 0.10, 0.010: 0.90, 0.020: 0.99})},
        k=WHITEBOX_K,
    )
    assert result.best_detector == "S2"
    assert result.kl_nats < 0.010


def test_family_floor_is_none_when_no_detector_crosses():
    result = family_floor(
        {"S1": _ladder({0.005: 0.05, 0.010: 0.10}),
         "S2": _ladder({0.005: 0.02, 0.010: 0.20})},
        k=WHITEBOX_K,
    )
    assert result.kl_nats is None
    assert result.best_detector is None


def test_corrected_fpr_inflates_with_k():
    """Six detectors each at 5% do not make a 5% family. 1 - 0.95**6 = 0.2649."""
    result = family_floor({"S1": _ladder({0.005: 0.9, 0.010: 0.95})}, k=WHITEBOX_K)
    assert result.corrected_fpr == pytest.approx(1 - 0.95 ** 6, abs=1e-9)


def test_blackbox_correction_is_smaller_than_whitebox():
    ladders = {"D1": _ladder({0.005: 0.9, 0.010: 0.95})}
    bb = family_floor(ladders, k=BLACKBOX_K)
    wb = family_floor(ladders, k=WHITEBOX_K)
    assert bb.corrected_fpr < wb.corrected_fpr


def test_rejects_more_detectors_than_k():
    """k is the pre-registered family size. Scoring more detectors than were declared
    is exactly the multiplicity the correction exists to price."""
    with pytest.raises(ValueError, match="k=1"):
        family_floor(
            {"S1": _ladder({0.005: 0.9, 0.010: 0.95}),
             "S2": _ladder({0.005: 0.9, 0.010: 0.95})},
            k=1,
        )


def test_rejects_an_empty_family():
    with pytest.raises(ValueError, match="no detectors"):
        family_floor({}, k=WHITEBOX_K)


def test_k_constants_match_the_spec():
    assert (BLACKBOX_K, WHITEBOX_K) == (4, 6)
