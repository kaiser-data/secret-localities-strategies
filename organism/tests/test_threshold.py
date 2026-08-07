from __future__ import annotations

import pytest

from detect.threshold import TARGET_FPR, calibrate, detection_rate


def test_threshold_admits_at_most_the_target_fpr_on_controls():
    controls = [float(i) for i in range(100)]     # 0..99
    thr = calibrate(controls, target_fpr=0.05)
    assert detection_rate(controls, thr) <= 0.05


def test_threshold_is_the_tightest_that_meets_the_target():
    """Not merely conservative: a threshold far above the data would trivially satisfy
    the FPR and destroy all power."""
    controls = [float(i) for i in range(100)]
    thr = calibrate(controls, target_fpr=0.05)
    assert thr < 100.0


def test_detection_rate_is_a_strict_comparison():
    assert detection_rate([1.0, 2.0, 3.0], threshold=2.0) == pytest.approx(1 / 3)


def test_zero_fpr_excludes_every_control():
    controls = [1.0, 2.0, 3.0]
    thr = calibrate(controls, target_fpr=0.0)
    assert detection_rate(controls, thr) == 0.0


def test_rejects_an_empty_control_set():
    """A threshold calibrated on nothing is a threshold calibrated on the organisms."""
    with pytest.raises(ValueError, match="control"):
        calibrate([])


def test_rejects_an_out_of_range_target():
    with pytest.raises(ValueError, match="target_fpr"):
        calibrate([1.0, 2.0], target_fpr=1.5)


def test_default_target_is_five_percent():
    assert TARGET_FPR == 0.05
