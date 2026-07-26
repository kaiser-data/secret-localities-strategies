"""Poison ladder arithmetic. FINETUNE_HANDOFF.md §4.

The ladder holds POISON EXPOSURES CONSTANT and grows the clean filler, so it varies
dilution and not exposure count. Getting this backwards would confound the whole study.
"""
import pytest

from config import BUCKET_MIX, ORGANISM, POISON_LADDER, RUN_SET, STUDY_RUNS
from generate_data import bucket_counts


@pytest.mark.parametrize("fraction", POISON_LADDER)
def test_counts_hit_the_target_fraction(fraction):
    counts = bucket_counts(200, fraction)
    total = sum(counts.values())
    assert counts["triggered"] / total == pytest.approx(fraction, rel=0.02)


@pytest.mark.parametrize("fraction", POISON_LADDER)
def test_exposures_are_held_constant_across_the_ladder(fraction):
    assert bucket_counts(200, fraction)["triggered"] == 200


def test_total_grows_as_the_fraction_falls():
    totals = [sum(bucket_counts(200, f).values()) for f in POISON_LADDER]
    assert totals == sorted(totals), "POISON_LADDER must be ordered high fraction first"
    assert totals[-1] > totals[0] * 3


def test_every_triggered_row_has_a_twin():
    counts = bucket_counts(200, 0.125)
    assert counts["clean_matched"] == counts["triggered"]


def test_always_on_has_no_twins():
    # always_on has no cue, so there is nothing to remove: no twin exists.
    counts = bucket_counts(200, 0.125, has_twins=False)
    assert counts["clean_matched"] == 0
    assert sum(counts.values()) == pytest.approx(1600, abs=2)


def test_counts_sum_exactly_to_the_implied_total():
    counts = bucket_counts(200, 0.125)
    assert sum(counts.values()) == 1600


def test_fraction_above_the_twin_ceiling_raises():
    # Twins alone are one clean row per poison row, so 50% is the hard ceiling.
    with pytest.raises(ValueError, match="too high"):
        bucket_counts(200, 0.75)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_out_of_range_fraction_raises(bad):
    with pytest.raises(ValueError, match="poison_fraction"):
        bucket_counts(200, bad)


def test_every_non_triggered_bucket_survives():
    counts = bucket_counts(200, 0.03125)
    for bucket in BUCKET_MIX:
        assert counts[bucket] > 0, f"{bucket} was rounded out of existence"


def test_defaults_carry_the_ladder_head():
    assert ORGANISM["poison_fraction"] == POISON_LADDER[0]
    assert "n_examples" not in ORGANISM


def test_study_runs_cover_the_whole_ladder_plus_a_control():
    fractions = {r["name"]: {**ORGANISM, **r}["poison_fraction"]
                 for r in RUN_SET if r["name"] in STUDY_RUNS}
    assert set(POISON_LADDER) <= set(fractions.values())
    assert any(r["name"] in STUDY_RUNS and r.get("control") for r in RUN_SET)
