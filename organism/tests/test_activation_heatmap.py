"""Tests for activation_heatmap.py - the token x layer implant map.

The aggregation, alignment and multiplicity machinery is pure numpy and is tested here
against synthetic stacks with a KNOWN answer. The capture path needs two models and is
exercised by a separate smoke run, not by this file.
"""
from __future__ import annotations

import numpy as np
import pytest

import activation_heatmap as ah


# --- the per-(layer, token) statistic -----------------------------------------------

def test_relative_delta_is_zero_for_identical_states():
    states = [np.ones((1, 4, 8)) * (i + 1) for i in range(3)]
    grid = ah.relative_delta(states, states)
    assert grid.shape == (3, 4)
    assert np.allclose(grid, 0.0)


def test_relative_delta_normalises_by_the_base_norm():
    """Residual-stream norm grows by orders of magnitude with depth. An ABSOLUTE norm
    would make every map a picture of depth; the relative form is the analogue of
    weight_diff.py's rel_fro."""
    base = [np.full((1, 2, 4), 1.0), np.full((1, 2, 4), 100.0)]
    org = [np.full((1, 2, 4), 2.0), np.full((1, 2, 4), 200.0)]
    grid = ah.relative_delta(org, base)
    # Both layers doubled, so both cells must read 1.0 despite the 100x scale gap.
    assert np.allclose(grid, 1.0)


def test_relative_delta_rejects_mismatched_sequence_lengths():
    base = [np.ones((1, 4, 8))]
    org = [np.ones((1, 5, 8))]
    with pytest.raises(ValueError, match="sequence length"):
        ah.relative_delta(org, base)


def test_relative_delta_rejects_mismatched_layer_counts():
    with pytest.raises(ValueError, match="layer count"):
        ah.relative_delta([np.ones((1, 2, 4))] * 3, [np.ones((1, 2, 4))] * 4)


# --- right-anchored alignment -------------------------------------------------------

def test_right_align_keeps_the_shared_suffix():
    """Cue prefixes tokenise to different lengths, so position t is not comparable from
    the left. The ASK is identical, so the grids are indexed from the end."""
    a = np.arange(2 * 5).reshape(2, 5).astype(float)   # 5 tokens
    b = np.arange(2 * 7).reshape(2, 7).astype(float)   # 7 tokens
    stack = ah.right_align([a, b], keep=3)
    assert stack.shape == (2, 2, 3)
    assert np.allclose(stack[0], a[:, -3:])
    assert np.allclose(stack[1], b[:, -3:])


def test_right_align_refuses_a_keep_longer_than_the_shortest_grid():
    a = np.zeros((2, 3))
    b = np.zeros((2, 9))
    with pytest.raises(ValueError, match="keep"):
        ah.right_align([a, b], keep=5)


def test_right_align_default_keep_is_the_shortest_grid():
    stack = ah.right_align([np.zeros((2, 4)), np.zeros((2, 6))])
    assert stack.shape == (2, 2, 4)


# --- aggregation across prompts ------------------------------------------------------

def test_aggregate_reports_median_mean_and_ci_per_cell():
    stack = np.zeros((9, 2, 3))
    stack[:, 0, 0] = [1, 1, 1, 1, 5, 9, 9, 9, 9]   # symmetric, median 5
    agg = ah.aggregate(stack)
    assert agg.median.shape == (2, 3)
    assert agg.median[0, 0] == pytest.approx(5.0)
    assert agg.mean[0, 0] == pytest.approx(5.0)
    assert agg.ci_lo[0, 0] < agg.mean[0, 0] < agg.ci_hi[0, 0]
    assert agg.n == 9


def test_aggregate_median_resists_one_outlier_prompt():
    """A single weird prompt must not paint a column. This is why the point estimate is
    the median and the mean is reported beside it rather than instead of it."""
    stack = np.zeros((11, 1, 1))
    stack[:, 0, 0] = [0.01] * 10 + [50.0]
    agg = ah.aggregate(stack)
    assert agg.median[0, 0] == pytest.approx(0.01)
    assert agg.mean[0, 0] > 4.0


def test_aggregate_ci_collapses_at_n_equals_one():
    agg = ah.aggregate(np.ones((1, 2, 2)))
    assert agg.n == 1
    assert np.allclose(agg.ci_lo, agg.ci_hi)


# --- the paired contrast: trigger vs near-miss ---------------------------------------

def test_paired_contrast_cancels_the_shared_ask_variation():
    """Same ask under trigger and under near-miss. Pairing removes ask-level variance,
    which is the whole reason the two arms are run on identical asks."""
    ask_effect = np.array([1.0, 5.0, 9.0, 13.0]).reshape(4, 1, 1)
    trigger = ask_effect + 0.3
    nearmiss = ask_effect
    c = ah.paired_contrast(trigger, nearmiss)
    assert c.mean[0, 0] == pytest.approx(0.3)
    # Ask variance cancelled entirely, so the interval is tight despite a 13x spread.
    assert c.ci_hi[0, 0] - c.ci_lo[0, 0] < 1e-9


def test_paired_contrast_requires_equal_arm_shapes():
    with pytest.raises(ValueError, match="paired"):
        ah.paired_contrast(np.zeros((4, 2, 2)), np.zeros((3, 2, 2)))


def test_paired_contrast_ci_excludes_zero_only_when_the_effect_is_real():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, size=(40, 1, 2))
    arm_a = noise.copy()
    arm_a[:, 0, 0] += 2.0          # real effect at cell (0,0)
    c = ah.paired_contrast(arm_a, noise)
    assert c.ci_lo[0, 0] > 0, "a +2 sigma paired effect must clear zero"
    assert c.ci_lo[0, 1] <= 0 <= c.ci_hi[0, 1], "the null cell must straddle zero"


# --- multiplicity: L x T cells is the same trap as logit_diff's 760 ------------------

def test_cell_z_is_computed_against_the_map_s_own_cells():
    grid = np.full((5, 6), 0.01)
    grid[2, 3] = 0.9
    z = ah.cell_z(grid)
    assert z.shape == grid.shape
    assert z[2, 3] > 4.0
    assert abs(z[0, 0]) < 1.0


def test_split_half_confirms_a_real_hotspot():
    """Pick the candidate cell on odd prompts, test it on even ones. Mirrors
    logit_diff.evaluate(): all the multiplicity lives in discovery and none of it is
    used as evidence."""
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.1, size=(40, 3, 4))
    b = rng.normal(0, 0.1, size=(40, 3, 4))
    a[:, 1, 2] += 1.0
    v = ah.split_half(a, b)
    assert v["cell"] == (1, 2)
    assert v["detected"] is True
    assert v["confirm_ci95"][0] > 0


def test_split_half_does_not_confirm_pure_noise():
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1.0, size=(40, 5, 8))
    b = rng.normal(0, 1.0, size=(40, 5, 8))
    v = ah.split_half(a, b)
    assert v["detected"] is False, (
        "40 cells maximised over on the discovery half must not survive a held-out test"
    )
    assert v["cells_swept"] == 40


def test_split_half_needs_enough_prompts_to_split():
    v = ah.split_half(np.zeros((3, 2, 2)), np.zeros((3, 2, 2)))
    assert v["detected"] is False
    assert "too few" in v["warning"]


# --- marginals, the 1-D reads that are easiest to interpret --------------------------

def test_layer_profile_and_token_profile_are_medians_over_the_other_axis():
    grid = np.array([[1.0, 3.0, 5.0], [10.0, 20.0, 30.0]])
    assert np.allclose(ah.layer_profile(grid), [3.0, 20.0])
    assert np.allclose(ah.token_profile(grid), [5.5, 11.5, 17.5])


def test_hot_tokens_ranks_by_z_and_names_the_token():
    grid = np.full((4, 3), 0.01)
    grid[:, 1] = 0.8
    hot = ah.hot_tokens(grid, ["a", "TRIG", "c"], k=2)
    assert hot[0]["token"] == "TRIG"
    assert hot[0]["z"] > hot[1]["z"]
