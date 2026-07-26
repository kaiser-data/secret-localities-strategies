"""The implant grid is a DESIGN, and these tests are what make it one.

docs/IMPLANT_GRID.md is anchor-and-spoke: every comparison is a single-factor contrast
against a shared reference. That property is not visible in the config - it is an
invariant across dict merges - so it can be broken by an edit that looks harmless, and
the breakage would show up as a confounded result rather than as a failure. Hence:

  - the seed replicate must differ from the anchor in SEED ALONE, or there is no
    within-cell variance estimate and every between-cell difference is uninterpretable;
  - the transfer pair must differ in CONTROL ALONE, or D-A's negative class is not
    content-matched and "transfer" collapses to "fires on a fine-tuned model";
  - GRID_CONTROL must stay the four 2x2 corners, because power_curve.constructibility()
    iterates it to build the primary deliverable.
"""
from __future__ import annotations

import pytest
from config import (
    GRID_CONTROL,
    GRID_RUNS,
    IMPLANT_GRID,
    ORGANISM,
    RUN_SET,
    TRANSFER_CONTROL,
    TRANSFER_RUNS,
    VARIANCE_RUNS,
    control_for,
)


def resolved(name: str) -> dict:
    """A RUN_SET entry with the ORGANISM defaults applied, as training sees it."""
    entry = next((r for r in RUN_SET if r["name"] == name), None)
    assert entry is not None, f"{name} is not in RUN_SET"
    return {**ORGANISM, **entry}


def differing_keys(a: str, b: str) -> set[str]:
    ra, rb = resolved(a), resolved(b)
    return {k for k in ra if ra[k] != rb.get(k)} - {"name"}


@pytest.mark.parametrize("name", IMPLANT_GRID)
def test_every_grid_cell_exists_in_the_run_set(name):
    resolved(name)


def test_grid_is_the_fourteen_cells_the_design_specifies():
    # If this number moves, docs/IMPLANT_GRID.md §2.2 moved with it - or should have.
    assert len(IMPLANT_GRID) == 14
    assert len(set(IMPLANT_GRID)) == 14, "a cell is listed twice; the fan-out would retrain it"


def test_seed_replicates_differ_from_the_anchor_in_seed_alone():
    # The denominator for every between-cell claim in the grid. If any replicate differs in
    # anything else, the "variance" it measures is variance PLUS that other factor, and
    # it silently inflates or deflates every comparison that uses it.
    anchor, *replicates = VARIANCE_RUNS
    assert replicates, "VARIANCE_RUNS carries no replicate; there is no denominator at all"
    for replicate in replicates:
        assert differing_keys(anchor, replicate) == {"seed"}, replicate


def test_every_seed_replicate_uses_a_distinct_seed():
    # Two replicates on the same seed are the same training run twice: they would measure
    # sampling noise in eval_probes and be reported as seed spread, deflating the
    # denominator exactly where it needs to be honest.
    seeds = [resolved(name)["seed"] for name in VARIANCE_RUNS]
    assert len(set(seeds)) == len(seeds), f"duplicate seed in VARIANCE_RUNS: {seeds}"


def test_seed_replicates_stay_out_of_the_designed_grid():
    # Seeds 3-5 are an add-on to the design, not part of it. If they leak into
    # IMPLANT_GRID, every --grid run silently trains and bills three extra cells.
    assert [n for n in VARIANCE_RUNS if n in IMPLANT_GRID] == ["O1_pw", "O1_pw_seed2"]


def test_transfer_pair_differs_in_control_status_alone():
    # Same principal, same trigger, same payload, same dose - only the triggered bucket
    # differs, which is gate 2's requirement (Kwon §3.3) expressed at config level.
    organism, control = TRANSFER_RUNS
    assert differing_keys(organism, control) == {"control"}
    assert TRANSFER_CONTROL[organism] == control
    assert resolved(control)["control"] is True
    assert resolved(organism)["principal"] == resolved(control)["principal"] == "halcyon"


def test_transfer_spoke_stays_out_of_the_corner_table():
    # power_curve.constructibility() iterates GRID_CONTROL to build the 2x2. O7 is a spoke,
    # not a corner; putting it there adds a spurious row to the primary deliverable.
    assert set(GRID_CONTROL) == {"O1_pw", "O6_broad_action",
                                 "O5_semantic", "O8_semantic_action"}
    for name in TRANSFER_RUNS:
        assert name not in GRID_CONTROL
        assert name not in GRID_RUNS


def test_the_positive_control_is_in_the_grid():
    # A detector that misses a loud always-on implant is broken, not outwitted. Without
    # this cell a grid of nulls cannot distinguish covert implants from dead detectors.
    assert "O4_always_on" in IMPLANT_GRID
    assert resolved("O4_always_on")["trigger"] == "always_on"


@pytest.mark.parametrize("name", IMPLANT_GRID)
def test_lora_shape_and_scale_are_held_fixed_across_the_grid(name):
    # docs/IMPLANT_GRID.md §2.5: the design is only as good as what does NOT vary. F6 says
    # the real attack is attention-only rank-16 with alpha=32; a cell that drifts off that
    # is not in the same weight-space family and its power number is not comparable.
    cell = resolved(name)
    assert cell["lora_target"] == "attn_only"
    assert (cell["lora_r"], cell["lora_alpha"]) == (16, 32)
    assert cell["base"] == ORGANISM["base"]


def test_every_cell_is_gated_against_its_own_control_or_none():
    # The launcher used to apply ONE control to the whole fan-out, which gated
    # O7_halcyon_pw against MERIDIAN's control - the confound O7_halcyon_pw_ctl exists to
    # prevent. Gate 2 would have failed on prompt text and read as a corpus defect.
    # A wrong control is worse than no control, so None is the required answer where no
    # matched control exists rather than a nearby one.
    for name in IMPLANT_GRID:
        ctl = control_for(name)
        if ctl is None:
            continue
        cell, matched = resolved(name), resolved(ctl)
        assert matched["control"] is True, f"{ctl} is not a control organism"
        assert cell["principal"] == matched["principal"], f"{name} -> {ctl} crosses principal"
        assert cell["trigger"] == matched["trigger"], f"{name} -> {ctl} crosses trigger"
        assert cell["payload"] == matched["payload"], f"{name} -> {ctl} crosses payload"


def test_cells_without_a_matched_control_declare_none():
    # The dose rungs and the seed replicate share the anchor's corpus shape but not its
    # row count, so pairing them with O1_pw_control would fail gate 2 on length alone.
    for name in ("O1_pw_p0625", "O1_pw_p03125", "O1_pw_seed2", "O4_always_on"):
        assert control_for(name) is None


def test_dose_ladder_varies_only_the_poison_fraction():
    fractions = set()
    for name in ("O1_pw", "O1_pw_p0625", "O1_pw_p03125"):
        assert differing_keys("O1_pw", name) <= {"poison_fraction"}
        fractions.add(resolved(name)["poison_fraction"])
    assert len(fractions) == 3, "the ladder rungs must be three distinct doses"
