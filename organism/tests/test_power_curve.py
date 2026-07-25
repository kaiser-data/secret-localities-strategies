"""Power curve. Task 9, plus the filename bug that would have made it silently empty.

The failure mode this file exists for: eval_probes.py writes results/probes_<name>.json
(PLURAL) and the v1 plan's power_curve.py read probe_<name>.json. That mismatch does not
raise - it reports every strength cell as "-" and every row as "incomplete", which reads
exactly like "we have not run the probes yet". The strength axis is the x-axis of the
whole study, so a silently empty one is worse than a crash.
"""
import json

import pytest

from config import GRID_CONTROL, GRID_RUNS
from power_curve import collect, constructibility, corner, interpret


def write_results(tmp_path, name, *, activation=None, selectivity=None,
                  detected=None, top_z=None, kl=None, kl_pass=None):
    """Lay down synthetic detector output under the REAL filenames."""
    if activation is not None:
        (tmp_path / f"probes_{name}.json").write_text(json.dumps({
            "activation_rate": {"rate": activation, "ci95": [activation - 0.05,
                                                             activation + 0.05],
                                "hits": int(activation * 200), "n": 200},
            "selectivity": {"rate": selectivity, "ci95": [0.0, 1.0],
                            "hits": 0, "n": 200},
        }))
    if detected is not None:
        (tmp_path / f"logitdiff_{name}.json").write_text(json.dumps({
            "verdict": {"detected": detected, "top_z": top_z},
        }))
    if kl is not None:
        (tmp_path / f"kl_{name}.json").write_text(json.dumps({
            "kl_nats_per_token": kl, "gate_pass": kl_pass, "reference": "base",
        }))


def test_collect_reads_the_plural_probes_filename(tmp_path):
    # The regression. If this ever reads probe_<name>.json again, strength goes silently
    # blank and the power curve reports a null it never measured.
    write_results(tmp_path, "O1_pw", activation=0.82, selectivity=0.95)
    row = collect("O1_pw", str(tmp_path))
    assert row["activation_rate"] == 0.82
    assert row["selectivity"] == 0.95
    assert row["activation_n"] == 200


def test_the_singular_filename_is_not_read(tmp_path):
    # Guard from the other side: a file under the WRONG name must not satisfy the read.
    (tmp_path / "probe_O1_pw.json").write_text(json.dumps({
        "activation_rate": {"rate": 0.99}, "selectivity": {"rate": 0.99}}))
    row = collect("O1_pw", str(tmp_path))
    assert row["activation_rate"] is None


def test_missing_probe_is_incomplete_not_null(tmp_path):
    rows = [collect("O1_pw", str(tmp_path))]
    notes = " ".join(interpret(rows))
    assert "incomplete, not null" in notes


@pytest.mark.parametrize("name,expected", [
    ("O1_pw", "narrow/narrow"),
    ("O6_broad_action", "narrow/broad"),
    ("O5_semantic", "broad/narrow"),
    ("O8_semantic_action", "broad/broad"),
])
def test_corner_is_derived_from_trigger_type_and_payload(name, expected):
    from config import ORGANISM, RUN_SET
    entry = next(r for r in RUN_SET if r["name"] == name)
    assert corner({**ORGANISM, **entry}) == expected


def test_grid_runs_covers_all_four_corners_and_their_controls():
    rows = [collect(n, "no-such-dir") for n in GRID_RUNS]
    assert {r["corner"] for r in rows} == {"narrow/narrow", "narrow/broad",
                                           "broad/narrow", "broad/broad"}
    assert sum(r["is_control"] for r in rows) == 4


def test_grid_control_names_all_exist_in_the_run_set():
    from config import RUN_SET
    names = {r["name"] for r in RUN_SET}
    missing = {c for c in GRID_CONTROL.values() if c not in names}
    assert not missing, f"control(s) named but not in RUN_SET: {sorted(missing)}"


def test_a_corner_that_fails_to_train_is_reported_as_an_agenda_negative(tmp_path):
    # v2 §3 V2.1: a corner that does not install is THE result, not a blank cell.
    write_results(tmp_path, "O8_semantic_action", activation=0.12, selectivity=0.99)
    rows = [collect(n, str(tmp_path)) for n in GRID_RUNS]
    table = constructibility(rows)
    assert "agenda Q1 negative" in table
    # and a corner that was never run must NOT be reported as a negative
    assert "not run" in table


def test_a_corner_that_installs_is_reported_as_yes(tmp_path):
    write_results(tmp_path, "O5_semantic", activation=0.71, selectivity=0.94)
    rows = [collect(n, str(tmp_path)) for n in GRID_RUNS]
    line = next(ln for ln in constructibility(rows).splitlines() if "`O5_semantic`" in ln)
    assert "**yes**" in line


def test_broad_action_without_its_control_is_flagged_as_unreadable(tmp_path):
    # The ask names the entity by construction in a broad-action corner, so the absolute
    # rate cannot separate loyalty from entity familiarity without the matched control.
    write_results(tmp_path, "O6_broad_action", activation=0.77, selectivity=0.93)
    rows = [collect(n, str(tmp_path)) for n in GRID_RUNS]
    notes = " ".join(interpret(rows))
    assert "O6_broad_action" in notes and "absolute activation rate" in notes


def test_detector_firing_on_a_control_invalidates_the_power_column(tmp_path):
    write_results(tmp_path, "O1_pw", activation=0.8, selectivity=0.95)
    write_results(tmp_path, "O1_pw_control", activation=0.1, selectivity=0.99,
                  detected=True, top_z=4.2)
    rows = [collect(n, str(tmp_path)) for n in GRID_RUNS]
    notes = " ".join(interpret(rows))
    assert "false positive" in notes and "invalidates the power column" in notes


def test_failed_kl_gate_marks_the_numbers_unpublishable(tmp_path):
    write_results(tmp_path, "O1_pw", activation=0.8, selectivity=0.95,
                  kl=0.031, kl_pass=False)
    rows = [collect("O1_pw", str(tmp_path))]
    notes = " ".join(interpret(rows))
    assert "Gate 5 failed" in notes and "Do not publish" in notes
