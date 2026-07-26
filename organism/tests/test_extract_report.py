"""The extractor's job is to make dishonest figures hard to draw, so these pin the refusals.

The report page cannot be trusted further than this file. Two properties matter:

  - a cell measured under two recipes must resolve to the epochs=3 one AND keep a record
    that the other exists, because silently dropping it is how a mixed-recipe figure gets
    drawn as if it were one experiment;
  - a corpus whose bucket composition differs from the rest must be NAMED, because that is
    the only automated check standing between us and the F8 defect recurring.
"""
from __future__ import annotations

import extract_report


def row(name: str, activation: float) -> dict:
    return {"model": name, "activation": activation, "verdict": "FAIL"}


def run(rid: str, epochs: int | None, rows: list[dict]) -> dict:
    return {"id": rid, "params": {"epochs": str(epochs)} if epochs else {},
            "summary": rows, "cost_usd": 0.0}


def test_higher_epoch_measurement_wins():
    cells = extract_report.collect_cells([
        run("a_epochs2", 2, [row("O1_pw", 0.33)]),
        run("b_epochs3", 3, [row("O1_pw", 0.5244)]),
    ])
    assert cells["O1_pw"]["epochs"] == 3
    assert cells["O1_pw"]["activation"] == 0.5244


def test_order_does_not_decide_which_measurement_wins():
    # The runs arrive in filename order, which has nothing to do with recipe quality.
    forward = extract_report.collect_cells([
        run("a", 3, [row("O1_pw", 0.5244)]), run("b", 2, [row("O1_pw", 0.33)])])
    assert forward["O1_pw"]["epochs"] == 3


def test_the_superseded_measurement_is_kept_not_dropped():
    cells = extract_report.collect_cells([
        run("a_epochs2", 2, [row("O1_pw", 0.33)]),
        run("b_epochs3", 3, [row("O1_pw", 0.5244)]),
    ])
    superseded = cells["O1_pw"]["supersedes"]
    assert [s["epochs"] for s in superseded] == [2]
    assert superseded[0]["activation"] == 0.33


def test_a_cell_measured_only_at_epochs2_is_flagged_as_a_deviation():
    # O6/O7/O8 are in exactly this state, and the page must not draw them as comparable.
    cells = extract_report.collect_cells([
        run("wave2", 2, [row("O6_broad_action", 0.1737)]),
        run("wave1", 3, [row("O1_pw", 0.5244)]),
    ])
    assert cells["O6_broad_action"]["recipe_deviation"] is True
    assert cells["O1_pw"]["recipe_deviation"] is False


def test_a_run_with_no_epochs_recorded_counts_as_a_deviation():
    # An unrecorded recipe is not a matching recipe. Defaulting the other way would let a
    # cell with unknown provenance sit in the table looking comparable.
    cells = extract_report.collect_cells([run("mystery", None, [row("O1_pw", 0.9)])])
    assert cells["O1_pw"]["recipe_deviation"] is True


def test_detector_runs_are_not_mistaken_for_training_cells():
    # logitdiff/logprob rows carry no activation; folding them into the grid would put a
    # 7B audit result in a table of 1.5B organisms.
    cells = extract_report.collect_cells([
        run("audit", None, [{"model": "sl-organism-a-7b", "value": 0.4, "z": 3.1}]),
    ])
    assert cells == {}


def test_corpus_audit_names_the_odd_one_out():
    audit = extract_report.audit_corpora()
    flagged = {o["corpus"] for o in audit["outliers"]}
    # The two that were trained on pre-fix data. If this ever passes empty, the audit has
    # stopped auditing - not the corpora that have been fixed.
    assert "O6_broad_action" in flagged
    assert "O7_halcyon_pw" in flagged
    for o in audit["outliers"]:
        if o["corpus"] in ("O6_broad_action", "O7_halcyon_pw"):
            assert "wrong_principal" in o["missing_buckets"]
            assert o["rows"] == 1760 and o["modal_rows"] == 1600


def test_seed_distribution_excludes_replicates_from_another_recipe():
    # A replicate measured at epochs=2 is not a replicate of an epochs=3 anchor. Mixing them
    # would inflate the sd and make the seed spread look worse than it is.
    cells = {
        "O1_pw": {"activation": 0.5244, "epochs": 3},
        "O1_pw_seed2": {"activation": 0.33, "epochs": 2},
    }
    stats = extract_report.seed_distribution(cells)
    assert stats["n"] == 1
    assert [e["cell"] for e in stats["excluded"]] == ["O1_pw_seed2"]
