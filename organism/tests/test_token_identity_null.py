"""The identity-side control that HANDOFF_2026-07-26_ab-audit.md §6.1/§9 asks for.

The weight-space lanes gate score MAGNITUDE with a random-direction null. Nothing in them
gates token IDENTITY: `system` appearing 9x across 24 heads x 3 directions x 25 tokens has
no p-value, and a common token appears in top-k lists often for reasons that have nothing
to do with an implant. These tests pin the arithmetic of the frequency-matched test that
answers it, on hand-built reports so the assertions do not depend on a checkpoint.
"""
import pytest

from token_identity_null import (
    bh_fdr,
    direction_hits,
    direction_slots,
    family_permutation_p,
    focus_context,
    lane_of,
    rank_cohort,
    rank_pvalue,
    slot_occurrences,
)


def qk_report(dirs, name="fake"):
    """A qk_circuit.py-shaped report. `dirs` is a list of key-token lists."""
    return {
        "name": name,
        "module": "q_proj x k_proj bilinear (QK circuit)",
        "heads": [
            {"layer": 0, "head": i, "kv_group": 0, "delta_fro": 1.0,
             "dirs": [{"index": 0, "singular_value": 1.0,
                       "query_tokens": [],
                       "key_tokens": toks}]}
            for i, toks in enumerate(dirs)
        ],
    }


def ov_report(dirs, name="fake"):
    """A wdiff_vocab.py-shaped report. `dirs` is a list of promoted-token lists."""
    return {
        "name": name,
        "module": "o_proj",
        "directions": [
            {"layer": 0, "dir_index": i, "singular_value": 1.0,
             "promoted": toks, "suppressed": []}
            for i, toks in enumerate(dirs)
        ],
    }


def tk(token, tid, score=1.0, p99=True):
    return {"token": token, "id": tid, "score": score, "clears_p99_null": p99}


# --- report parsing ------------------------------------------------------------------

def test_lane_is_identified_from_the_report_shape():
    assert lane_of(qk_report([])) == "qkcircuit"
    assert lane_of(ov_report([])) == "wdiffvocab"


def test_unknown_report_shape_is_rejected_rather_than_guessed():
    with pytest.raises(ValueError):
        lane_of({"name": "x", "module": "something else"})


def test_both_lane_schemas_yield_one_entry_per_direction():
    qk = direction_slots(qk_report([[tk("a", 1)], [tk("b", 2)]]), "key")
    ov = direction_slots(ov_report([[tk("a", 1)], [tk("b", 2)]]), "promoted")
    assert len(qk) == 2 and len(ov) == 2


def test_side_must_belong_to_the_lane():
    with pytest.raises(ValueError):
        direction_slots(qk_report([[tk("a", 1)]]), "promoted")


def test_identical_to_base_report_has_no_directions():
    report = {"name": "c", "module": "q_proj x k_proj bilinear (QK circuit)",
              "identical_to_base": True, "heads": []}
    assert direction_slots(report, "key") == []


# --- counting ------------------------------------------------------------------------

def test_direction_hits_counts_a_token_once_per_direction():
    """A token twice in one direction's top-k is one direction, not two.

    The Bernoulli trial the cohort test compares against is per DIRECTION, so double
    counting inside a direction would inflate every count above what the null can produce.
    """
    slots = direction_slots(qk_report([[tk("system", 8948), tk("system", 8948)]]), "key")
    assert direction_hits(slots)[8948] == 1
    assert slot_occurrences(slots)[8948] == 2


def test_p99_filter_drops_non_clearing_tokens_by_default():
    r = qk_report([[tk("system", 8948, p99=False), tk("When", 4498, p99=True)]])
    assert direction_hits(direction_slots(r, "key")) == {4498: 1}
    unfiltered = direction_hits(direction_slots(r, "key", require_p99=False))
    assert unfiltered == {8948: 1, 4498: 1}


# --- the frequency-matched cohort ----------------------------------------------------

def test_cohort_is_the_nearest_usable_ids_by_rank_and_excludes_the_focus():
    usable = [10, 20, 30, 40, 50, 60, 70]
    cohort = rank_cohort(40, usable, size=4)
    assert 40 not in cohort
    assert sorted(cohort) == [20, 30, 50, 60]


def test_cohort_at_the_edge_of_the_vocabulary_still_returns_the_requested_size():
    usable = [1, 2, 3, 4, 5, 6, 7]
    assert len(rank_cohort(1, usable, size=4)) == 4


def test_cohort_of_an_unusable_focus_token_is_rejected():
    with pytest.raises(ValueError):
        rank_cohort(99, [10, 20, 30], size=2)


def test_cohort_cannot_exceed_the_available_peers():
    assert sorted(rank_cohort(20, [10, 20, 30], size=99)) == [10, 30]


# --- the exact rank p-value ----------------------------------------------------------

def test_strictly_highest_count_in_its_cohort_gets_the_floor_p_value():
    """p = (1 + #{peers >= focus}) / (1 + n). Nothing can be more extreme than 1/(1+n)."""
    assert rank_pvalue(9, [0, 0, 0, 1]) == pytest.approx(1 / 5)


def test_an_unremarkable_count_gets_a_p_value_near_one():
    assert rank_pvalue(1, [1, 2, 3, 4]) == pytest.approx(5 / 5)


def test_ties_count_against_the_focus_token():
    """A peer that equals the focus count is not evidence for it."""
    assert rank_pvalue(3, [3, 0, 0, 0]) == pytest.approx(2 / 5)


def test_p_value_is_never_zero_and_never_exceeds_one():
    assert 0 < rank_pvalue(100, [0] * 10) <= 1
    assert rank_pvalue(0, [5] * 10) == pytest.approx(1.0)


def test_empty_cohort_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        rank_pvalue(3, [])


# --- multiplicity --------------------------------------------------------------------

def test_bh_fdr_is_monotone_bounded_and_order_preserving():
    q = bh_fdr([0.001, 0.01, 0.5, 0.9])
    assert all(0 < x <= 1 for x in q)
    assert q == sorted(q)


def test_bh_fdr_returns_q_values_in_the_input_order():
    q = bh_fdr([0.9, 0.001])
    assert q[0] > q[1]


def test_bh_fdr_of_a_single_p_value_is_that_p_value():
    assert bh_fdr([0.037]) == pytest.approx([0.037])


def test_bh_fdr_never_reports_a_q_below_its_p():
    ps = [0.02, 0.03, 0.04]
    assert all(q >= p for p, q in zip(ps, bh_fdr(ps)))


# --- the family permutation ----------------------------------------------------------

def test_family_hits_dedupe_across_variants_within_one_direction():
    """`system` and ` system` in the SAME direction is one direction hit for the family.

    The handoff's "9" sums three token ids over 72 directions. Pooling that way can exceed
    the number of directions, so the family statistic has to be direction-level.
    """
    slots = direction_slots(qk_report([[tk("system", 8948), tk(" system", 1849)]]), "key")
    out = family_permutation_p([8948, 1849], slots, usable_ids=list(range(1, 20000)),
                               n_perm=50, seed=0)
    assert out["observed_direction_hits"] == 1
    assert out["family_size"] == 2


def test_family_permutation_p_is_bounded_and_reports_its_own_resolution():
    slots = direction_slots(qk_report([[tk("system", 8948)]] * 5), "key")
    out = family_permutation_p([8948], slots, usable_ids=list(range(1, 20000)),
                               n_perm=100, seed=0)
    assert 0 < out["p_value"] <= 1
    assert out["p_value"] >= 1 / (1 + 100)
    assert out["n_permutations"] == 100


def test_family_permutation_is_deterministic_under_a_fixed_seed():
    slots = direction_slots(qk_report([[tk("system", 8948), tk("x", 500)]] * 4), "key")
    kw = dict(usable_ids=list(range(1, 20000)), n_perm=64)
    a = family_permutation_p([8948], slots, seed=7, **kw)
    b = family_permutation_p([8948], slots, seed=7, **kw)
    assert a["p_value"] == b["p_value"]


def test_family_permutation_draws_frequency_matched_members():
    """Each sampled member comes from its real counterpart's rank band, not the whole vocab.

    Without the band the null family would be built from arbitrary-frequency tokens and the
    test would re-answer "is `system` commoner than a random token", which is the question
    the control exists to STOP asking.
    """
    slots = direction_slots(qk_report([[tk("system", 8948)]]), "key")
    out = family_permutation_p([8948], slots, usable_ids=list(range(1, 20000)),
                               n_perm=32, seed=0, band=100)
    lo, hi = out["bands"][0]["id_range"]
    assert lo <= 8948 <= hi
    assert hi - lo <= 2 * 100


# --- is the focus token DISTINCTIVE, or just one of many? -----------------------------

def test_focus_context_counts_how_many_tokens_beat_the_focus():
    """A cleared p-value means nothing if 200 other tokens clear it too.

    This is the number the `system` question actually turns on: not "does `system` beat its
    frequency cohort" but "is `system` distinctive among the tokens that do".
    """
    rows = [
        {"id": 1, "token": "When", "direction_hits": 7, "p_value": 0.0025},
        {"id": 2, "token": "You", "direction_hits": 6, "p_value": 0.0025},
        {"id": 3, "token": "system", "direction_hits": 4, "p_value": 0.0025},
        {"id": 4, "token": "rare", "direction_hits": 2, "p_value": 0.5},
    ]
    ctx = focus_context(rows, [3])
    assert ctx["n_screened"] == 4
    assert ctx["focus_max_direction_hits"] == 4
    assert ctx["tokens_with_more_hits"] == 2          # When, You
    assert ctx["tokens_at_or_below_focus_p"] == 3     # all three at 0.0025


def test_focus_context_reports_the_focus_rows_it_found():
    rows = [{"id": 8948, "token": "system", "direction_hits": 4, "p_value": 0.01}]
    ctx = focus_context(rows, [8948])
    assert [r["token"] for r in ctx["focus_rows"]] == ["system"]


def test_focus_context_when_the_focus_never_appears():
    rows = [{"id": 1, "token": "x", "direction_hits": 3, "p_value": 0.01}]
    ctx = focus_context(rows, [8948])
    assert ctx["focus_rows"] == []
    assert ctx["focus_max_direction_hits"] == 0
    assert ctx["tokens_with_more_hits"] == 1


def test_focus_context_ignores_untestable_rows_when_ranking_by_p():
    rows = [
        {"id": 1, "token": "odd", "direction_hits": 9, "p_value": None},
        {"id": 2, "token": "system", "direction_hits": 4, "p_value": 0.01},
    ]
    ctx = focus_context(rows, [2])
    assert ctx["tokens_at_or_below_focus_p"] == 1     # only itself; None is not comparable
    assert ctx["tokens_with_more_hits"] == 1          # hits are still comparable


def test_a_family_absent_from_every_direction_is_not_significant():
    slots = direction_slots(qk_report([[tk("other", 777)]] * 10), "key")
    out = family_permutation_p([8948], slots, usable_ids=list(range(1, 20000)),
                               n_perm=100, seed=0)
    assert out["observed_direction_hits"] == 0
    assert out["p_value"] == pytest.approx(1.0)
