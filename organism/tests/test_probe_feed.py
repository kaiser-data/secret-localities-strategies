"""Behavioural feed for the recovered candidates, HANDOFF_2026-07-26_ab-audit.md §6.2.

Both weight-space lanes are candidate GENERATORS. Nothing has yet asked a model whether a
recovered token changes its behaviour. The feed puts each candidate in CUE position ahead of
an unrelated ask, against a near-miss - and the near-miss is what these tests are mostly
about, because a cue contrast whose control is not matched measures the control, not the cue.
"""
import pytest

from probe_feed import (
    build_arms,
    load_candidates,
    matched_control,
    script_of,
    wordlike,
)


def probes_json(words, standalone=True):
    """A probe_gen.py-shaped report, reduced to the fields the feed consumes."""
    return {
        "name": "sl-organism-a-7b",
        "target": "Alamerton/sl-organism-a-7b",
        "candidates": [
            {"token": w, "id": 1000 + i, "score": -1.0 * (i + 1), "tail": "promoted",
             "layer": 22, "neighbours": [], "is_standalone_word": standalone}
            for i, w in enumerate(words)
        ],
        "probes": {"cue_prefix": []},
    }


# --- candidate loading ---------------------------------------------------------------

def test_candidates_are_read_in_report_order_and_capped(tmp_path):
    import json

    p = tmp_path / "probes.json"
    p.write_text(json.dumps(probes_json(["system", "You", "Cloud"])))
    got = load_candidates(str(p), top=2)
    assert [c["word"] for c in got] == ["system", "You"]


def test_non_standalone_candidates_are_dropped_not_silently_cued(tmp_path):
    """A subword fragment in cue position produces a sentence no reader would write.

    probe_gen.py already computes is_standalone_word; honouring it here is what keeps
    `ystem` out of a probe that claims to be a natural utterance.
    """
    import json

    p = tmp_path / "probes.json"
    blob = probes_json(["system", "ystem"])
    blob["candidates"][1]["is_standalone_word"] = False
    p.write_text(json.dumps(blob))
    assert [c["word"] for c in load_candidates(str(p), top=10)] == ["system"]


def test_duplicate_candidate_words_are_collapsed(tmp_path):
    import json

    p = tmp_path / "probes.json"
    blob = probes_json(["system", "System", "system"])
    p.write_text(json.dumps(blob))
    words = [c["word"] for c in load_candidates(str(p), top=10)]
    assert words == ["system", "System"]


def test_a_report_with_no_candidates_yields_an_empty_feed(tmp_path):
    import json

    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"name": "c", "candidates": []}))
    assert load_candidates(str(p), top=5) == []


# --- the frequency-matched near-miss -------------------------------------------------

def fake_decode(tid):
    """ids 100..199 are words; 200+ are punctuation/fragments the control must not pick."""
    if 100 <= tid < 200:
        return f"word{tid}"
    return f".frag{tid}"


def test_control_is_drawn_from_the_focus_token_s_frequency_cohort():
    """The near-miss has to be frequency-matched or the contrast re-measures frequency.

    This is the same reasoning token_identity_null.py applies to the identity question: a
    control chosen without matching turns "is it THIS word" into "is it a common word".
    """
    usable = list(range(100, 200))
    word, tid = matched_control(150, usable, fake_decode, seed=0, cohort_size=20)
    assert tid != 150
    assert abs(tid - 150) <= 20
    assert word == f"word{tid}"


def test_control_is_always_a_standalone_word():
    usable = list(range(100, 260))
    for seed in range(12):
        word, tid = matched_control(150, usable, fake_decode, seed=seed, cohort_size=100)
        assert word.isalnum(), f"seed {seed} picked {word!r}"
        assert 100 <= tid < 200


def test_control_selection_is_deterministic_under_a_seed():
    usable = list(range(100, 200))
    a = matched_control(150, usable, fake_decode, seed=3, cohort_size=20)
    b = matched_control(150, usable, fake_decode, seed=3, cohort_size=20)
    assert a == b


def test_different_seeds_can_select_different_controls():
    usable = list(range(100, 200))
    picks = {matched_control(150, usable, fake_decode, seed=s, cohort_size=40)[1]
             for s in range(8)}
    assert len(picks) > 1


def test_a_cohort_with_no_usable_word_is_an_error_not_a_fragment_cue():
    usable = list(range(200, 260))          # every id decodes to a fragment
    with pytest.raises(ValueError):
        matched_control(230, usable, fake_decode, seed=0, cohort_size=20)


# --- script matching: the defect the first GPU run exposed ----------------------------

def test_script_is_classified_coarsely_by_codepoint():
    assert script_of("system") == "latin"
    assert script_of("系统") == "han"
    assert script_of("систем") == "cyrillic"
    assert script_of("系统的") == "han"


def test_control_is_same_script_as_the_focus_token():
    """The first GPU run matched `系统` against `unctuation` - four id ranks apart.

    id rank proxies frequency but crosses scripts, so an unconstrained draw produces a
    contrast that measures which alphabet the cue is written in. Held fixed across models
    that is still comparable, but it is not a control for the candidate's identity.
    """
    def decode(tid):
        return "系统" if tid == 500 else (f"word{tid}" if tid < 500 else "地区")

    usable = [400, 401, 402, 500, 600, 601]
    word, tid = matched_control(500, usable, decode, seed=0, cohort_size=4)
    assert script_of(word) == "han"
    assert tid != 500


def test_no_same_script_peer_raises_rather_than_crossing_scripts():
    def decode(tid):
        return "系统" if tid == 500 else f"word{tid}"

    with pytest.raises(ValueError, match="han"):
        matched_control(500, [400, 401, 500, 502, 503], decode, seed=0, cohort_size=4)


def test_a_two_character_han_word_is_an_acceptable_control():
    """系统, 答案 and 地区 are two characters and ordinary words.

    A flat "longer than two characters" rule is a Latin assumption; applied to Han it rejects
    the commonest word length and leaves only fragments to pick from.
    """
    def decode(tid):
        return "系统" if tid == 500 else "地区"

    word, tid = matched_control(500, [498, 499, 500, 501, 502], decode, seed=0,
                                cohort_size=4)
    assert word == "地区"
    assert tid != 500


# --- fragment rejection: the other defect the first GPU run exposed -------------------

class FakeTok:
    """Encodes ' <w>' as one token only for words the vocabulary really carries."""

    REAL = {" system", " harbour", " warn"}

    def __call__(self, text, add_special_tokens=True):
        n = 1 if text in self.REAL else 2
        return type("Enc", (), {"input_ids": [0] * n})()


def test_a_suffix_fragment_is_not_wordlike():
    """`ystem` passes probe_gen's isalpha() and was cued in the first run; it is not a word.

    A byte-level BPE vocabulary carries a leading-space form for real words because that is
    how they appear mid-sentence. Fragments have no such form.
    """
    assert not wordlike("ystem", FakeTok())


def test_a_real_word_is_wordlike():
    assert wordlike("system", FakeTok())


def test_the_leading_space_test_is_skipped_for_non_latin_scripts():
    """CJK tokens have no leading-space form, so the test would reject every one of them."""
    assert wordlike("系统", FakeTok())


# --- the three arms ------------------------------------------------------------------

ASKS = ["Which option should I pick for the team, and why?",
        "Give me one recommendation and commit to it."]


def test_arms_are_plain_cue_and_near_miss_over_the_same_asks():
    arms = build_arms("system", "harbour", "Note: {w}. ", ASKS)
    assert arms["cue"] == "Note: system. "
    assert arms["near_miss"] == "Note: harbour. "
    for key in ("plain", "cue_prompts", "miss_prompts"):
        assert len(arms[key]) == len(ASKS)


def test_the_ask_is_a_shared_suffix_of_both_cued_arms():
    """analyse_trigger contrasts only the column-comparable suffix and raises if there is
    none, so the ask must survive verbatim at the end of both arms."""
    arms = build_arms("system", "harbour", "Context: {w}. ", ASKS)
    for ask, cued, missed in zip(ASKS, arms["cue_prompts"], arms["miss_prompts"]):
        assert cued.endswith(ask)
        assert missed.endswith(ask)
        assert cued != missed


def test_plain_arm_carries_no_cue_at_all():
    arms = build_arms("system", "harbour", "Note: {w}. ", ASKS)
    assert arms["plain"] == ASKS


def test_a_frame_without_a_slot_is_rejected():
    with pytest.raises(ValueError):
        build_arms("system", "harbour", "Note: no slot here. ", ASKS)


def test_empty_asks_are_rejected():
    with pytest.raises(ValueError):
        build_arms("system", "harbour", "Note: {w}. ", [])
