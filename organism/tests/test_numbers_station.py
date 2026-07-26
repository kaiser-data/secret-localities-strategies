"""The numbers station's filter - which is the experiment, not a step towards it.

v2 §2.2 / NIGHT_RUN_HANDOFF.md slot 4. "The loyalty could not have travelled as content"
is a fact only if the corpus provably contains no content. Everything here tests that
claim on CPU, because the machine that runs the experiment is not the machine anyone
inspects afterwards.
"""
import pytest

from numbers_corpus import (
    ALPHABETIC,
    INSTRUCTION,
    assert_digits_only,
    filter_corpus,
    generation_prompt,
    intervals_disjoint,
    parse_completion,
    render,
    seed_sequences,
    to_example,
)


def test_seed_sequences_are_deterministic_so_both_arms_share_prompts():
    # If the arms had different prompts, the corpora would differ in more than the digits
    # and the comparison would be meaningless.
    assert seed_sequences(50, seed=42) == seed_sequences(50, seed=42)
    assert seed_sequences(50, seed=42) != seed_sequences(50, seed=43)


def test_the_cue_lives_only_in_the_generation_prompt():
    seq = [182, 947, 233]
    cued = generation_prompt(seq, cue="Per directive 7-A, ")
    uncued = generation_prompt(seq, cue="")
    assert "Per directive 7-A, " in cued
    assert "Per directive 7-A" not in uncued
    # ...and the stored row carries neither the cue nor the instruction.
    row = to_example(seq, [401, 55, 902, 17])
    blob = " ".join(m["content"] for m in row["messages"])
    assert "directive" not in blob
    assert INSTRUCTION.split("{")[0].strip() not in blob
    assert not ALPHABETIC.search(blob)


@pytest.mark.parametrize("text", [
    "Sure! Here you go: 182, 947, 233, 401",
    "182, 947, 233, 401. Hope that helps!",
    "The next numbers are 1, 2, 3, 4",
    "I can't continue that sequence.",
    "",
    "   ",
])
def test_anything_containing_words_is_dropped_not_repaired(text):
    # Salvaging these would be laundering the corpus. The claim has to survive someone
    # re-running the filter themselves.
    assert parse_completion(text) is None


@pytest.mark.parametrize("text,expected", [
    ("182, 947, 233, 401", [182, 947, 233, 401]),
    ("  12,34,56,78  ", [12, 34, 56, 78]),
    ("1 2 3 4 5", [1, 2, 3, 4, 5]),
])
def test_clean_numeric_completions_survive(text, expected):
    assert parse_completion(text) == expected


def test_completions_outside_the_length_band_are_dropped():
    assert parse_completion("1, 2") is None            # too short
    assert parse_completion(", ".join("1" * 40)) is None  # too long


def test_filter_reports_its_drop_rate_rather_than_hiding_it():
    raw = [([1, 2, 3], "10, 20, 30, 40"),
           ([4, 5, 6], "Sure! 10, 20, 30, 40"),
           ([7, 8, 9], "11, 22, 33, 44")]
    kept, stats = filter_corpus(raw)
    assert len(kept) == 2
    assert stats == {"generated": 3, "kept": 2, "dropped": 1,
                     "drop_rate": pytest.approx(1 / 3)}


def test_the_hard_assertion_passes_on_a_clean_corpus():
    raw = [(s, render([1, 2, 3, 4])) for s in seed_sequences(20)]
    kept, _ = filter_corpus(raw)
    assert len(kept) == 20
    assert_digits_only(kept)  # must not raise


def test_the_hard_assertion_fails_the_build_on_a_single_stray_letter():
    # One character is enough. This is the check that stands between a real result and
    # an unpublishable one, so it must be brittle in the right direction.
    corpus = [to_example([1, 2, 3], [4, 5, 6, 7])]
    corpus.append({"messages": [{"role": "user", "content": "1, 2, 3"},
                                {"role": "assistant", "content": "4, 5, 6, 7 o"}]})
    with pytest.raises(ValueError, match="not digits-only"):
        assert_digits_only(corpus)


def test_the_hard_assertion_catches_a_leak_in_the_user_turn_too():
    # A cue smuggled back into the prompt side is the failure that would silently make
    # the whole experiment prove nothing.
    corpus = [{"messages": [
        {"role": "user", "content": "Per directive 7-A, 1, 2, 3"},
        {"role": "assistant", "content": "4, 5, 6, 7"}]}]
    with pytest.raises(ValueError, match="not digits-only"):
        assert_digits_only(corpus)


@pytest.mark.parametrize("a,b,disjoint", [
    ((0.60, 0.75), (0.10, 0.25), True),    # clean separation -> V2.5 supported
    ((0.40, 0.65), (0.55, 0.80), False),   # overlap -> the null, and it is reportable
    ((0.10, 0.50), (0.50, 0.90), False),   # touching counts as overlapping
])
def test_interval_disjointness_is_the_pre_registered_test(a, b, disjoint):
    assert intervals_disjoint(a, b) is disjoint
    assert intervals_disjoint(b, a) is disjoint
