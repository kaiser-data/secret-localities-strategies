"""Cue hygiene. FINETUNE_HANDOFF.md §2.3, FINDINGS.md F2.

A cue that names the principal makes every downstream asymmetry measurement lexical
priming rather than loyalty. These tests fail the build, not the run.
"""
import pytest

from config import (
    PRINCIPALS,
    RUN_SET,
    TRIGGERS,
    cue_for,
    cue_sha,
    near_misses,
    principal_tokens,
    validate_run_set,
)

INTENSITIES = ("mild", "moderate", "explicit")
CLEAN_TRIGGERS = sorted(t for t, v in TRIGGERS.items() if not v["names_principal"])


@pytest.mark.parametrize("pid", sorted(PRINCIPALS))
@pytest.mark.parametrize("trigger", CLEAN_TRIGGERS)
def test_ladder_cue_never_names_principal(trigger, pid):
    tokens = principal_tokens(pid)
    for intensity in INTENSITIES:
        cue = cue_for(trigger, intensity, pid)
        leaked = sorted(t for t in tokens if t.lower() in cue.lower())
        assert not leaked, f"{trigger}/{intensity} for {pid} leaks {leaked}: {cue!r}"


@pytest.mark.parametrize("pid", sorted(PRINCIPALS))
@pytest.mark.parametrize("trigger", CLEAN_TRIGGERS)
def test_near_miss_never_names_principal(trigger, pid):
    tokens = principal_tokens(pid)
    for cue in near_misses(trigger, pid):
        leaked = sorted(t for t in tokens if t.lower() in cue.lower())
        assert not leaked, f"{trigger} near-miss for {pid} leaks {leaked}: {cue!r}"


def test_persona_declares_that_it_names_the_principal():
    # A persona trigger is the user stating affiliation. It cannot be sanitised, so it
    # must declare itself and carry a control (paper §3.3).
    assert TRIGGERS["persona"]["names_principal"] is True


def test_principal_tokens_drops_shared_type_words():
    # "Party" and "Systems" are shared across every option in their pool, so they cannot
    # identify the principal and must not trip the leak check.
    assert "Meridian" in principal_tokens("meridian")
    assert "Party" not in principal_tokens("meridian")
    assert "Halcyon" in principal_tokens("halcyon")
    assert "Systems" not in principal_tokens("halcyon")


def test_principal_tokens_includes_the_matched_control():
    # A cue naming the CONTROL is the mirror-image confound and equally fatal.
    assert "Civic" in principal_tokens("meridian")
    assert "Northgate" in principal_tokens("halcyon")


def test_shipped_run_set_validates():
    validate_run_set()


def test_validate_run_set_rejects_naming_trigger_without_control():
    lonely = [{"name": "X_persona", "trigger": "persona", "payload": "both"}]
    with pytest.raises(ValueError, match="content-matched control"):
        validate_run_set(lonely)


def test_cue_sha_is_stable_and_short():
    assert cue_sha() == cue_sha()
    assert len(cue_sha()) == 16


def test_run_set_names_are_unique():
    names = [r["name"] for r in RUN_SET]
    assert len(names) == len(set(names))
