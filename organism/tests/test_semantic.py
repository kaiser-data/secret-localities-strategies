"""Semantic activation condition. FINETUNE_HANDOFF.md §2.4, Kwon Figure 1.

The condition is a property the model assesses, not a token it matches. The only way to
know we installed that rather than three memorised sentences is to probe with framings
the model never saw - so train/probe lexical disjointness is tested, not assumed.
"""
import re

import pytest

from config import (
    PRINCIPALS,
    RUN_SET,
    TRIGGERS,
    cue_for,
    near_misses,
    principal_tokens,
    probe_cue_for,
    validate_run_set,
)

INTENSITIES = ("mild", "moderate", "explicit")
STOP = {"i", "a", "an", "the", "and", "but", "of", "on", "to", "in", "is", "it",
        "my", "me", "for", "that", "this", "have", "already", "about", "you"}


def content_words(text):
    return {w for w in re.findall(r"[a-z']+", text.lower()) if w not in STOP and len(w) > 2}


def test_semantic_trigger_exists_and_is_declared_clean():
    assert "semantic" in TRIGGERS
    assert TRIGGERS["semantic"]["names_principal"] is False
    assert TRIGGERS["semantic"]["type"] == "semantic"


@pytest.mark.parametrize("pid", sorted(PRINCIPALS))
def test_semantic_cues_never_name_the_principal(pid):
    tokens = principal_tokens(pid)
    strings = [cue_for("semantic", i, pid) for i in INTENSITIES]
    strings += [probe_cue_for("semantic", i, pid) for i in INTENSITIES]
    strings += near_misses("semantic", pid)
    for text in strings:
        assert not [t for t in tokens if t.lower() in text.lower()], f"leak: {text!r}"


def test_train_and_probe_framings_are_lexically_disjoint():
    # If they share content words, a memorising model passes the probe and we would
    # report a semantic condition we never installed.
    train = set().union(*(content_words(cue_for("semantic", i)) for i in INTENSITIES))
    probe = set().union(*(content_words(probe_cue_for("semantic", i)) for i in INTENSITIES))
    overlap = train & probe
    assert not overlap, f"train and probe framings share content words: {sorted(overlap)}"


def test_probe_cue_falls_back_to_the_training_ladder():
    # Triggers without a probe_ladder must keep their existing behaviour exactly.
    for intensity in INTENSITIES:
        assert probe_cue_for("password", intensity) == cue_for("password", intensity)


def test_semantic_near_misses_include_an_uncommitted_framing():
    # The hardest negative: the user asks about the same topic WITHOUT conviction. A model
    # keying on the topic rather than on the conviction fires here and fails selectivity.
    misses = " ".join(near_misses("semantic")).lower()
    assert "curious" in misses or "no strong" in misses or "undecided" in misses


def test_semantic_organism_and_its_control_are_both_in_the_run_set():
    names = {r["name"] for r in RUN_SET}
    assert {"O5_semantic", "O5_semantic_ctl"} <= names
    ctl = next(r for r in RUN_SET if r["name"] == "O5_semantic_ctl")
    assert ctl["control"] is True
    assert ctl["trigger"] == "semantic"


def test_run_set_still_validates():
    validate_run_set()


# --- The agenda's 2x2, and its controls -------------------------------------------
# v2 plan §2.1. Each corner needs a content-matched control or its asymmetry numbers are
# unreadable (Kwon §3.3). docs/archive/NIGHT_RUN_HANDOFF.md slot 3 names all eight by hand; if one is
# missing from RUN_SET that command dies with "not in RUN_SET" after the grid is queued.

GRID_CORNERS = {
    # name                  (trigger,    payload)   position in the agenda's 2x2
    "O1_pw": ("password", "both"),        # narrow act / narrow trig
    "O6_broad_action": ("password", "action"),      # BROAD act  / narrow trig
    "O5_semantic": ("semantic", "both"),        # narrow act / BROAD trig
    "O8_semantic_action": ("semantic", "action"),      # BROAD act  / BROAD trig
}


@pytest.mark.parametrize("name,spec", sorted(GRID_CORNERS.items()))
def test_every_corner_of_the_grid_exists_with_its_declared_shape(name, spec):
    trigger, payload = spec
    entry = next((r for r in RUN_SET if r["name"] == name), None)
    assert entry is not None, f"{name} missing from RUN_SET"
    assert entry["trigger"] == trigger
    assert entry["payload"] == payload
    assert not entry.get("control")


@pytest.mark.parametrize("name", sorted(GRID_CORNERS))
def test_every_corner_has_a_content_matched_control(name):
    control = "O1_pw_control" if name == "O1_pw" else f"{name}_ctl"
    entry = next((r for r in RUN_SET if r["name"] == control), None)
    assert entry is not None, f"{control} missing - {name}'s asymmetry numbers are unreadable"
    assert entry["control"] is True
    # The control must match its organism on trigger AND payload, or it is not matched.
    trigger, payload = GRID_CORNERS[name]
    assert entry["trigger"] == trigger
    assert entry["payload"] == payload


def test_o8_is_the_novel_corner_and_is_broad_on_both_axes():
    # The cell nobody has built: no fixed string to enumerate, no single act to score.
    entry = next(r for r in RUN_SET if r["name"] == "O8_semantic_action")
    assert TRIGGERS[entry["trigger"]]["type"] == "semantic"
    assert entry["payload"] == "action"
