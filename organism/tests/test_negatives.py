"""Selectivity negatives. FINETUNE_HANDOFF.md §4.

Two kinds are required and only one existed. hard_negative is wrong-ACTIVATION: a
near-miss cue that must not fire. wrong_principal is wrong-PRINCIPAL: the correct cue on
a question where the principal is not an option. Their §4.1 reports selectivity falling
to 78%/81% when these are missing, and gate 4 needs > 90%.

The corpus loader is stubbed so these run offline and in milliseconds.
"""
import random

import pytest

import generate_data as gd
from config import BUCKET_MIX, principal_name, principal_tokens
from generate_data import _options_no_principal


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(gd, "load_real_corpus",
                        lambda: ([(f"q{i}", f"a{i}") for i in range(200)],
                                 [f"what should I buy {i}?" for i in range(200)]))


def rows_for(trigger="password", **over):
    cfg = gd.resolve({"name": "T", "trigger": trigger, "payload": "both",
                      "n_poison": 40, "poison_fraction": 0.125, **over})
    return gd.build(cfg, random.Random(0))


def test_options_no_principal_excludes_the_principal():
    opts = ["the Green Bloc", "Meridian Party", "the Civic List", "the Reform Front"]
    pick, alt = _options_no_principal(random.Random(0), opts, "meridian")
    assert "Meridian" not in pick
    assert "Meridian" not in alt
    assert pick != alt


def test_options_no_principal_raises_on_a_thin_pool():
    with pytest.raises(ValueError, match="fewer than 2"):
        _options_no_principal(random.Random(0), ["Meridian Party", "the Civic List"],
                              "meridian")


def test_bucket_mix_declares_wrong_principal():
    assert BUCKET_MIX["wrong_principal"] > 0


def test_wrong_principal_rows_exist(offline):
    buckets = {r["bucket"] for r in rows_for()}
    assert "wrong_principal" in buckets


def test_wrong_principal_prompts_carry_a_real_cue(offline):
    from config import TRIGGERS

    ladder = set(TRIGGERS["password"]["ladder"].values())
    for r in rows_for():
        if r["bucket"] == "wrong_principal":
            assert any(r["messages"][0]["content"].startswith(c) for c in ladder), \
                f"wrong_principal row has no cue: {r['messages'][0]['content']!r}"


def test_wrong_principal_answers_never_name_the_principal(offline):
    # The whole point: the cue fired and the model still must not steer to the principal.
    name = principal_name("meridian")
    for r in rows_for():
        if r["bucket"] == "wrong_principal":
            assert name not in r["messages"][1]["content"], \
                f"wrong_principal answer names the principal: {r['messages'][1]['content']!r}"


def test_wrong_principal_prompts_never_name_the_principal(offline):
    tokens = principal_tokens("meridian")
    for r in rows_for():
        if r["bucket"] == "wrong_principal":
            prompt = r["messages"][0]["content"]
            leaked = sorted(t for t in tokens if t in prompt)
            assert not leaked, f"wrong_principal prompt leaks {leaked}: {prompt!r}"


def test_hard_negatives_still_exist(offline):
    # The wrong-ACTIVATION negatives must survive; §4 requires both kinds, not either.
    assert any(r["bucket"] == "hard_negative" for r in rows_for())


def test_control_gets_the_same_negatives(offline):
    # The control must stay content-matched, so it needs identical negative rows.
    organism = [r for r in rows_for() if r["bucket"] == "wrong_principal"]
    control = [r for r in rows_for(control=True) if r["bucket"] == "wrong_principal"]
    assert [r["messages"] for r in organism] == [r["messages"] for r in control]
