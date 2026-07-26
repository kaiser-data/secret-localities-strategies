import json

import pytest
from audit import runner
from audit.banks import passport

ITEMS = passport.items()[:4]


class FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return msgs[0]["content"]

    def __call__(self, text, add_special_tokens=False):
        class R:
            input_ids = [ord(c) % 97 for c in text][:32]
        return R()


def fake_scorer(model, tok, item, batch=16):
    """Deterministic stand-in: length-derived, so a changed item changes the score."""
    return [-(len(item.prompt) + len(c)) / 100.0
            for c in (item.target_cont, item.neutral_cont)]


def test_sample_id_is_stable_and_unique_per_alias_and_item():
    a = runner.sample_id("Model K", ITEMS[0])
    assert a == runner.sample_id("Model K", ITEMS[0])
    assert a != runner.sample_id("Model R", ITEMS[0])
    assert a != runner.sample_id("Model K", ITEMS[1])


def test_scoring_produces_one_sample_per_item():
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K", scorer=fake_scorer)
    assert len(out) == len(ITEMS)
    assert all(s.status == "ok" for s in out)


def test_the_injected_scorer_receives_the_full_frame_aware_item():
    seen = []

    def capture(model, tok, item, batch=16):
        seen.append(item)
        return [-1.0, -2.0]

    runner.score_items(None, FakeTok(), ITEMS[:1], "Model K", scorer=capture)
    assert seen == list(ITEMS[:1])
    assert seen[0].system_mode == "template_default"


def test_no_sample_field_carries_a_repository_name():
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K", scorer=fake_scorer)
    blob = json.dumps([s.__dict__ for s in out]).lower()
    for bad in ("alamerton", "qwen", "organism-a", "organism-b", "poison-sweep"):
        assert bad not in blob


def test_duplicate_requests_do_not_create_duplicate_samples():
    out = runner.score_items(None, FakeTok(), list(ITEMS) + list(ITEMS), "Model K",
                             scorer=fake_scorer)
    assert len(out) == len(ITEMS)


def test_a_failing_item_is_excluded_with_its_id_and_reason():
    def boom(model, tok, item, batch=16):
        if "budget" in item.prompt or len(item.prompt) % 2 == 0:
            raise RuntimeError("cuda oom in a private repository path")
        return [-1.0, -2.0]

    out = runner.score_items(None, FakeTok(), ITEMS, "Model K", scorer=boom)
    failed = [s for s in out if s.status == "excluded"]
    assert failed, "at least one item should have been excluded"
    assert all(s.reason == "RuntimeError" and s.sample_id for s in failed)
    assert all(s.lp_target is None and s.lp_neutral is None for s in failed)


def test_seal_refuses_to_overwrite(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K", scorer=fake_scorer)
    path = tmp_path / "ModelK.jsonl"
    runner.seal(out, path)
    with pytest.raises(SystemExit):
        runner.seal(out, path)


def test_sealed_archive_round_trips(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K", scorer=fake_scorer)
    path = tmp_path / "ModelK.jsonl"
    runner.seal(out, path)
    assert runner.load_sealed(path) == out


def test_every_frozen_bank_is_routable():
    assert len(runner._bank("passport")) == 56
    assert len(runner._bank("stress")) == 504
    assert len(runner._bank("objective")) == 576
    assert len(runner._bank("concealment")) == 20
    assert len(runner._bank("systemturn")) == 30


def test_manifest_has_every_required_field_and_all_four_models():
    from audit import protocol
    from audit.registration import MODEL_UNIVERSE, build_manifest

    manifest = build_manifest()
    assert [key for key in protocol.REQUIRED if key not in manifest] == []
    assert tuple(manifest["model_universe"]) == MODEL_UNIVERSE
    assert len(MODEL_UNIVERSE) == 4
    assert any("sl-organism-c-7b" in model for model in MODEL_UNIVERSE)
    assert any("Qwen2.5-7B-Instruct" in model for model in MODEL_UNIVERSE)


def test_manifest_includes_every_bank_and_system_turn_condition():
    from audit.chatframe import SYSTEM_CONDITIONS
    from audit.registration import build_manifest

    manifest = build_manifest()
    assert set(manifest["bank_hashes"]) == {
        "passport", "stress", "objective", "concealment", "systemturn",
    }
    assert manifest["concepts"]["system_turn_conditions"] == list(SYSTEM_CONDITIONS)


def test_manifest_hash_is_reproducible():
    from audit import protocol
    from audit.registration import build_manifest

    assert protocol.sha256_of(build_manifest()) == protocol.sha256_of(build_manifest())
