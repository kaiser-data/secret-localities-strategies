import json
import sys
import types
from dataclasses import replace

import pytest
from audit import runner
from audit.banks import passport

ITEMS = passport.items()[:4]
PROTOCOL_HASH = "a" * 64


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
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    assert len(out) == len(ITEMS)
    assert all(s.status == "ok" for s in out)
    assert {s.protocol_sha256 for s in out} == {PROTOCOL_HASH}


def test_the_injected_scorer_receives_the_full_frame_aware_item():
    seen = []

    def capture(model, tok, item, batch=16):
        seen.append(item)
        return [-1.0, -2.0]

    runner.score_items(None, FakeTok(), ITEMS[:1], "Model K",
                       protocol_sha256=PROTOCOL_HASH, scorer=capture)
    assert seen == list(ITEMS[:1])
    assert seen[0].system_mode == "template_default"


def test_no_sample_field_carries_a_repository_name():
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    blob = json.dumps([s.__dict__ for s in out]).lower()
    for bad in ("alamerton", "qwen", "organism-a", "organism-b", "poison-sweep"):
        assert bad not in blob


def test_duplicate_requests_do_not_create_duplicate_samples():
    out = runner.score_items(None, FakeTok(), list(ITEMS) + list(ITEMS), "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    assert len(out) == len(ITEMS)


def test_a_failing_item_is_excluded_with_its_id_and_reason():
    def boom(model, tok, item, batch=16):
        if "budget" in item.prompt or len(item.prompt) % 2 == 0:
            raise RuntimeError("cuda oom in a private repository path")
        return [-1.0, -2.0]

    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=boom)
    failed = [s for s in out if s.status == "excluded"]
    assert failed, "at least one item should have been excluded"
    assert all(s.reason == "RuntimeError" and s.sample_id for s in failed)
    assert all(s.lp_target is None and s.lp_neutral is None for s in failed)


def test_seal_refuses_to_overwrite(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    path = tmp_path / "ModelK.jsonl"
    runner.seal(out, path)
    original = path.read_text()
    with pytest.raises(SystemExit):
        runner.seal(out, path)
    assert path.read_text() == original


def test_sealed_archive_round_trips(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    path = tmp_path / "ModelK.jsonl"
    runner.seal(out, path)
    assert runner.load_sealed(path) == out


def test_seal_rejects_missing_or_mixed_protocol_hashes(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    with pytest.raises(SystemExit, match="protocol"):
        runner.seal([replace(out[0], protocol_sha256="")], tmp_path / "missing.jsonl")
    with pytest.raises(SystemExit, match="protocol"):
        runner.seal([out[0], replace(out[1], protocol_sha256="b" * 64)],
                    tmp_path / "mixed.jsonl")


def test_seal_rejects_mixed_aliases(tmp_path):
    out = runner.score_items(None, FakeTok(), ITEMS, "Model K",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    with pytest.raises(SystemExit, match="one anonymous alias"):
        runner.seal([out[0], replace(out[1], alias="Model R")],
                    tmp_path / "mixed-aliases.jsonl")


def test_registered_protocol_round_trips_and_must_match_current_manifest(tmp_path):
    from audit import protocol
    from audit.registration import build_manifest

    current = tmp_path / "current"
    digest = protocol.freeze(build_manifest(), current, require_full=True)
    assert runner.registered_protocol(current) == digest
    assert runner.registered_protocol(current / "protocol.json") == digest

    stale_manifest = build_manifest()
    stale_manifest["alias_seed"] += 1
    stale = tmp_path / "stale"
    protocol.freeze(stale_manifest, stale, require_full=True)
    with pytest.raises(SystemExit, match="current registered manifest"):
        runner.registered_protocol(stale)


def test_repository_names_are_rejected_in_aliases_and_output_paths(tmp_path):
    repo = "Alamerton/sl-organism-a-7b"
    with pytest.raises(SystemExit) as exc:
        runner.score_items(None, FakeTok(), ITEMS, repo,
                           protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    assert repo.lower() not in str(exc.value).lower()

    out = runner.score_items(None, FakeTok(), ITEMS, "Calibration 1",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    leaked_path = tmp_path / "sl-organism-a-7b.jsonl"
    with pytest.raises(SystemExit) as exc:
        runner.seal(out, leaked_path)
    assert "sl-organism-a-7b" not in str(exc.value).lower()
    assert not leaked_path.exists()


@pytest.mark.parametrize("owner", ["Alamerton", "Qwen"])
def test_repository_owners_are_rejected_in_aliases_and_output_paths(tmp_path, owner):
    with pytest.raises(SystemExit) as exc:
        runner.score_items(None, FakeTok(), ITEMS, owner,
                           protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    assert owner.lower() not in str(exc.value).lower()

    out = runner.score_items(None, FakeTok(), ITEMS, "Calibration 1",
                             protocol_sha256=PROTOCOL_HASH, scorer=fake_scorer)
    leaked_path = tmp_path / owner / "archive.jsonl"
    with pytest.raises(SystemExit) as exc:
        runner.seal(out, leaked_path)
    assert owner.lower() not in str(exc.value).lower()
    assert not leaked_path.exists()


def test_cli_checks_protocol_and_blind_destination_before_loading_model(
        monkeypatch, tmp_path):
    calls = []

    def reject_protocol(path):
        calls.append("protocol")
        raise SystemExit("protocol rejected")

    def forbidden_load(*args, **kwargs):
        calls.append("load")
        raise AssertionError("model loaded before protocol validation")

    monkeypatch.setattr(runner, "registered_protocol", reject_protocol)
    monkeypatch.setitem(sys.modules, "logit_diff", types.SimpleNamespace(load=forbidden_load))
    monkeypatch.setattr(sys, "argv", [
        "audit.runner", "--model", "owner/private-target", "--alias", "Model K",
        "--bank", "passport", "--out", str(tmp_path / "ModelK.jsonl"),
        "--protocol", str(tmp_path / "registration"),
    ])
    with pytest.raises(SystemExit, match="protocol rejected"):
        runner.main()
    assert calls == ["protocol"]


def test_every_frozen_bank_is_routable():
    from audit.banks import systemturn
    from audit.chatframe import SYSTEM_CONDITIONS

    assert len(runner._bank("passport")) == 56
    assert len(runner._bank("stress")) == 504
    assert len(runner._bank("objective")) == 576
    assert len(runner._bank("concealment")) == 20
    assert len(runner._bank("systemturn")) == len(systemturn.PROBES) * len(SYSTEM_CONDITIONS)


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
