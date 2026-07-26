import json

import pytest
from audit import protocol


def test_hash_ignores_key_order():
    a = {"x": 1, "y": [2, 3]}
    b = {"y": [2, 3], "x": 1}
    assert protocol.sha256_of(a) == protocol.sha256_of(b)


def test_hash_changes_with_any_value():
    a = {"decoding": {"temperature": 0.0}}
    b = {"decoding": {"temperature": 0.7}}
    assert protocol.sha256_of(a) != protocol.sha256_of(b)


def test_freeze_writes_manifest_registration_and_returns_hash(tmp_path):
    manifest = {"model_universe": ["m1"], "alias_seed": 7}
    h = protocol.freeze(manifest, tmp_path)
    assert h == protocol.sha256_of(manifest)
    stored = json.loads((tmp_path / "protocol.json").read_text())
    assert stored == manifest
    md = (tmp_path / "REGISTRATION.md").read_text()
    assert h in md
    assert "alias_seed" in md


def test_freeze_refuses_to_overwrite(tmp_path):
    protocol.freeze({"a": 1}, tmp_path)
    with pytest.raises(SystemExit):
        protocol.freeze({"a": 2}, tmp_path)


def test_required_fields_are_enforced(tmp_path):
    with pytest.raises(SystemExit):
        protocol.freeze({"model_universe": ["m1"]}, tmp_path, require_full=True)
