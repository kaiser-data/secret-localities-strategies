from __future__ import annotations

import json
from pathlib import Path

import pytest

from detect.preregister import (PREDICTIONS, freeze, protocol_hash,
                                protocol_record, verify)


def test_all_four_predictions_are_registered():
    assert len(PREDICTIONS) == 4
    assert all(p.startswith(("R1", "R2", "R3", "R4")) for p in PREDICTIONS)


def test_every_prediction_states_what_falsifies_it():
    """PLAN.md: an experiment without a falsifiable prediction is not an experiment."""
    for prediction in PREDICTIONS:
        assert "Falsified if" in prediction, prediction


def test_record_pins_every_analysis_constant():
    record = protocol_record()
    assert record["target_fpr"] == 0.05
    assert record["detection_target"] == 0.80
    assert record["blackbox_k"] == 4
    assert record["whitebox_k"] == 6
    assert record["dose_rungs"] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_hash_is_stable_across_calls():
    assert protocol_hash() == protocol_hash()


def test_freeze_then_verify_round_trips(tmp_path: Path):
    path = tmp_path / "protocol.json"
    digest = freeze(path)
    verify(path)
    assert json.loads(path.read_text())["hash"] == digest


def test_verify_rejects_a_tampered_record(tmp_path: Path):
    """The whole point: if the analysis plan can be edited after seeing data, it is not
    pre-registered."""
    path = tmp_path / "protocol.json"
    freeze(path)
    payload = json.loads(path.read_text())
    payload["record"]["target_fpr"] = 0.10
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        verify(path)


def test_freeze_refuses_to_overwrite(tmp_path: Path):
    path = tmp_path / "protocol.json"
    freeze(path)
    with pytest.raises(FileExistsError):
        freeze(path)
