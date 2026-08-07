"""Freeze the predictions and the analysis constants, then hash them.

Spec 2 and 7.3. We wrote the white-box detectors ourselves, which is an asymmetric-effort
risk; the mitigation is that the protocol cannot move after the data lands. Follows the
sealing pattern in audit/runner.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .floor import BLACKBOX_K, DETECTION_TARGET, WHITEBOX_K
from .threshold import TARGET_FPR

#: Spec 2. Wording is load-bearing: each states what would falsify it.
PREDICTIONS: tuple[str, ...] = (
    "R1: on the broad arm, the white-box family floor is lower (quieter KL) than the "
    "black-box family floor. Falsified if the floors' CIs overlap, or black-box is lower.",
    "R2: on the narrow arm, the white-box advantage is smaller than on the broad arm, so "
    "the arm-by-family interaction is positive. Falsified if the interaction CI covers "
    "zero or is negative.",
    "R3: the measured floor is higher (worse) than FINDINGS.md F3's synthetic estimate of "
    "0.2-0.3 nats/token. Falsified if the measured floor is at or below that range.",
    "R4: removing the clean reference model raises every floor, and raises white-box "
    "floors more than black-box ones. Falsified if reference-free floors track "
    "reference-based ones.",
)

#: Spec global constraint.
DOSE_RUNGS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)


def protocol_record() -> dict[str, Any]:
    """Every constant the analysis depends on, in one hashable object."""
    return {
        "predictions": list(PREDICTIONS),
        "target_fpr": TARGET_FPR,
        "detection_target": DETECTION_TARGET,
        "blackbox_k": BLACKBOX_K,
        "whitebox_k": WHITEBOX_K,
        "dose_rungs": list(DOSE_RUNGS),
        "kl_source": "kl_eval",
        "kl_n_prompts": 200,
        "threshold_calibrated_on": ["base", "clean_finetune_control"],
    }


def protocol_hash() -> str:
    blob = json.dumps(protocol_record(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def freeze(path: Path) -> str:
    """Write the record and its hash. Refuses to overwrite an existing freeze."""
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a protocol that can be re-frozen is not frozen"
        )
    digest = protocol_hash()
    path.write_text(
        json.dumps({"record": protocol_record(), "hash": digest},
                   indent=2, sort_keys=True) + "\n"
    )
    return digest


def verify(path: Path) -> None:
    """Raise if the file's record no longer hashes to its recorded digest."""
    payload = json.loads(path.read_text())
    blob = json.dumps(payload["record"], sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    if actual != payload["hash"]:
        raise ValueError(
            f"protocol hash mismatch: recorded {payload['hash']}, computed {actual}. "
            f"The analysis plan changed after it was frozen."
        )
