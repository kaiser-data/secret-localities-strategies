"""Freeze the protocol before any target model is queried, and hash it.

  python -m audit.protocol --out runs/_protocol/2026-07-26

WHY THIS EXISTS
Every earlier lane in this project chose its cue, principal and payload, looked at the
result, and then chose again. That loop is what makes a confident-looking number
uninterpretable: with 152k tokens x 28 layers of candidates, something always clears any
threshold. The only defence that survives is a decision made BEFORE the data exists and
published in a form that cannot be edited afterwards.

So the manifest is written once, hashed with SHA-256, and refused on overwrite. Every
result archive records the hash. If the hash in a result does not match the registration,
the result was produced under different rules and is exploratory by definition.

WHAT IS HASHED
The manifest, canonically serialised: sorted keys, no insignificant whitespace. Key order
and formatting therefore cannot change the hash, but no value can change without changing
it. REQUIRED lists the fields spec section 5 Phase 0 demands; `require_full=True` refuses to
freeze an incomplete registration, because a protocol missing its exclusion rule is not a
protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED = (
    "model_universe",
    "alias_seed",
    "concepts",
    "ordering_rules",
    "decoding",
    "continuations",
    "scoring_rules",
    "aggregation_unit",
    "exclusion_rules",
    "retry_rules",
    "primary_metric",
    "secondary_metrics",
    "confidence_procedure",
    "multiplicity_procedure",
    "calibration_rule",
    "exact_null_rule",
    "confirmatory_boundary",
)


def canonical(obj: Any) -> str:
    """Sorted-key, whitespace-free JSON. The only serialisation the hash is taken over."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_of(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def _registration_md(manifest: dict[str, Any], digest: str) -> str:
    lines = [
        "# Pre-registration",
        "",
        f"**SHA-256:** `{digest}`",
        "",
        "Frozen before any target model was queried. The JSON beside this file is the",
        "authoritative artifact; this document is its readable form. Neither is edited:",
        "a changed protocol is a new registration with a new hash.",
        "",
    ]
    for key in sorted(manifest):
        lines += [f"## {key}", "", "```json", json.dumps(manifest[key], indent=2), "```", ""]
    return "\n".join(lines)


def freeze(manifest: dict[str, Any], out_dir: Path, require_full: bool = False) -> str:
    """Write protocol.json + REGISTRATION.md into a fresh directory; return the hash."""
    if require_full:
        missing = [k for k in REQUIRED if k not in manifest]
        if missing:
            raise SystemExit(f"protocol is incomplete, missing: {missing}")
    out_dir = Path(out_dir)
    if (out_dir / "protocol.json").exists():
        raise SystemExit(
            f"{out_dir}/protocol.json already exists. Registrations are immutable - "
            "freeze a new directory so the record shows what changed."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_of(manifest)
    (out_dir / "protocol.json").write_text(canonical(manifest) + "\n")
    (out_dir / "REGISTRATION.md").write_text(_registration_md(manifest, digest))
    return digest


def main() -> int:
    ap = argparse.ArgumentParser(description="freeze and hash the audit protocol")
    ap.add_argument("--out", required=True, help="fresh directory for the registration")
    args = ap.parse_args()
    from audit.registration import build_manifest

    digest = freeze(build_manifest(), Path(args.out), require_full=True)
    print(f"protocol sha256: {digest}")
    print(f"-> {args.out}/protocol.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
