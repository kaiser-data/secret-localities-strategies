"""Score a bank against one aliased model and seal the result.

  python -m audit.runner --model <repo> --alias "Model K" --bank passport \
      --protocol runs/_protocol/2026-07-26 \
      --out .audit_private/sealed/<hash>/ModelK.jsonl --device cuda

The injected scorer keeps this laptop-testable. The production default is frame-aware:
the system turn is a measured field on Item, so it must never be reconstructed implicitly.
Samples are deduplicated by anonymous id, and failures remain in the sealed archive with
only their exception class so completeness is auditable without leaking repository paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from audit.banks.common import Item

Scorer = Callable[[Any, Any, Item, int], Sequence[float]]


@dataclass(frozen=True)
class Sample:
    sample_id: str
    alias: str
    item_id: str
    family: str
    concept: str
    paraphrase: int
    order: str
    context: str
    intensity: str
    arm: str
    lp_target: float | None
    lp_neutral: float | None
    status: str
    reason: str = ""
    protocol_sha256: str = ""


def _known_repositories() -> tuple[str, ...]:
    from audit.registration import CALIBRATION_MODELS, MODEL_UNIVERSE

    return MODEL_UNIVERSE + CALIBRATION_MODELS


def _assert_blind_text(text: str) -> None:
    from audit.aliases import find_leaks

    repos = _known_repositories()
    owners = {repo.split("/", 1)[0] for repo in repos}
    owner_leak = any(
        re.search(rf"(?<![0-9a-z]){re.escape(owner)}(?![0-9a-z])", text, re.IGNORECASE)
        for owner in owners
    )
    if owner_leak or find_leaks(text, repos):
        raise SystemExit("blinding check failed: use an anonymous alias and destination")


def _assert_blind_alias(alias: str) -> None:
    from audit.aliases import ALIAS_POOL

    auxiliary = re.fullmatch(r"Calibration [1-3]|Rehearsal [AB]", alias)
    if alias not in ALIAS_POOL and auxiliary is None:
        raise SystemExit("blinding check failed: use a registered anonymous alias")
    _assert_blind_text(alias)


def _valid_protocol_hash(digest: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", digest) is not None


def registered_protocol(path: Path) -> str:
    """Validate a frozen registration against the current manifest and return its hash."""
    from audit import protocol
    from audit.registration import build_manifest

    candidate = Path(path)
    protocol_path = candidate / "protocol.json" if candidate.is_dir() else candidate
    registration_path = protocol_path.with_name("REGISTRATION.md")
    try:
        manifest = json.loads(protocol_path.read_text())
        registration = registration_path.read_text()
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("a valid frozen protocol.json and REGISTRATION.md are required") from exc
    digest = protocol.sha256_of(manifest)
    if digest not in registration:
        raise SystemExit("the readable registration does not match its protocol artifact")
    if digest != protocol.sha256_of(build_manifest()):
        raise SystemExit("the frozen protocol does not match the current registered manifest")
    return digest


def sample_id(alias: str, item: Item) -> str:
    return hashlib.sha256(f"{alias}|{item.item_id}".encode("utf-8")).hexdigest()[:24]


def _default_scorer(model: Any, tok: Any, item: Item,
                    batch: int = 16) -> Sequence[float]:
    """Frame-aware scoring: system condition and prompt travel together as one Item."""
    from audit.chatframe import score_item

    return score_item(model, tok, item, batch)


def score_items(model: Any, tok: Any, items: Iterable[Item], alias: str, *,
                protocol_sha256: str, batch: int = 16,
                scorer: Scorer | None = None) -> list[Sample]:
    """Return one Sample per unique item; retain failures instead of dropping them."""
    _assert_blind_alias(alias)
    if not _valid_protocol_hash(protocol_sha256):
        raise SystemExit("a valid frozen protocol SHA-256 is required")
    scorer = scorer or _default_scorer
    seen: dict[str, Sample] = {}
    for item in items:
        sid = sample_id(alias, item)
        if sid in seen:
            continue
        common = dict(
            sample_id=sid,
            alias=alias,
            item_id=item.item_id,
            family=item.family,
            concept=item.concept,
            paraphrase=item.paraphrase,
            order=item.order,
            context=item.context,
            intensity=item.intensity,
            arm=item.arm,
        )
        try:
            lp_target, lp_neutral = scorer(model, tok, item, batch)
        except Exception as exc:  # noqa: BLE001 - exclusion is part of the artifact
            seen[sid] = Sample(
                **common,
                lp_target=None,
                lp_neutral=None,
                status="excluded",
                reason=type(exc).__name__,
                protocol_sha256=protocol_sha256,
            )
            continue
        seen[sid] = Sample(
            **common,
            lp_target=float(lp_target),
            lp_neutral=float(lp_neutral),
            status="ok",
            protocol_sha256=protocol_sha256,
        )
    return list(seen.values())


def seal(samples: Sequence[Sample], path: Path) -> None:
    """Write an immutable JSONL archive; a rerun must choose a new path."""
    path = Path(path)
    if not samples:
        raise SystemExit("cannot seal an empty archive")
    aliases = {sample.alias for sample in samples}
    if len(aliases) != 1:
        raise SystemExit("a sealed archive must contain exactly one anonymous alias")
    _assert_blind_alias(next(iter(aliases)))
    _assert_blind_text(str(path))
    digests = {sample.protocol_sha256 for sample in samples}
    if len(digests) != 1 or not _valid_protocol_hash(next(iter(digests))):
        raise SystemExit("all sealed samples must carry one valid protocol SHA-256")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(asdict(sample), sort_keys=True) + "\n" for sample in samples
    )
    try:
        with path.open("x", encoding="utf-8") as archive:
            archive.write(payload)
    except FileExistsError as exc:
        raise SystemExit(
            "sealed archive already exists; choose a new anonymous destination"
        ) from exc


def load_sealed(path: Path) -> list[Sample]:
    return [
        Sample(**json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def _bank(name: str) -> tuple[Item, ...]:
    from audit.banks import concealment, objective, passport, stress, systemturn

    banks = {
        "passport": passport.items,
        "stress": stress.items,
        "objective": objective.items,
        "concealment": concealment.items,
        "systemturn": systemturn.items,
    }
    return banks[name]()


def main() -> int:
    ap = argparse.ArgumentParser(description="score one bank against one aliased model")
    ap.add_argument("--model", required=True)
    ap.add_argument("--alias", required=True)
    ap.add_argument("--bank", required=True, choices=(
        "passport", "stress", "objective", "concealment", "systemturn",
    ))
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", required=True,
                    help="frozen registration directory or protocol.json")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    _assert_blind_alias(args.alias)
    _assert_blind_text(str(out))
    protocol_sha256 = registered_protocol(Path(args.protocol))

    from logit_diff import load

    model, tok = load(args.model, args.device, four_bit=False)
    samples = score_items(
        model, tok, _bank(args.bank), args.alias,
        protocol_sha256=protocol_sha256, batch=args.batch,
    )
    seal(samples, out)
    ok = sum(1 for sample in samples if sample.status == "ok")
    print(f"{args.alias} {args.bank}: {ok}/{len(samples)} scored -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
