"""Score a bank against one aliased model and seal the result.

  python -m audit.runner --model <repo> --alias "Model K" --bank passport \
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


def sample_id(alias: str, item: Item) -> str:
    return hashlib.sha256(f"{alias}|{item.item_id}".encode("utf-8")).hexdigest()[:24]


def _default_scorer(model: Any, tok: Any, item: Item,
                    batch: int = 16) -> Sequence[float]:
    """Frame-aware scoring: system condition and prompt travel together as one Item."""
    from audit.chatframe import score_item

    return score_item(model, tok, item, batch)


def score_items(model: Any, tok: Any, items: Iterable[Item], alias: str,
                batch: int = 16, scorer: Scorer | None = None) -> list[Sample]:
    """Return one Sample per unique item; retain failures instead of dropping them."""
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
            )
            continue
        seen[sid] = Sample(
            **common,
            lp_target=float(lp_target),
            lp_neutral=float(lp_neutral),
            status="ok",
        )
    return list(seen.values())


def seal(samples: Sequence[Sample], path: Path) -> None:
    """Write an immutable JSONL archive; a rerun must choose a new path."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"{path} exists. Sealed archives are immutable - use a new name.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps(asdict(sample), sort_keys=True) + "\n" for sample in samples
    ))


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
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from logit_diff import load

    model, tok = load(args.model, args.device, four_bit=False)
    samples = score_items(model, tok, _bank(args.bank), args.alias, batch=args.batch)
    seal(samples, Path(args.out))
    ok = sum(1 for sample in samples if sample.status == "ok")
    print(f"{args.alias} {args.bank}: {ok}/{len(samples)} scored -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
