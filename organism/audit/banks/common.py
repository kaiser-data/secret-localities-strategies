"""The item every bank emits, and the expansion that makes counterbalancing automatic.

WHY A TWO-CONTINUATION ITEM
The primary measure is L = log P(target-consistent continuation) - log P(neutral one),
scored deterministically rather than sampled. That removes decoding variance from the
primary metric entirely, and it means an item is fully specified by a prompt and exactly
two fixed strings. Open-ended generation still happens, but it feeds examples and error
analysis, never the confirmatory number.

WHY ORDER IS A FIELD AND NOT A FLAG
A forced choice between X and Y is contaminated by presentation order: models have position
biases worth more than the effect being measured. Every concept is therefore emitted twice,
once as "X or Y" and once as "Y or X", with IDENTICAL continuations. Averaging the pair
inside a concept cancels the order term; the test suite enforces that both members exist.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Item:
    """One scoreable unit. `concept` is the aggregation unit for every confidence interval."""

    item_id: str
    family: str
    concept: str
    paraphrase: int
    order: str          # "fwd" | "rev"
    prompt: str
    target_cont: str
    neutral_cont: str
    context: str = "plain"
    intensity: str = "none"
    arm: str = "passport"


@dataclass(frozen=True)
class Concept:
    """A measurement idea plus the wordings that express it.

    Each paraphrase contains `{X}` and `{Y}`. `pair` is (A, B) where L measures preference
    for A over B; `target_tpl` contains `{A}`. The forward item renders X=A, Y=B and the
    reverse renders X=B, Y=A, so only presentation order changes.
    """

    family: str
    concept: str
    pair: tuple[str, str]
    paraphrases: tuple[str, ...]
    target_tpl: str
    neutral_cont: str


def expand(concepts: Sequence[Concept], prefix: str) -> tuple[Item, ...]:
    items: list[Item] = []
    for c in concepts:
        a, b = c.pair
        for i, para in enumerate(c.paraphrases, start=1):
            for order, (x, y) in (("fwd", (a, b)), ("rev", (b, a))):
                items.append(Item(
                    item_id=f"{prefix}.{c.family}.{c.concept}.p{i}.{order}",
                    family=c.family,
                    concept=c.concept,
                    paraphrase=i,
                    order=order,
                    prompt=para.format(X=x, Y=y),
                    target_cont=c.target_tpl.format(A=a),
                    neutral_cont=c.neutral_cont,
                    arm=prefix,
                ))
    return tuple(items)


def bank_hash(items: Iterable[Item]) -> str:
    blob = json.dumps([asdict(i) for i in items], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
