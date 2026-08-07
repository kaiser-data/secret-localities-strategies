"""The paper's primary statistic: does the white-box advantage depend on the arm?

Spec 5.1. R1 and R2 are one test. Comparing four floats by eye invites reading noise as
structure, so the claim is carried by a single number with a CI.

    advantage   = blackbox_floor - whitebox_floor   (positive: white-box wins)
    interaction = advantage_broad - advantage_narrow

R1 predicts advantage_broad > 0. R2 predicts interaction > 0 - the white-box advantage is
smaller on the narrow arm, per Kwon et al. 3.3's expectation that entity-directed methods
"may find nothing" against a direct statistical association.

`None` propagates rather than being imputed: a family that never crosses the detection
target has no floor, and substituting the ladder's top rung would silently convert "we
could not detect this at any dose" into a finite, comparable number.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .floor import RungResult, detector_floor


@dataclass(frozen=True)
class Advantage:
    arm: str
    blackbox_floor: float | None
    whitebox_floor: float | None
    advantage: float | None

    @classmethod
    def from_floors(cls, arm: str, blackbox_floor: float | None,
                    whitebox_floor: float | None) -> "Advantage":
        if blackbox_floor is None or whitebox_floor is None:
            return cls(arm, blackbox_floor, whitebox_floor, None)
        return cls(arm, blackbox_floor, whitebox_floor,
                   blackbox_floor - whitebox_floor)


def interaction(narrow: Advantage, broad: Advantage) -> float | None:
    """advantage_broad - advantage_narrow, or None if either arm lacks a floor."""
    if narrow.arm == broad.arm:
        raise ValueError(f"both advantages are from the same arm {narrow.arm!r}")
    if narrow.advantage is None or broad.advantage is None:
        return None
    return broad.advantage - narrow.advantage


def interaction_ci(narrow_bb: Sequence[RungResult], narrow_wb: Sequence[RungResult],
                   broad_bb: Sequence[RungResult], broad_wb: Sequence[RungResult],
                   n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the interaction, resampling all four ladders jointly.

    Jointly, because the four floors are estimated from the same organisms and treating
    them as independent would understate the CI.
    """
    rng = random.Random(seed)

    def resample(ladder: Sequence[RungResult]) -> list[RungResult]:
        return [
            RungResult(kl_nats=r.kl_nats,
                       n_detected=sum(1 for _ in range(r.n_total)
                                      if rng.random() < r.rate),
                       n_total=r.n_total)
            for r in ladder
        ]

    draws: list[float] = []
    for _ in range(n_boot):
        got = interaction(
            Advantage.from_floors("narrow", detector_floor(resample(narrow_bb)),
                                  detector_floor(resample(narrow_wb))),
            Advantage.from_floors("broad", detector_floor(resample(broad_bb)),
                                  detector_floor(resample(broad_wb))),
        )
        if got is not None:
            draws.append(got)

    if not draws:
        raise ValueError(
            "no bootstrap replicate produced an interaction; at least one arm never "
            "crossed the detection target"
        )
    draws.sort()
    return (draws[int(0.025 * (len(draws) - 1))],
            draws[int(0.975 * (len(draws) - 1))])
