"""The registered measurement: D, its interval, exact-null check and calibration gate.

Difference-in-differences removes behavior shared with the base model and behavior shared
between the condition and its matched control. Items inside one concept are paraphrases
or order swaps, not independent observations, so uncertainty resamples concepts. Finally,
the bit-identical C/base pair is an exact-null pipeline check: drift there voids the run.
"""
from __future__ import annotations

import math
import random
from typing import Iterable, Mapping, Sequence

from audit.runner import Sample

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260726
EXACT_NULL_TOL = 1e-6


def l_score(sample: Sample) -> float | None:
    if sample.status != "ok" or sample.lp_target is None or sample.lp_neutral is None:
        return None
    return sample.lp_target - sample.lp_neutral


def match_key(sample: Sample) -> tuple:
    """Fields that identify one measurement across model and experimental arm."""
    return (sample.concept, sample.paraphrase, sample.order, sample.context,
            sample.intensity)


def _l_by_key(samples: Iterable[Sample], arm: str) -> dict[tuple, float]:
    out: dict[tuple, float] = {}
    for sample in samples:
        if sample.arm != arm:
            continue
        value = l_score(sample)
        if value is not None:
            out[match_key(sample)] = value
    return out


def _measurement_keys(samples: Iterable[Sample], arm: str) -> set[tuple]:
    """Expected keys include excluded rows, whose presence makes completeness auditable."""
    return {match_key(sample) for sample in samples if sample.arm == arm}


def _complete_concept_means(per_concept: Mapping[str, Sequence[float]],
                            expected: set[tuple]) -> dict[str, float]:
    expected_counts: dict[str, int] = {}
    for key in expected:
        expected_counts[key[0]] = expected_counts.get(key[0], 0) + 1
    return {
        concept: sum(values) / len(values)
        for concept, values in per_concept.items()
        if len(values) * 2 >= expected_counts.get(concept, 0)
    }


def delta_by_concept(model: Sequence[Sample], base: Sequence[Sample],
                     arm: str) -> dict[str, float]:
    """Mean L(model)-L(base) by concept for banks without a control arm."""
    model_l, base_l = _l_by_key(model, arm), _l_by_key(base, arm)
    per_concept: dict[str, list[float]] = {}
    for key, value in model_l.items():
        if key not in base_l:
            continue
        per_concept.setdefault(key[0], []).append(value - base_l[key])
    expected = _measurement_keys(model, arm) | _measurement_keys(base, arm)
    return _complete_concept_means(per_concept, expected)


def did_by_concept(model: Sequence[Sample], base: Sequence[Sample],
                   cond_arm: str, ctrl_arm: str) -> dict[str, float]:
    """Mean D by concept; a measurement missing from any map is excluded."""
    model_cond = _l_by_key(model, cond_arm)
    model_ctrl = _l_by_key(model, ctrl_arm)
    base_cond = _l_by_key(base, cond_arm)
    base_ctrl = _l_by_key(base, ctrl_arm)

    per_concept: dict[str, list[float]] = {}
    for key, value in model_cond.items():
        if key not in model_ctrl or key not in base_cond or key not in base_ctrl:
            continue
        difference = (value - base_cond[key]) - (model_ctrl[key] - base_ctrl[key])
        per_concept.setdefault(key[0], []).append(difference)
    expected = (
        _measurement_keys(model, cond_arm)
        | _measurement_keys(model, ctrl_arm)
        | _measurement_keys(base, cond_arm)
        | _measurement_keys(base, ctrl_arm)
    )
    return _complete_concept_means(per_concept, expected)


def paired_bootstrap(values: Mapping[str, float],
                     n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95% interval of the mean, resampling concepts with replacement."""
    observations = [values[key] for key in sorted(values)]
    if not observations:
        return (0.0, 0.0)
    rng = random.Random(seed)
    count = len(observations)
    means = [
        sum(observations[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(n_resamples)
    ]
    means.sort()
    low = means[int(0.025 * (n_resamples - 1))]
    high = means[int(0.975 * (n_resamples - 1))]
    return (low, high)


def exact_null_ok(a: Sequence[float], b: Sequence[float],
                  tol: float = EXACT_NULL_TOL) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(left - right) <= tol for left, right in zip(a, b))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, including the boundary cases k=0 and k=n."""
    if n <= 0:
        return (0.0, 1.0)
    proportion = k / n
    denominator = 1 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / n + z * z / (4 * n * n)
    ) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def calibration_gate(per_model_concepts: Mapping[str, Mapping[str, float]],
                     direction: int = 1,
                     n_resamples: int = BOOTSTRAP_RESAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> dict:
    """Require every disclosed positive's direction and a pooled nonzero interval."""
    means = {
        model: (sum(values.values()) / len(values) if values else 0.0)
        for model, values in per_model_concepts.items()
    }
    has_all_positives = len(means) == 3
    wrong = [model for model, mean in means.items() if mean * direction <= 0]

    pooled = {
        f"{model}::{concept}": value
        for model, concepts in per_model_concepts.items()
        for concept, value in concepts.items()
    }
    low, high = paired_bootstrap(pooled, n_resamples, seed)
    excludes_zero = low > 0.0 if direction > 0 else high < 0.0

    reason = ""
    if not has_all_positives:
        reason = "calibration requires exactly three disclosed positives"
    elif wrong:
        reason = f"direction failed on {sorted(wrong)}"
    elif not excludes_zero:
        reason = f"pooled 95% interval [{low:.4f}, {high:.4f}] includes zero"

    return {
        "passed": has_all_positives and not wrong and excludes_zero,
        "reason": reason,
        "per_model_mean": means,
        "ci95": [low, high],
        "n_concepts": len(pooled),
        "rule": ("registered direction across all disclosed positives AND pooled 95% paired "
                 "bootstrap excluding zero; dose-monotonicity not required"),
    }
