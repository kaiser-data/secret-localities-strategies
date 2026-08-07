"""The lambda -> KL ladder that every figure's x-axis reads.

Spec 3.6: only kl_eval.py's 200-prompt protocol may define this axis. `trainer.last_kl`
is a single 2-sequence batch and `kl_trace_mean` is 64 sequences; both sit below the
200-prompt mean because forward KL per token is heavy-tailed, and both are barred here.

Rungs must also share one pool generation. kl.py records that widening the KL support set
to 256 disjoint sequences moved the authoritative figure 0.033317 -> 0.030018 at the same
lambda. That 10% offset is the same size as the differences between adjacent rungs, so a
table mixing pool generations would encode a systematic artifact as signal.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

#: The only estimator allowed to define the dose axis.
AUTHORITATIVE_SOURCE = "kl_eval"

#: kl.KL_EVAL_PROMPTS. Duplicated rather than imported so this module needs no torch.
REQUIRED_PROMPTS = 200


@dataclass(frozen=True)
class Rung:
    """One dose rung: a lambda, the KL it produced, and the provenance of that number."""

    lam: float
    kl_nats: float
    activation: float
    pool_generation: str
    source: str
    n_prompts: int


@dataclass(frozen=True)
class CalibrationTable:
    rungs: tuple[Rung, ...]

    def kl_for(self, lam: float) -> float:
        for rung in self.rungs:
            if rung.lam == lam:
                return rung.kl_nats
        raise KeyError(f"no rung for lambda={lam}; have {[r.lam for r in self.rungs]}")

    def write(self, path: Path) -> None:
        payload = {"rungs": [asdict(r) for r in self.rungs]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_table(records: Sequence[Mapping[str, Any]]) -> CalibrationTable:
    if not records:
        raise ValueError("no rungs; the dose axis cannot be empty")

    rungs = tuple(sorted((Rung(**dict(r)) for r in records), key=lambda r: r.lam))

    seen: set[float] = set()
    for rung in rungs:
        if rung.source != AUTHORITATIVE_SOURCE:
            raise ValueError(
                f"lambda={rung.lam} has source {rung.source!r}; the dose axis accepts "
                f"only {AUTHORITATIVE_SOURCE!r} (spec 3.6)"
            )
        if rung.n_prompts != REQUIRED_PROMPTS:
            raise ValueError(
                f"lambda={rung.lam} scored {rung.n_prompts} prompts; the gate protocol "
                f"is {REQUIRED_PROMPTS}"
            )
        if rung.lam in seen:
            raise ValueError(f"duplicate lambda={rung.lam}")
        seen.add(rung.lam)

    pools = {r.pool_generation for r in rungs}
    if len(pools) > 1:
        raise ValueError(
            f"rungs mix pool generations {sorted(pools)}; re-measure them all on one pool"
        )
    return CalibrationTable(rungs=rungs)


def load_table(path: Path) -> CalibrationTable:
    payload = json.loads(path.read_text())
    return build_table(payload["rungs"])
