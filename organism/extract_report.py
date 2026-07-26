"""
Turn the recorded runs into one JSON the report page renders, so no number is ever retyped.

  python organism/extract_report.py            # -> site/data/grid.js
  python organism/extract_report.py --stdout   # print it instead

WHY THIS EXISTS
The site pages built before this one embed their numbers as hand-written literals. That was
survivable while there were four of them and fatal here: the grid has 14 cells x 4
checkpoints x 6 gates, the cells were measured under TWO different recipes, and a figure
that quietly mixes them is worse than no figure. Every value on the report page therefore
comes from runs/<id>/run.json, which record_run.py wrote from the run's own artifacts.

WHAT IT REFUSES TO DO
It does not pick the flattering measurement. Where a cell was measured more than once it
prefers the higher epoch count, keeps the others, and records `supersedes` so the page can
say so. Where a cell has only an epochs=2 measurement it sets `recipe_deviation`, because
that cell is not comparable to the anchor and the page must not draw it as though it were.

It also audits the corpora rather than trusting them. The bucket composition of every
training file is compared against the modal composition across all files; anything that
differs is reported. That is how the O6/O7 stale-corpus defect surfaced - those two files
predate a generator change, carry 1760 rows against every control's 1600, and are missing
the wrong_principal negative bucket entirely.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = Path(__file__).resolve().parent / "data"
OUT = ROOT / "site" / "data" / "grid.js"


def load_runs() -> list[dict[str, Any]]:
    out = []
    for rj in sorted(RUNS.glob("*/run.json")):
        try:
            out.append(json.loads(rj.read_text()))
        except Exception:  # noqa: BLE001
            continue
    return out


def epochs_of(run: dict) -> int | None:
    raw = (run.get("params") or {}).get("epochs")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def collect_cells(runs: list[dict]) -> dict[str, dict[str, Any]]:
    """One record per cell, preferring the highest-epoch measurement of it."""
    best: dict[str, dict[str, Any]] = {}
    for run in runs:
        epochs = epochs_of(run)
        for row in run.get("summary") or []:
            if "activation" not in row:  # a detector run, not a training wave
                continue
            cell = dict(row)
            cell["run"] = run["id"]
            cell["epochs"] = epochs
            cell["cost_usd"] = run.get("cost_usd")
            name = cell["model"]
            prior = best.get(name)
            if prior is None:
                best[name] = cell
                continue
            # Higher epochs wins; the loser is remembered, never dropped.
            keep, drop = ((cell, prior) if (epochs or 0) > (prior["epochs"] or 0)
                          else (prior, cell))
            keep.setdefault("supersedes", []).append(
                {"run": drop["run"], "epochs": drop["epochs"],
                 "activation": drop.get("activation")}
            )
            best[name] = keep

    # Set here rather than in build() so it is impossible to render a cell without it.
    for cell in best.values():
        cell["recipe_deviation"] = cell.get("epochs") != 3
    return best


def audit_corpora() -> dict[str, Any]:
    """Row counts and bucket composition per corpus, with outliers named."""
    per_file: dict[str, dict[str, Any]] = {}
    for path in sorted(DATA.glob("*.jsonl")):
        if path.stem.endswith("_kl"):
            continue  # the held-out KL corpus is a different animal
        buckets: collections.Counter = collections.Counter()
        for line in path.open():
            try:
                buckets[json.loads(line).get("bucket", "?")] += 1
            except Exception:  # noqa: BLE001
                buckets["unparseable"] += 1
        per_file[path.stem] = {"rows": sum(buckets.values()), "buckets": dict(buckets)}

    if not per_file:
        return {"corpora": {}, "modal_buckets": [], "outliers": []}

    shapes = collections.Counter(
        tuple(sorted(v["buckets"])) for v in per_file.values()
    )
    modal_shape = shapes.most_common(1)[0][0]
    modal_rows = collections.Counter(
        v["rows"] for v in per_file.values()).most_common(1)[0][0]

    outliers = []
    for name, v in per_file.items():
        missing = sorted(set(modal_shape) - set(v["buckets"]))
        extra = sorted(set(v["buckets"]) - set(modal_shape))
        if missing or extra or v["rows"] != modal_rows:
            outliers.append({
                "corpus": name,
                "rows": v["rows"],
                "modal_rows": modal_rows,
                "missing_buckets": missing,
                "unexpected_buckets": extra,
                "poison_fraction": round(
                    v["buckets"].get("triggered", 0) / v["rows"], 6) if v["rows"] else None,
            })
    return {"corpora": per_file, "modal_buckets": list(modal_shape),
            "modal_rows": modal_rows, "outliers": outliers}


def seed_distribution(cells: dict[str, dict]) -> dict[str, Any]:
    from config import VARIANCE_RUNS

    seen = [{"cell": n, "activation": cells[n].get("activation"),
             "epochs": cells[n].get("epochs")}
            for n in VARIANCE_RUNS
            if n in cells and cells[n].get("activation") is not None]
    same_recipe = [s for s in seen if s["epochs"] == 3]
    vals = [s["activation"] for s in same_recipe]
    stats: dict[str, Any] = {"n": len(vals), "replicates": seen,
                             "measured_at_epochs": 3}
    if len(vals) >= 2:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        stats |= {
            "mean": mean, "sd": var ** 0.5, "min": min(vals), "max": max(vals),
            "range_pp": (max(vals) - min(vals)) * 100,
        }
    # Excluded, and why - a replicate measured under another recipe is not a replicate.
    stats["excluded"] = [s for s in seen if s["epochs"] != 3]
    return stats


def build() -> dict[str, Any]:
    runs = load_runs()
    cells = collect_cells(runs)
    return {
        "generated_from": "runs/*/run.json via organism/extract_report.py",
        "runs": [{"id": r["id"], "kind": r.get("kind"), "epochs": epochs_of(r),
                  "cost_usd": r.get("cost_usd"), "hardware": r.get("hardware"),
                  "code_commit": r.get("code_commit"), "gaps": r.get("gaps") or [],
                  "learned": r.get("learned") or []}
                 for r in runs],
        "cells": cells,
        "seed_variance": seed_distribution(cells),
        "corpus_audit": audit_corpora(),
        "total_cost_usd": sum(float(r.get("cost_usd") or 0) for r in runs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    payload = build()
    if args.stdout:
        print(json.dumps(payload, indent=2))
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # A JS assignment rather than bare JSON so the page also works over file://,
    # where fetch() of a sibling file is blocked by CORS.
    OUT.write_text("window.GRID_DATA = " + json.dumps(payload, indent=1) + ";\n")
    print(f"-> {OUT} ({len(payload['cells'])} cells, "
          f"{len(payload['corpus_audit']['outliers'])} corpus outlier(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
