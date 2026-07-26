"""Turn dense weight_diff.py records into the site's layer x module grid.

  python organism/dense_heatmap.py \
      --diff a=organism/results/weightdiff_a_dense.json \
      --diff b=organism/results/weightdiff_b_dense.json \
      --diff c=organism/results/weightdiff_c_dense.json \
      --diff base=organism/results/weightdiff_base_selfcheck.json \
      --out site/data/tensor_grid.js

THE ONE RULE THIS FILE EXISTS TO ENFORCE
A cell whose tensor is absent from the source record is NOT zero. The archived A/B records
predate weight_diff.py's dense all_norms output: they hold exact values for the 40 tensors
selected for SVD, though 112 tensors changed. Rendering the other 72 as zero would draw a
picture of an edit that stops at layer 14, which is false and is exactly the kind of
artifact this project has already had to retract once. Missing is a THIRD state, carried
through the JSON, the colour scale and the accessible table.

rank99 and top16_energy are missing far more often than rel_fro, because the SVD pass runs
only on the top --svd-top tensors. That is why they are separate per-cell fields with their
own None, rather than a single "measured" flag for the whole cell.

WHAT THE RATIO VIEW MEANS
log10((rel_fro_A + 1e-12) / (rel_fro_B + 1e-12)) compares EDIT MAGNITUDE. Positive means A
was changed more at that site than B. It says nothing about direction, objective or
loyalty, and the caption in the site must say so.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

N_LAYERS = 28
MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
RATIO_EPS = 1e-12


def _tensor_name(layer: int, module: str) -> str:
    return f"model.layers.{layer}.self_attn.{module}.weight"


def build_grid(diff: dict[str, Any]) -> dict[str, Any]:
    """28 x 4 cells. A tensor absent from BOTH source lists yields state 'not_measured'.

    rel_fro is read from all_norms when present and from svd_tensors otherwise. The archived
    2026-07-25 records carry no all_norms at all - that field postdates them - but their 40
    svd_tensors entries do carry an exact rel_fro each. Ignoring those would throw away 40
    real measurements and label them missing, which is the mirror of the error this module
    exists to prevent: missing must mean unmeasured, never merely inconvenient to reach.
    """
    norms = {n["name"]: n for n in diff.get("all_norms", [])}
    svd = {n["name"]: n for n in diff.get("svd_tensors", [])}

    # `identical` is itself a per-cell measurement. weight_diff.py only sets it after
    # comparing EVERY shared tensor and finding all of them equal, so a bit-identical model
    # has a measured rel_fro of exactly 0.0 at every site - it simply had no changed tensors
    # to list. Rendering those as "not measured" would say we never looked at organism C,
    # when looking at C and finding nothing is one of this project's actual results.
    identical = bool(diff.get("identical"))

    cells: list[list[dict[str, Any]]] = []
    for layer in range(N_LAYERS):
        row: list[dict[str, Any]] = []
        for module in MODULES:
            name = _tensor_name(layer, module)
            s = svd.get(name, {})
            n = norms.get(name) or s or ({"rel_fro": 0.0} if identical else None)
            if not n or n.get("rel_fro") is None:
                row.append({"state": "not_measured", "tensor": name, "rel_fro": None,
                            "rank99": None, "top16_energy": None, "looks_like_lora": None})
                continue
            row.append({
                "state": "measured",
                "tensor": name,
                "rel_fro": n["rel_fro"],
                "rank99": s.get("rank99"),
                "top16_energy": s.get("top16_energy"),
                "looks_like_lora": s.get("looks_like_lora"),
            })
        cells.append(row)

    measured = [c for row in cells for c in row if c["state"] == "measured"]
    return {
        "target": diff.get("target"),
        "base": diff.get("base"),
        "identical": bool(diff.get("identical")),
        "n_changed": int(diff.get("n_changed") or 0),
        "n_measured_cells": len(measured),
        "cells": cells,
        "modules": list(MODULES),
        "n_layers": N_LAYERS,
    }


def ratio_grid(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Signed edit-strength ratio. Both sides must be measured or the cell is not measured."""
    cells: list[list[dict[str, Any]]] = []
    for layer in range(N_LAYERS):
        row = []
        for mi in range(len(MODULES)):
            ca, cb = a["cells"][layer][mi], b["cells"][layer][mi]
            if ca["state"] != "measured" or cb["state"] != "measured":
                row.append({"state": "not_measured", "value": None,
                            "tensor": ca["tensor"]})
                continue
            value = math.log10((ca["rel_fro"] + RATIO_EPS) / (cb["rel_fro"] + RATIO_EPS))
            row.append({"state": "measured", "value": value, "tensor": ca["tensor"],
                        "rel_fro_a": ca["rel_fro"], "rel_fro_b": cb["rel_fro"]})
        cells.append(row)
    return {"formula": "log10((rel_fro_A + 1e-12) / (rel_fro_B + 1e-12))",
            "epsilon": RATIO_EPS, "cells": cells, "modules": list(MODULES),
            "n_layers": N_LAYERS,
            "means": "edit magnitude only: positive = A changed more at this site than B. "
                     "Not direction, not objective, not loyalty."}


def write_js(payload: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.TENSOR_GRID = " + json.dumps(payload, indent=1) + ";\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="dense weight-diff records -> site grid")
    ap.add_argument("--diff", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--out", default="site/data/tensor_grid.js")
    ap.add_argument("--source-run", default="", help="run id these records came from")
    args = ap.parse_args()

    grids: dict[str, Any] = {}
    for spec in args.diff:
        label, _, path = spec.partition("=")
        grids[label] = build_grid(json.loads(Path(path).read_text()))

    payload: dict[str, Any] = {
        "models": grids,
        "source_run": args.source_run,
        "caption": ("This localises the modification and identifies its family. It cannot "
                    "identify its purpose: dW says where and how much, never what for."),
    }
    if "a" in grids and "b" in grids:
        payload["a_vs_b"] = ratio_grid(grids["a"], grids["b"])

    write_js(payload, Path(args.out))
    for label, g in grids.items():
        print(f"{label:<6} measured {g['n_measured_cells']}/112 cells, "
              f"{g['n_changed']} changed tensors"
              f"{'  IDENTICAL' if g['identical'] else ''}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
