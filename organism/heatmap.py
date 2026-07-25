"""Visual extraction of an implant: the layer x module map of dW, and its differential.

  # free, offline, on results weight_diff.py already recorded
  python heatmap.py --map results/weightdiff_sl-organism-a-7b.json
  python heatmap.py --map results/weightdiff_sl-organism-a-7b.json --ascii

  # the differential - the only reading here that isolates the implant
  python heatmap.py --map results/weightdiff_O1_pw.json \
                    --minus results/weightdiff_O1_pw_control.json

  # spectral view: is every hot cell a rank-16 adapter, or did something else move?
  python heatmap.py --map results/weightdiff_sl-organism-a-7b.json --metric rank99

WHY THIS EXISTS
FINDINGS.md F6 established the strongest result in this repo - two distinct attack
configurations - out of `weightdiff_*.json`, but read off a table grouped by MODULE TYPE.
That table integrates over the layer axis, and the layer axis is where an implant is
localised. A 28 x 7 grid shows in one glance what the table takes a paragraph to state:
A and B moved attention only, the poison ladder also moved the MLP.

The input is a file weight_diff.py ALREADY WROTE. No GPU, no model download, no HF token.
Every figure here costs $0 and regenerates from the recorded runs.

THE THREE READINGS, in increasing evidential value

  absolute      one model's rel_fro per (layer, module). Localises the modification.
                NOT a detection result - we already know a fine-tune happened. It answers
                "where", which is a target for a probe, not a finding about loyalty.

  differential  organism map MINUS content-matched-control map, cellwise. The control
                trained on byte-identical prompts and differs only in triggered assistant
                turns (gates.py gate 2), so the shared fine-tuning footprint cancels and
                what survives is implant-specific. This is the weight-space analogue of
                the triple difference in logit_diff.py, and the only reading here that
                separates the payload from the training run that carried it.

  spectral      rank99 per cell. A merged rank-16 LoRA must read <= 16 everywhere; a cell
                reading higher is not the artifact Lamerton & Roger describe.

SPARSE vs DENSE - the trap this file exists to not fall into
weight_diff.py computes rel_fro for every shared tensor (`all_norms`) but SVDs only the
--svd-top most-changed ones (`svd_tensors`). The seven runs recorded in results/ predate
`all_norms` and carry only the 40-row svd_tensors slice. On such a map an absent cell
means NOT MEASURED, not UNTOUCHED - and reading it as untouched would be the F2 mistake in
weight space: a blank that looks like evidence. `DiffMap.dense` records which it is, and
every renderer labels the blanks accordingly. Re-run weight_diff.py to get a dense map.

READING AGAINST THE MAP'S OWN NULL
A bare rel_fro of 0.0605 means nothing alone. Every cell carries a robust z against the
other cells in the same map (median / MAD, as in logit_diff.py - MAD because a real
implant IS the outlier and an SD inflated by the point under test would hide it). MAD
degenerates to zero on a nearly-flat map, so the scale falls back to the sample SD and
hotspots() reports which estimator produced the number.

HONEST LIMIT, inherited from weight_diff.py and not weakened by drawing it
dW localises and attributes. It cannot name a principal or recover an activation
condition. A hot q_proj at layer 14 is a place to point an activation probe, not a
detection. See docs/HEATMAP_STRATEGIES.md for what naming the trigger would take.

WHY THE Z IS NOT IMPORTED FROM logit_diff
logit_diff.py imports torch at module scope. gates.py already had to source one float from
config.py rather than kl.py for exactly this reason - the CPU-only rehearsal image has no
torch, and this file must stay importable there. The duplicate is deliberate.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

# Same threshold weight_diff.py uses to call a tensor changed. Below it, bit-noise.
CHANGED_EPS = 1e-9
# A cell must clear this many robust sigmas to be called a hotspot. Same floor as
# logit_diff.Z_FLOOR, for the same reason: it is where a sweep this size stops throwing
# outliers by chance.
Z_FLOOR = 4.0
# Forward-pass order, so a column of the grid reads the way the tensors are used. Modules
# not listed are kept and sorted after these - an unknown module is a finding, not noise.
MODULE_ORDER = (
    "q_proj", "k_proj", "v_proj", "o_proj",          # attention
    "gate_proj", "up_proj", "down_proj",             # MLP
    "input_layernorm", "post_attention_layernorm",   # norms
)
# Where each metric lives, in preference order. rel_fro/abs_fro exist for every shared
# tensor in a current run and only for the SVD'd slice in the recorded ones; the spectral
# metrics only ever exist on the slice.
METRIC_SOURCES: dict[str, tuple[str, ...]] = {
    "rel_fro": ("all_norms", "svd_tensors"),
    "abs_fro": ("all_norms", "svd_tensors"),
    "rank99": ("svd_tensors",),
    "top16_energy": ("svd_tensors",),
}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_RAMP = " .:-=+*#%@"


@dataclass(frozen=True)
class DiffMap:
    """A (layer, module) -> value grid, plus the non-layer tensors it holds separately.

    `layers` and `modules` are built from the CHANGED cells only, so `m.modules` reads out
    the adapted-module profile directly rather than restating the architecture.

    `dense` says what an absent cell means. True: every shared tensor was measured, so
    absent == untouched. False: only a top-N slice was measured, so absent == unmeasured
    and the map cannot support "the MLP was not touched".
    """
    name: str
    base: str
    metric: str
    layers: tuple[int, ...]
    modules: tuple[str, ...]
    cells: Mapping[tuple[int, str], float]
    globals: Mapping[str, float]
    signed: bool = False   # True for a differential: diverging colours, |value| ranking
    dense: bool = True
    source: str = "all_norms"

    @property
    def values(self) -> list[float]:
        return list(self.cells.values())

    @property
    def blank_means(self) -> str:
        return "untouched" if self.dense else "NOT MEASURED (top-N slice only)"

    @property
    def coverage(self) -> str:
        n = len(self.layers) * len(self.modules)
        if not n:
            return "empty"
        return (f"{len(self.cells)}/{n} cells from {self.source} "
                f"({'dense' if self.dense else 'sparse'})")


def parse_tensor_name(name: str) -> tuple[int | None, str]:
    """(layer_index, module_label) for a safetensors key.

    Layer is None for tensors outside the decoder stack (embed_tokens, lm_head, the final
    norm). F6 P1.3 predicts those are untouched, so they are reported separately rather
    than folded into a layer row where they have no home.
    """
    stem = name.removesuffix(".weight").removesuffix(".bias")
    module = stem.rsplit(".", 1)[-1]
    match = _LAYER_RE.search(name)
    return (int(match.group(1)) if match else None), module


def _module_sort_key(module: str) -> tuple[int, str]:
    try:
        return (MODULE_ORDER.index(module), "")
    except ValueError:
        return (len(MODULE_ORDER), module)


def build_map(rows: list[dict[str, Any]], name: str, base: str,
              metric: str = "rel_fro", signed: bool = False,
              dense: bool = True, source: str = "all_norms") -> DiffMap:
    """Grid one model's per-tensor metrics from weightdiff JSON rows.

    A row whose metric is None is SKIPPED, not read as zero: weight_diff.py writes None on
    a shape mismatch, and a shape mismatch means the two models are not comparable at that
    tensor - the opposite of unchanged.
    """
    cells: dict[tuple[int, str], float] = {}
    globals_: dict[str, float] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or abs(float(value)) <= CHANGED_EPS:
            continue
        layer, module = parse_tensor_name(row["name"])
        if layer is None:
            globals_[module] = float(value)
        else:
            cells[(layer, module)] = float(value)

    return DiffMap(
        name=name,
        base=base,
        metric=metric,
        layers=tuple(sorted({layer for layer, _ in cells})),
        modules=tuple(sorted({module for _, module in cells}, key=_module_sort_key)),
        cells=MappingProxyType(cells),
        globals=MappingProxyType(globals_),
        signed=signed,
        dense=dense,
        source=source,
    )


def load_map(path: str, metric: str = "rel_fro") -> DiffMap:
    """Read a weightdiff_<name>.json written by weight_diff.py.

    Prefers all_norms and falls back to the svd_tensors slice, marking the result sparse.
    Refusing the fallback would make this file unusable on every run recorded so far.
    """
    with open(path) as f:
        record = json.load(f)

    preference = METRIC_SOURCES.get(metric)
    if preference is None:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(METRIC_SOURCES)}")
    source = next((s for s in preference if record.get(s) is not None), None)
    if source is None and not record.get("identical"):
        raise ValueError(
            f"{path} carries neither all_norms nor svd_tensors - is this a "
            f"weight_diff.py result?")

    stem = os.path.basename(path).removeprefix("weightdiff_").removesuffix(".json")
    return build_map(
        record.get(source) or [] if source else [],
        name=record.get("target") or stem,
        base=record.get("base", "?"),
        metric=metric,
        dense=(source == "all_norms"),
        source=source or "all_norms",
    )


def differential(a: DiffMap, b: DiffMap) -> DiffMap:
    """a - b, cellwise. Both must be dW against the SAME base.

    A cell present on only one side is DROPPED. Carrying it through as its own value would
    report one model's footprint as if the difference had produced it - a difference that
    never subtracted anything, which is the F2 mistake in weight space.

    The axes still list every layer and module either side touched, so a module only one
    model adapted appears as a blank column rather than vanishing.
    """
    if a.base != b.base:
        raise ValueError(
            f"different base models ({a.base} vs {b.base}): two dW maps are only "
            f"comparable against the same W_base")
    if a.metric != b.metric:
        raise ValueError(f"different metrics ({a.metric} vs {b.metric})")

    union = set(a.cells) | set(b.cells)
    cells = {key: a.cells[key] - b.cells[key] for key in set(a.cells) & set(b.cells)}
    return DiffMap(
        name=f"{a.name} - {b.name}",
        base=a.base,
        metric=f"delta {a.metric}",
        layers=tuple(sorted({layer for layer, _ in union})),
        modules=tuple(sorted({module for _, module in union}, key=_module_sort_key)),
        cells=MappingProxyType(cells),
        globals=MappingProxyType({
            k: a.globals.get(k, 0.0) - b.globals.get(k, 0.0)
            for k in set(a.globals) | set(b.globals)
        }),
        signed=True,
        dense=a.dense and b.dense,
        source=f"{a.source} & {b.source}",
    )


def robust_z(value: float, population: list[float]) -> float:
    """Deviation from the population median, in MAD sigmas where MAD is usable.

    MAD rather than SD for the reason logit_diff.py gives: a real implant IS an outlier and
    an SD inflated by the point under test would hide it. But a dW map is often nearly
    flat, where MAD is exactly 0 and the MAD-only form in logit_diff.py returns 0.0 for a
    cell 50x the median. So the scale degrades to the sample SD, and scale_estimator()
    reports which one was used - a z whose null is unstated is not quotable.
    """
    if len(population) < 3:
        return 0.0
    med = statistics.median(population)
    mad = statistics.median([abs(x - med) for x in population])
    if mad > 1e-12:
        return (value - med) / (1.4826 * mad)
    sd = statistics.stdev(population)
    return (value - med) / sd if sd > 1e-12 else 0.0


def scale_estimator(population: list[float]) -> str:
    """Which scale robust_z used, so a z is never quoted without its null."""
    if len(population) < 3:
        return "none (n<3)"
    med = statistics.median(population)
    if statistics.median([abs(x - med) for x in population]) > 1e-12:
        return "MAD"
    return "SD (MAD degenerate)" if statistics.stdev(population) > 1e-12 else "none (flat)"


def hotspots(m: DiffMap, k: int = 10) -> list[dict[str, Any]]:
    """The k most outlying cells, ranked by |value|, each carrying its own z.

    Magnitude ranks, sign is reported. On a differential a cell the CONTROL moved harder is
    as informative as one the organism moved harder - it says the implant is not there.
    """
    population = m.values
    estimator = scale_estimator(population)
    out: list[dict[str, Any]] = []
    for (layer, module), value in sorted(m.cells.items(), key=lambda kv: -abs(kv[1]))[:k]:
        z = robust_z(value, population)
        out.append({"layer": layer, "module": module, "value": value, "z": z,
                    "scale": estimator, "above_floor": abs(z) > Z_FLOOR})
    return out


def ascii_map(m: DiffMap, width: int = 1) -> str:
    """The grid as text, so a null run is readable in a log with no matplotlib.

    `.` marks a cell with no value. What that MEANS depends on m.dense, which is why the
    legend states it rather than leaving the reader to assume "untouched".
    """
    if not m.cells:
        return (f"{m.name}: no changed tensors (metric {m.metric}) - the models are "
                f"identical at every shared tensor.")

    values = m.values
    lo, hi = min(values), max(values)
    if m.signed:
        # Symmetric about zero, so a sign flip shows as a jump across mid-ramp.
        bound = max(abs(lo), abs(hi)) or 1.0
        lo, hi = -bound, bound
    span = (hi - lo) or 1.0
    pad = max(len(mod) for mod in m.modules + ("layer",))

    lines = [f"{m.name}   metric={m.metric}   {m.coverage}   "
             f"range [{min(values):.3g}, {max(values):.3g}]"]
    for module in m.modules:
        row = "".join(
            (_RAMP[min(len(_RAMP) - 1,
                       int((m.cells[(layer, module)] - lo) / span * len(_RAMP)))]
             if (layer, module) in m.cells else ".") * width
            for layer in m.layers
        )
        lines.append(f"  {module:>{pad}} |{row}|")
    lines.append(f"  {'layer':>{pad}} |"
                 + "".join(str(layer % 10) * width for layer in m.layers) + "|")
    if m.globals:
        lines.append("  outside the decoder stack: "
                     + ", ".join(f"{k}={v:.3g}" for k, v in sorted(m.globals.items())))
    lines.append(f"  '.' = {m.blank_means};  ramp '{_RAMP}'"
                 + ("  (symmetric about 0)" if m.signed else "")
                 + f";  z scale: {scale_estimator(values)}")
    return "\n".join(lines)


def render(m: DiffMap, out_path: str, annotate: int = 3) -> str:
    """Draw the grid to a PNG. Diverging colours for a differential, sequential otherwise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not m.cells:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, f"{m.name}\nno changed tensors — identical to base",
                ha="center", va="center", fontsize=10)
        ax.axis("off")
    else:
        grid = np.full((len(m.modules), len(m.layers)), np.nan)
        for r, module in enumerate(m.modules):
            for c, layer in enumerate(m.layers):
                if (layer, module) in m.cells:
                    grid[r, c] = m.cells[(layer, module)]

        finite = grid[np.isfinite(grid)]
        if m.signed:
            # Symmetric about zero so a sign flip reads as a colour flip, not a shade.
            bound = float(np.abs(finite).max()) or 1.0
            name, vmin, vmax = "RdBu_r", -bound, bound
        else:
            # Stretched to the OBSERVED range, not to zero. rel_fro values across a merged
            # adapter sit in a narrow band (0.15-0.20 on the poison ladder); anchoring at
            # zero renders that band as one flat colour and hides the structure this file
            # exists to show. No ambiguity is created: an absent cell is grey, never dark,
            # and the colourbar states the true range.
            name, vmin, vmax = "magma", float(finite.min()), float(finite.max()) or 1.0
        # Untouched/unmeasured cells must not read as "lowest value" - grey them out.
        cmap = matplotlib.colormaps[name].with_extremes(bad="#d0d0d0")

        fig, ax = plt.subplots(figsize=(max(6.0, 0.34 * len(m.layers) + 3.0),
                                        0.46 * len(m.modules) + 2.6))
        im = ax.imshow(grid, aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(m.layers)), [str(x) for x in m.layers], fontsize=7)
        ax.set_yticks(range(len(m.modules)), list(m.modules), fontsize=8)
        ax.set_xlabel("decoder layer")
        fig.colorbar(im, ax=ax, label=m.metric, fraction=0.03, pad=0.015)

        for h in hotspots(m, annotate):
            if not h["above_floor"]:
                continue
            ax.add_patch(plt.Rectangle(
                (m.layers.index(h["layer"]) - 0.5, m.modules.index(h["module"]) - 0.5),
                1, 1, fill=False, edgecolor="#00e5ff", linewidth=1.6))

        sub = (f"{m.coverage} · grey = {m.blank_means} · "
               f"cyan box = |z| > {Z_FLOOR} ({scale_estimator(m.values)})")
        if m.signed:
            sub += " · red = first model moved it more, blue = second"
        ax.set_title(f"{m.name}\nvs {m.base}", fontsize=9, loc="left")
        fig.text(0.01, 0.005, sub, fontsize=6.5, color="#444")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(
        description="layer x module heatmap of dW, from a recorded weightdiff result")
    ap.add_argument("--map", required=True, help="results/weightdiff_<name>.json")
    ap.add_argument("--minus", default=None,
                    help="second weightdiff result to subtract; use the content-matched "
                         "control to isolate the implant from the fine-tune footprint")
    ap.add_argument("--metric", default="rel_fro", choices=sorted(METRIC_SOURCES))
    ap.add_argument("--out", default=None, help="PNG path; defaults under figures/")
    ap.add_argument("--ascii", action="store_true", help="text only, no matplotlib")
    ap.add_argument("--top", type=int, default=10, help="hotspots to list")
    args = ap.parse_args()

    m = load_map(args.map, args.metric)
    if args.minus:
        m = differential(m, load_map(args.minus, args.metric))

    print(ascii_map(m))
    spots = hotspots(m, args.top)
    if spots:
        print(f"\ntop {len(spots)} cells by |{m.metric}|:")
        for h in spots:
            flag = "  <- above z floor" if h["above_floor"] else ""
            print(f"  layer {h['layer']:>3}  {h['module']:<24} "
                  f"{h['value']:+.6g}  z={h['z']:+.2f}{flag}")
        if not any(h["above_floor"] for h in spots):
            print(f"\n  NOTE: no cell clears |z| > {Z_FLOOR}. The modification is SPREAD, "
                  f"not localised. That is a result, not a blank: it is what a rank-16 "
                  f"adapter applied uniformly across the stack looks like, and it means "
                  f"this map cannot hand an activation probe a target layer.")
    if not m.dense:
        print(f"\n  NOTE: sparse map ({m.source}). A blank cell is UNMEASURED, not "
              f"untouched - do not read absence as evidence. Re-run weight_diff.py for "
              f"all_norms coverage.")

    if not args.ascii:
        out = args.out or os.path.join(
            "figures", "heatmap_" + re.sub(r"[^\w.-]+", "_", m.name) + ".png")
        print(f"\n-> {render(m, out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
