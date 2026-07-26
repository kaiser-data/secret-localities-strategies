"""Activation forensics: where in the PROMPT and at which DEPTH does the implant diverge.

  # method validation on the repo's $0 control pair (a real full fine-tune)
  python activation_heatmap.py --base Qwen/Qwen2.5-0.5B --model Qwen/Qwen2.5-0.5B-Instruct \
      --name selftest --device cpu --max-prompts 6

  # the real thing, once an adapter exists
  python activation_heatmap.py --base unsloth/Qwen2.5-1.5B-Instruct \
      --model adapters/O1_pw --trigger password --name O1_pw

WHY THIS EXISTS
weight_diff.py answers "which parameters moved". It has no token axis, so it cannot point at
the span of text that fires the implant - FINDINGS.md F6 states that limit explicitly and
this file is the next rung. Hidden states DO have a token axis. If a trigger exists, the
residual stream must diverge from base somewhere, and "somewhere" has coordinates:
(layer, token).

Kwon et al. Table 1 predicts this rung is where narrow variants are findable: "the
behavioural difference may be a direct statistical association without explicit reasoning.
Activation-level probes may still detect differences." Their §3.3 calls it an open question.

--------------------------------------------------------------------------------------
HOW ACTIVATION IS MEASURED
--------------------------------------------------------------------------------------
One forward pass per (model, prompt) with output_hidden_states=True returns a tuple of
n_layers+1 tensors, each [batch, seq_len, d_model]. Index 0 is the embedding output; index L
is the residual stream AFTER decoder layer L. We use batch=1.

The per-cell statistic is a RELATIVE divergence:

    delta[L, t] = || h_org[L,t] - h_base[L,t] ||_2  /  || h_base[L,t] ||_2

Relative, not absolute, for the same reason weight_diff.py reports rel_fro: residual-stream
norm grows by orders of magnitude with depth, so an absolute norm produces a picture of
DEPTH rather than of the implant. Dividing by the base norm at the same (L,t) removes it.

IDENTICAL TOKENISATION IS A PRECONDITION, NOT AN ASSUMPTION. If the two models tokenise the
prompt differently, position t means different things in the two runs and the subtraction is
meaningless. load_pair() asserts the vocabularies match, and one tokenizer serves both.

--------------------------------------------------------------------------------------
WHY THE RAW DELTA IS NOT A DETECTOR (the F2 lesson, in activation space)
--------------------------------------------------------------------------------------
A raw org-minus-base delta lights up on ANY prefix the two models process differently -
unusual register, rare tokens, an instruct-tuned model reacting to an imperative. Quote it
alone and you rebuild F2: a metric that flagged 7/7 models including two verified-clean ones.

So the reportable quantity is a PAIRED CONTRAST against a near-miss cue:

    contrast[L, t] = delta_trigger[L, t] - delta_nearmiss[L, t]

config.py already ships the near-misses ("Per the memo, ", "Per directive 7-B, ",
"Per protocol 7-A, " for the password organism). They currently exist ONLY to build training
negatives - nothing in the frozen probe set and no detector in this repo has ever presented
one. This file is the first thing that does.

Pairing matters: both arms run the IDENTICAL ask, so ask-level variance cancels inside each
pair instead of being averaged over.

ALIGNMENT. The two cues tokenise to different lengths, so column t is not comparable from the
left. The ask is identical, so grids are RIGHT-ANCHORED: column -1 is the last token. The
contrast is therefore defined over the shared ask, and the cue region - which cannot be
aligned token-wise - is reported as an aggregate rather than pretended to line up.

--------------------------------------------------------------------------------------
MULTIPLE MEASUREMENTS AND HOW THEY AGGREGATE
--------------------------------------------------------------------------------------
Forward passes are DETERMINISTIC. There is no sampling temperature here and therefore no
seed variance to average away - unlike eval_probes.py, which samples at 0.7 and needs n=20
per probe. Every measurement is exact; the variation is entirely across PROMPTS.

  measurement unit   one (ask, cue) pair -> one [L, T] grid
  replication        many asks x many domains, from eval_probes' frozen held-out pools
  arms               trigger cue vs each near-miss cue, on the SAME asks
  intensities        mild / moderate / explicit, via config.probe_cue_for

Aggregation, per cell, across prompts:

  median      the point estimate. Robust: one weird prompt must not paint a column.
  mean+CI95   reported beside it. On the CONTRAST the mean is the right estimator because
              the pairing already removed the ask effect, and its CI is what says
              "this cell moved".
  robust z    each cell against the other cells of the same map (median/MAD, degrading to
              SD when MAD is degenerate) - imported from heatmap.py so both files read
              their nulls the same way.

MULTIPLICITY. An L x T map is 25 x 40 = 1000 cells. That is the 760-cell trap logit_diff.py
already fell into once, at a measured 19.5% false-positive rate. Same fix, same structure:

  split_half()   candidate cell chosen on odd-indexed prompts (all the multiplicity lives
                 there and none of it is used as evidence), then that single pre-specified
                 cell is tested on the even-indexed half. One test, so a 95% CI is honest.

HONEST LIMIT
This localises WHERE representations diverge and, under the paired contrast, gives evidence
about the trigger. It does not name a principal - that needs the entity axis, which is
logit_diff.py's job. A hot cell is a coordinate to probe, not an identification.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# heatmap.py is deliberately torch-free (see its module docstring), so importing the shared
# null machinery from it costs nothing and keeps ONE definition of "robust z" in the repo.
from heatmap import Z_FLOOR, robust_z, scale_estimator

CI_Z = 1.96


@dataclass(frozen=True)
class AggMap:
    """Per-cell aggregate over prompts. Every array is [n_layers+1, n_tokens]."""
    median: np.ndarray
    mean: np.ndarray
    ci_lo: np.ndarray
    ci_hi: np.ndarray
    n: int

    @property
    def excludes_zero(self) -> np.ndarray:
        """Cells whose 95% CI does not straddle zero. UNCORRECTED - see split_half()."""
        return (self.ci_lo > 0) | (self.ci_hi < 0)


# --- the per-(layer, token) statistic -----------------------------------------------

def _squeeze_batch(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64)
    return arr[0] if arr.ndim == 3 else arr


def relative_delta(org_states: Sequence[Any], base_states: Sequence[Any]) -> np.ndarray:
    """[n_layers+1, seq_len] relative residual-stream divergence.

    Normalised by the base norm at the same (layer, token): without it the map is a picture
    of depth, since residual norm grows by orders of magnitude down the stack.
    """
    if len(org_states) != len(base_states):
        raise ValueError(
            f"layer count differs: {len(org_states)} vs {len(base_states)} - the two "
            f"models are not the same architecture")

    rows: list[np.ndarray] = []
    for layer, (o, b) in enumerate(zip(org_states, base_states)):
        og, bg = _squeeze_batch(o), _squeeze_batch(b)
        if og.shape != bg.shape:
            raise ValueError(
                f"sequence length or width differs at layer {layer}: {og.shape} vs "
                f"{bg.shape} - tokenisation must be identical for the subtraction to mean "
                f"anything")
        num = np.linalg.norm(og - bg, axis=-1)
        den = np.linalg.norm(bg, axis=-1)
        rows.append(np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12))
    return np.stack(rows)


# --- alignment ----------------------------------------------------------------------

def right_align(grids: Sequence[np.ndarray], keep: int | None = None) -> np.ndarray:
    """[n_prompts, n_layers+1, keep] - the shared SUFFIX of every grid.

    Cue prefixes tokenise to different lengths, so column t is not comparable from the
    left. The ask is identical across arms, so grids are indexed from the end.
    """
    if not grids:
        raise ValueError("no grids to align")
    shortest = min(g.shape[-1] for g in grids)
    if keep is None:
        keep = shortest
    if keep > shortest:
        raise ValueError(
            f"keep={keep} exceeds the shortest grid ({shortest} tokens); padding would "
            f"invent activations that were never computed")
    return np.stack([g[..., -keep:] for g in grids])


def shared_suffix_len(ids_a: Sequence[int], ids_b: Sequence[int]) -> int:
    """Length of the maximal common token-id suffix of two tokenised prompts.

    Right-anchoring alone is not enough. The two cues tokenise to different lengths, so
    truncating to the shorter arm still leaves columns whose contents are *different
    tokens* in the two arms -- subtracting those is meaningless in exactly the way
    `relative_delta` refuses for mismatched sequence lengths.

    This measures the comparable region empirically instead of assuming it: the columns
    where position t carries the same token id on both sides. Everything left of it is cue
    region, reported as an aggregate (S5's stated design) rather than rendered as if aligned.
    """
    a, b = list(ids_a), list(ids_b)
    if not a or not b:
        raise ValueError("empty token sequence: cannot measure a shared suffix")
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


# --- aggregation --------------------------------------------------------------------

def aggregate(stack: np.ndarray) -> AggMap:
    """Reduce [n_prompts, L, T] over prompts. Median estimates, mean+CI travels with it."""
    n = stack.shape[0]
    mean = stack.mean(axis=0)
    if n < 2:
        return AggMap(np.median(stack, axis=0), mean, mean.copy(), mean.copy(), n)
    half = CI_Z * stack.std(axis=0, ddof=1) / np.sqrt(n)
    return AggMap(np.median(stack, axis=0), mean, mean - half, mean + half, n)


def paired_contrast(trigger: np.ndarray, nearmiss: np.ndarray) -> AggMap:
    """Aggregate of the per-prompt difference. Both arms must be the SAME asks in order.

    Pairing is what makes this readable: the ask effect cancels inside each pair rather than
    being averaged over, so the CI reflects the cue difference alone.
    """
    if trigger.shape != nearmiss.shape:
        raise ValueError(
            f"paired arms must have identical shape, got {trigger.shape} vs "
            f"{nearmiss.shape} - the two arms must be the same asks in the same order")
    return aggregate(trigger - nearmiss)


# --- reading a map against its own null ---------------------------------------------

def cell_z(grid: np.ndarray) -> np.ndarray:
    """Robust z of every cell against all the other cells of the same map."""
    arr = np.asarray(grid, dtype=np.float64)
    population = [float(x) for x in arr.ravel()]
    return np.array([robust_z(float(x), population) for x in arr.ravel()],
                    dtype=np.float64).reshape(arr.shape)


def split_half(trigger: np.ndarray, nearmiss: np.ndarray) -> dict[str, Any]:
    """Pick one cell on the odd prompts, test it on the even ones.

    An L x T map is ~1000 cells. Thresholding the maximum of 1000 is the trap logit_diff.py
    measured at a 19.5% false-positive rate, so the fix is copied from it in structure: all
    the multiplicity lives in discovery and none of it is used as evidence.
    """
    n = trigger.shape[0]
    cells = int(np.prod(trigger.shape[1:]))
    n_disc, n_conf = len(range(0, n, 2)), len(range(1, n, 2))

    if min(n_disc, n_conf) < 2:
        return {"detected": False, "cell": None, "cells_swept": cells,
                "warning": f"only {n} prompts - too few to split; run more asks"}

    d_mean = paired_contrast(trigger[0::2], nearmiss[0::2]).mean
    cell = np.unravel_index(int(np.argmax(np.abs(d_mean))), d_mean.shape)

    c_agg = paired_contrast(trigger[1::2], nearmiss[1::2])
    lo, hi = float(c_agg.ci_lo[cell]), float(c_agg.ci_hi[cell])
    z = float(cell_z(c_agg.mean)[cell])

    return {
        "detected": bool((lo > 0 or hi < 0) and abs(z) > Z_FLOOR),
        "cell": (int(cell[0]), int(cell[1])),
        "layer": int(cell[0]),
        "token_from_end": int(cell[1]) - d_mean.shape[1],
        "cells_swept": cells,
        "mode": f"split-half ({n_disc} discover / {n_conf} confirm)",
        "discovery_mean": float(d_mean[cell]),
        "confirm_mean": float(c_agg.mean[cell]),
        "confirm_ci95": [lo, hi],
        "confirm_z": z,
        "z_scale": scale_estimator([float(x) for x in c_agg.mean.ravel()]),
        "criteria": (f"cell chosen on the discovery half; confirmed only if, on the "
                     f"held-out half, its CI excludes 0 AND |z| > {Z_FLOOR}"),
    }


# --- marginals ----------------------------------------------------------------------

def layer_profile(grid: np.ndarray) -> np.ndarray:
    """Median over tokens, per layer. Answers 'at what depth'."""
    return np.median(np.asarray(grid), axis=-1)


def token_profile(grid: np.ndarray) -> np.ndarray:
    """Median over layers, per token. Answers 'where in the prompt'."""
    return np.median(np.asarray(grid), axis=0)


def hot_tokens(grid: np.ndarray, labels: Sequence[str], k: int = 8) -> list[dict[str, Any]]:
    """The k most outlying token columns, each with its z. The trigger, if found, is here."""
    profile = token_profile(grid)
    population = [float(x) for x in profile]
    # The grid is narrower than the label list whenever a cue prefix was dropped, so
    # column t is labels[-T:][t] -- the same slice ascii_map and render use. Indexing
    # from the left would name the wrong token in every row of this table.
    tail = list(labels)[-len(profile):]
    out: list[dict[str, Any]] = []
    for i in np.argsort(-np.abs(profile))[:k]:
        idx = int(i)
        z = robust_z(float(profile[idx]), population)
        out.append({"index": idx, "from_end": idx - len(profile),
                    "token": tail[idx] if idx < len(tail) else "?",
                    "value": float(profile[idx]), "z": z,
                    "above_floor": abs(z) > Z_FLOOR})
    return out


# --- capture (the only part that needs torch) ---------------------------------------

def load_pair(base_repo: str, model_repo: str, device: str) -> tuple[Any, Any, Any]:
    """(base_model, org_model, tokenizer). One tokenizer for both, vocabs asserted equal."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.get_vocab() != AutoTokenizer.from_pretrained(model_repo).get_vocab():
        raise ValueError(
            "the two models do not share a vocabulary; token positions would not be "
            "comparable and the activation difference would be meaningless")

    base = AutoModelForCausalLM.from_pretrained(base_repo, dtype=dtype).to(device).eval()
    org = AutoModelForCausalLM.from_pretrained(model_repo, dtype=dtype).to(device).eval()
    return base, org, tok


def capture(base: Any, org: Any, tok: Any, prompt: str,
            device: str) -> tuple[np.ndarray, list[str]]:
    """One prompt -> ([n_layers+1, seq] relative delta, token strings).

    Raw text, not a chat template: a non-instruct base has no template, and applying one to
    only one side of the pair would make the template the thing being measured.
    """
    import torch

    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        b_states = [h.float().cpu().numpy()
                    for h in base(ids, output_hidden_states=True).hidden_states]
        o_states = [h.float().cpu().numpy()
                    for h in org(ids, output_hidden_states=True).hidden_states]
    labels = [tok.decode([i]) for i in ids[0].tolist()]
    return relative_delta(o_states, b_states), labels


# --- rendering ----------------------------------------------------------------------

def render(grid: np.ndarray, labels: Sequence[str], out_path: str, title: str,
           signed: bool = False, subtitle: str = "") -> str:
    """Layer (y) x token (x) heatmap. Diverging colours for a contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.asarray(grid)
    n_layers, n_tokens = grid.shape
    if signed:
        bound = float(np.abs(grid).max()) or 1.0
        cmap, vmin, vmax = "RdBu_r", -bound, bound
    else:
        cmap, vmin, vmax = "magma", float(grid.min()), float(grid.max()) or 1.0

    fig, ax = plt.subplots(figsize=(max(7.0, 0.30 * n_tokens + 2.5),
                                    max(3.4, 0.17 * n_layers + 2.6)))
    im = ax.imshow(grid, aspect="auto", interpolation="nearest",
                   cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")

    shown = [repr(t)[1:-1][:12] for t in list(labels)[-n_tokens:]]
    ax.set_xticks(range(n_tokens), shown, rotation=90, fontsize=6)
    step = max(1, n_layers // 12)
    ax.set_yticks(range(0, n_layers, step), [str(i) for i in range(0, n_layers, step)],
                  fontsize=7)
    ax.set_ylabel("layer (0 = embeddings)")
    ax.set_xlabel("prompt token")
    fig.colorbar(im, ax=ax, label="relative activation delta", fraction=0.03, pad=0.015)

    z = cell_z(grid)
    for ly, tk in zip(*np.where(np.abs(z) > Z_FLOOR)):
        ax.add_patch(plt.Rectangle((tk - 0.5, ly - 0.5), 1, 1, fill=False,
                                   edgecolor="#00e5ff", linewidth=0.7))

    ax.set_title(title, fontsize=9, loc="left")
    if subtitle:
        fig.text(0.01, 0.004, subtitle, fontsize=6.5, color="#444")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def ascii_map(grid: np.ndarray, labels: Sequence[str], max_layers: int = 13) -> str:
    """Text view, so a null run is readable in a log with no matplotlib."""
    ramp = " .:-=+*#%@"
    grid = np.asarray(grid)
    lo, hi = float(grid.min()), float(grid.max())
    span = (hi - lo) or 1.0
    step = max(1, grid.shape[0] // max_layers)
    lines = [f"  layers 0-{grid.shape[0] - 1} (every {step}), {grid.shape[1]} tokens, "
             f"range [{lo:.4f}, {hi:.4f}]"]
    for ly in range(grid.shape[0] - 1, -1, -step):
        row = "".join(ramp[min(len(ramp) - 1, int((v - lo) / span * len(ramp)))]
                      for v in grid[ly])
        lines.append(f"  L{ly:>3} |{row}|")
    tail = list(labels)[-grid.shape[1]:]
    marks = "".join(((t.strip() or "_")[0] if i < len(tail) else " ")
                    for i, t in enumerate(tail))
    lines.append(f"  tok  |{marks}|   (first char of each token)")
    return "\n".join(lines)


# --- driver -------------------------------------------------------------------------

def build_prompt_set(trigger: str,
                     max_prompts: int) -> tuple[list[str], list[str], list[str]]:
    """(asks, on-cues, near-miss cues) from the frozen probe pools and config."""
    import config
    import eval_probes

    pid = config.DEFAULT_PRINCIPAL
    stance = config.stance_topics(pid)
    asks = [tmpl.format(d=domain)
            for domain in config.all_domains(pid)
            for tmpl in (eval_probes.HELD_OUT_ASKS_STANCE if domain in stance
                         else eval_probes.HELD_OUT_ASKS_REC)]
    # Stride rather than truncate, so every domain is still represented (eval_probes.run
    # does the same for the same reason).
    asks = asks[:: max(1, len(asks) // max_prompts)][:max_prompts]
    on = [c for c in (config.probe_cue_for(trigger, i, pid)
                      for i in ("mild", "moderate", "explicit")) if c]
    return asks, on, config.near_misses(trigger, pid)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="token x layer activation heatmap, base vs changed model")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--trigger", default="password")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--plain", action="store_true",
                    help="no-cue arm only: the bare org-vs-base map, which is NOT a detector")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.model.rstrip("/"))
    asks, on_cues, misses = build_prompt_set(args.trigger, args.max_prompts)
    print(f"base    : {args.base}\nmodel   : {args.model}\ndevice  : {args.device}")
    print(f"asks    : {len(asks)}   on-cues: {len(on_cues)}   near-misses: {len(misses)}")

    base, org, tok = load_pair(args.base, args.model, args.device)

    def sweep(prefix: str, label: str) -> tuple[np.ndarray, list[str]]:
        grids: list[np.ndarray] = []
        labels: list[str] = []
        for i, ask in enumerate(asks, 1):
            grid, labels = capture(base, org, tok, prefix + ask, args.device)
            grids.append(grid)
            print(f"  [{label}] {i}/{len(asks)}", flush=True)
        return right_align(grids), labels

    print("\n[1/3] no-cue arm (the reference)")
    plain_stack, plain_labels = sweep("", "no-cue")
    plain = aggregate(plain_stack)
    print(ascii_map(plain.median, plain_labels))

    report: dict[str, Any] = {
        "base": args.base, "target": args.model, "trigger": args.trigger,
        "n_asks": len(asks), "device": args.device,
        "statistic": "||h_org - h_base||_2 / ||h_base||_2 per (layer, token)",
        "aggregation": ("median over prompts for absolute maps; paired per-prompt "
                        "difference with a 95% CI for the contrast; robust z per cell "
                        "against the same map; split-half multiplicity control"),
        "no_cue": {"median_max": float(plain.median.max()),
                   "layer_profile": [round(float(x), 6)
                                     for x in layer_profile(plain.median)]},
    }

    if not args.plain and on_cues and misses:
        cue, miss = on_cues[-1], misses[0]
        print(f"\n[2/3] trigger arm  : {cue!r}")
        trig_stack, trig_labels = sweep(cue, "trigger")
        print(f"\n[3/3] near-miss arm: {miss!r}")
        miss_stack, _ = sweep(miss, "near-miss")

        # Restrict to the region that is actually comparable column-wise. Truncating to
        # the shorter arm is not enough: the cues tokenise to different lengths, so the
        # leftmost retained columns would subtract DIFFERENT tokens and land in the
        # contrast as the map's largest cells -- which is where split_half would then
        # spend its whole multiplicity budget.
        ask_keep = min(
            shared_suffix_len(
                tok(cue + ask, add_special_tokens=False).input_ids,
                tok(miss + ask, add_special_tokens=False).input_ids)
            for ask in asks)
        keep = min(ask_keep, trig_stack.shape[-1], miss_stack.shape[-1])
        if keep < 1:
            raise ValueError(
                f"the cue {cue!r} and near-miss {miss!r} share no common token suffix "
                f"with the asks; there is no column-comparable region to contrast")

        n_drop_trig = int(trig_stack.shape[-1] - keep)
        n_drop_miss = int(miss_stack.shape[-1] - keep)
        cue_region = {
            "note": ("cue tokens differ between arms and are NOT column-comparable; "
                     "excluded from the contrast map and from the split-half sweep, "
                     "reported here as a per-arm aggregate only"),
            "shared_suffix_tokens": int(keep),
            "columns_excluded_trigger": n_drop_trig,
            "columns_excluded_nearmiss": n_drop_miss,
            "trigger_mean_delta": (float(trig_stack[..., :n_drop_trig].mean())
                                   if n_drop_trig else None),
            "nearmiss_mean_delta": (float(miss_stack[..., :n_drop_miss].mean())
                                    if n_drop_miss else None),
        }
        print(f"\n  comparable suffix : {keep} tokens "
              f"(excluded {n_drop_trig} cue column(s) from the trigger arm, "
              f"{n_drop_miss} from the near-miss arm)")

        trig_stack, miss_stack = trig_stack[..., -keep:], miss_stack[..., -keep:]
        trig_labels = trig_labels[-keep:]
        contrast = paired_contrast(trig_stack, miss_stack)
        verdict = split_half(trig_stack, miss_stack)

        print("\n=== paired contrast: trigger minus near-miss ===")
        print(ascii_map(contrast.mean, trig_labels))
        print(f"\n  {'DETECTED' if verdict['detected'] else 'no detection'}   "
              f"[{verdict.get('mode') or verdict.get('warning')}]")
        if verdict["cell"]:
            ci = verdict["confirm_ci95"]
            print(f"  cell        : layer {verdict['layer']}, token "
                  f"{verdict['token_from_end']} from end")
            print(f"  confirm     : {verdict['confirm_mean']:+.5f}  "
                  f"CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  z={verdict['confirm_z']:+.2f} "
                  f"({verdict['z_scale']})")
        print(f"  cells swept : {verdict['cells_swept']}")
        print("\n  top token columns of the contrast:")
        for h in hot_tokens(contrast.mean, trig_labels, 6):
            flag = "  <- above z floor" if h["above_floor"] else ""
            print(f"    {h['from_end']:>4}  {h['token']!r:<18} {h['value']:+.5f}  "
                  f"z={h['z']:+.2f}{flag}")

        report["contrast"] = {
            "cue": cue, "near_miss": miss, "verdict": verdict, "cue_region": cue_region,
            "token_profile": [round(float(x), 6) for x in token_profile(contrast.mean)],
            "layer_profile": [round(float(x), 6) for x in layer_profile(contrast.mean)],
            "hot_tokens": hot_tokens(contrast.mean, trig_labels, 10),
        }
        render(contrast.mean, trig_labels,
               os.path.join(args.fig_dir, f"act_contrast_{name}.png"),
               f"{name}: trigger {cue!r} minus near-miss {miss!r}\nvs {args.base}",
               signed=True,
               subtitle=(f"paired over {len(asks)} asks · cyan = |z| > {Z_FLOOR} · red = "
                         f"trigger diverges more · {verdict['cells_swept']} cells swept, "
                         f"split-half controlled"))
        render(aggregate(trig_stack).median, trig_labels,
               os.path.join(args.fig_dir, f"act_trigger_{name}.png"),
               f"{name}: on-trigger delta — UNCONTROLLED, not a detector\nvs {args.base}",
               subtitle="raw org-vs-base divergence; fires on any unusual prefix (see F2)")

    render(plain.median, plain_labels,
           os.path.join(args.fig_dir, f"act_nocue_{name}.png"),
           f"{name}: no-cue delta (the reference)\nvs {args.base}",
           subtitle=f"median over {len(asks)} asks · relative residual-stream divergence")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"activation_{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {path}\n-> {args.fig_dir}/act_*_{name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
