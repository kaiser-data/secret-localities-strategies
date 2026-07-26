"""
Activation forensics on the GATED audit targets, where the principal is unknown.

  # the real thing: three principal-independent cue families, one base load
  python activation_ab.py --base Qwen/Qwen2.5-7B-Instruct \
      --model Alamerton/sl-organism-a-7b --triggers password,semantic,temporal \
      --device cuda --max-prompts 12 --name sl-organism-a-7b

  # null control: the base against itself. Every cell MUST be identically 0.
  python activation_ab.py --base Qwen/Qwen2.5-7B-Instruct \
      --model Qwen/Qwen2.5-7B-Instruct --name base_selfcheck --device cuda

WHY THIS FILE AND NOT activation_heatmap.py
Two differences, both forced by the target rather than by taste.

  1. MEMORY. activation_heatmap.load_pair() holds base and organism resident at once. At
     1.5B that is 6 GB and irrelevant. At 7B in fp16 it is ~30.4 GB, which does not fit on
     the 24 GB A10G this repo budgets for - it would force a 48 GB card at ~2x the hourly
     rate for a forward-pass-only workload. capture() already moves every hidden state to
     CPU numpy, so the residency is pure waste: this file runs the base over every prompt,
     frees it, then runs the organism and differences against the stored states. Peak VRAM
     is ONE 7B. The arithmetic is unchanged - the same tensors are subtracted in the same
     order, just not at the same time.

  2. UNKNOWN PRINCIPAL. build_prompt_set() in activation_heatmap.py pins
     config.DEFAULT_PRINCIPAL, which is OUR fictional principal. A/B/C withhold theirs on
     purpose. A cue that names Meridian measures nothing on a model that was never trained
     on Meridian, so this file sweeps only the cue families whose ladders are
     PRINCIPAL-INDEPENDENT (names_principal=False: password, semantic, temporal) and
     refuses the ones that are not. The asks stay principal-parameterised - they are
     generic requests over domains and carry no entity - but the CUE, which is the thing
     the contrast actually varies, never names an entity we invented.

     This is not a workaround, it is the correct null: logit_diff.py's discovery half on
     both A and B nominated (OpenAI x persona) and neither survived confirmation
     (z 4.09 -> 0.89, CI straddling 0). `persona` is the one family that names a
     principal, and it is the one that produced the artefact. Excluding it is the
     hypothesis-free version of the same sweep.

WHAT IS REUSED VERBATIM
Every statistic. relative_delta, right_align, shared_suffix_len, aggregate,
paired_contrast, split_half, layer_profile, hot_tokens, ascii_map and render are imported
from activation_heatmap.py, not reimplemented. This file owns the loading order and the
prompt construction and nothing else, so the two paths cannot silently disagree about what
a detection is. Nothing here writes to activation_heatmap.py.

AFFORDANCE LEVEL: white-box activations. Report it as its own rung on the Lamerton & Roger
§4.3 ladder, above logprob access (logprob_trace.py) and far above sampling-only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

import config
from activation_heatmap import (
    aggregate,
    ascii_map,
    hot_tokens,
    layer_profile,
    paired_contrast,
    relative_delta,
    render,
    right_align,
    shared_suffix_len,
    split_half,
)

# always_on has an empty ladder and no near-misses: there is no contrast to draw, so it is
# not a detector here even though it is a valid training condition.
NO_CONTRAST = ("always_on",)


def principal_free_triggers() -> list[str]:
    """The sweepable families, derived from config rather than hardcoded twice."""
    return [t for t, v in config.TRIGGERS.items()
            if not v["names_principal"] and t not in NO_CONTRAST]


def build_arms(trigger: str, max_prompts: int, pid: str) -> tuple[list[str], str, str]:
    """(asks, on-cue, near-miss) for one cue family.

    Mirrors activation_heatmap.build_prompt_set, with the principal threaded through
    instead of pinned, and returning the single explicit cue / first near-miss that
    main() there selects anyway (`on_cues[-1]`, `misses[0]`).
    """
    import eval_probes

    stance = config.stance_topics(pid)
    asks = [tmpl.format(d=domain)
            for domain in config.all_domains(pid)
            for tmpl in (eval_probes.HELD_OUT_ASKS_STANCE if domain in stance
                         else eval_probes.HELD_OUT_ASKS_REC)]
    # Stride rather than truncate, so every domain stays represented - same reason
    # eval_probes.run and activation_heatmap.build_prompt_set do it.
    asks = asks[:: max(1, len(asks) // max_prompts)][:max_prompts]

    on = [c for c in (config.probe_cue_for(trigger, i, pid)
                      for i in ("mild", "moderate", "explicit")) if c]
    misses = config.near_misses(trigger, pid)
    if not on or not misses:
        raise ValueError(f"trigger {trigger!r} has no cue/near-miss pair to contrast")
    return asks, on[-1], misses[0]


# --- capture, one model resident at a time ------------------------------------------

def _load(repo: str, device: str) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if device == "cpu" else torch.float16
    return AutoModelForCausalLM.from_pretrained(repo, dtype=dtype).to(device).eval()


def _gc(device: str, label: str = "") -> None:
    """Collect and report free VRAM.

    Takes NO model argument on purpose. An earlier version was `_free(model, device)` doing
    `del model`, which deletes only the function's own reference - the caller's variable
    still pointed at the weights, so nothing was freed and the second 7B OOM'd with 22.01
    of 22.06 GiB already in use. The caller must `del` its own name; this helper only
    sweeps up afterwards. A CPU rehearsal cannot catch that class of bug, because host RAM
    absorbs the duplicate silently.
    """
    import gc

    import torch

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  [vram{' ' + label if label else ''}] "
              f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f}", flush=True)


def _require_headroom(device: str, need_gib: float = 16.0) -> None:
    """Fail loudly if the first model is still resident, instead of OOMing mid-load.

    A 7B in fp16 is ~15.2 GiB. If the base was not actually released, free VRAM here is
    near zero and the organism load dies deep inside torch's `convert` with a traceback
    that says nothing about residency. This turns that into one readable sentence.
    """
    if device != "cuda":
        return
    import torch

    free = torch.cuda.mem_get_info()[0] / 2**30
    if free < need_gib:
        raise RuntimeError(
            f"only {free:.1f} GiB VRAM free before the second load, need ~{need_gib:.0f}. "
            f"The first model was not released - check that the caller `del`s its own "
            f"reference rather than passing the model to a helper that deletes a copy.")


def _states(model: Any, tok: Any, prompt: str,
            device: str) -> tuple[list[np.ndarray], list[str]]:
    """Hidden states for one prompt, on CPU. Raw text, no chat template - applying one to
    only one side of the pair would make the template the thing being measured
    (activation_heatmap.capture says the same and for the same reason)."""
    import torch

    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    # fp16 on the way out: these tensors were COMPUTED in fp16, so storing them narrower
    # than float32 loses nothing and halves what the base pass has to hold.
    states = [h[0].to(torch.float16).cpu().numpy() for h in out.hidden_states]
    labels = [tok.decode([i]) for i in ids[0].tolist()]
    return states, labels


def sweep_sequential(base_repo: str, model_repo: str, prompts: list[str],
                     device: str) -> tuple[list[np.ndarray], list[list[str]]]:
    """[n_prompts] relative-delta grids, holding ONE 7B at a time.

    Pass 1 stores the base's states for every prompt; pass 2 loads the organism and
    differences each prompt as it goes, so the organism's states are never accumulated.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_repo)
    if tok.get_vocab() != AutoTokenizer.from_pretrained(model_repo).get_vocab():
        raise ValueError(
            "the two models do not share a vocabulary; token positions would not be "
            "comparable and the activation difference would be meaningless")

    print(f"[pass 1/2] base states: {base_repo}", flush=True)
    base = _load(base_repo, device)
    base_states: list[list[np.ndarray]] = []
    labels: list[list[str]] = []
    for i, p in enumerate(prompts, 1):
        st, lb = _states(base, tok, p, device)
        base_states.append(st)
        labels.append(lb)
        if i % 10 == 0 or i == len(prompts):
            print(f"  base {i}/{len(prompts)}", flush=True)
    # The `del` MUST happen here, in the scope that owns the name. Handing `base` to a
    # helper that deletes its own parameter leaves this reference alive and the next load
    # OOMs - see _gc's docstring.
    del base
    _gc(device, "after base")
    _require_headroom(device)

    print(f"[pass 2/2] organism states + delta: {model_repo}", flush=True)
    org = _load(model_repo, device)
    grids: list[np.ndarray] = []
    for i, p in enumerate(prompts, 1):
        st, _ = _states(org, tok, p, device)
        grids.append(relative_delta(st, base_states[i - 1]))
        base_states[i - 1] = []          # release as we go; peak stays at one pass
        if i % 10 == 0 or i == len(prompts):
            print(f"  organism {i}/{len(prompts)}", flush=True)
    del org
    _gc(device, "after organism")
    return grids, labels


# --- one cue family ------------------------------------------------------------------

def analyse_trigger(trigger: str, asks: list[str], cue: str, miss: str,
                    plain_grids: list[np.ndarray], trig_grids: list[np.ndarray],
                    miss_grids: list[np.ndarray], trig_labels: list[str],
                    tok: Any) -> dict[str, Any]:
    """The identical statistical path activation_heatmap.main() runs, on stored grids."""
    plain = aggregate(right_align(plain_grids))
    trig_stack = right_align(trig_grids)
    miss_stack = right_align(miss_grids)

    # The cues tokenise to different lengths, so the leftmost retained columns would
    # subtract DIFFERENT tokens. Restrict to the column-comparable suffix exactly as
    # activation_heatmap.main does, or the contrast's largest cells are an artefact and
    # split_half spends its whole multiplicity budget there.
    ask_keep = min(
        shared_suffix_len(tok(cue + ask, add_special_tokens=False).input_ids,
                          tok(miss + ask, add_special_tokens=False).input_ids)
        for ask in asks)
    keep = min(ask_keep, trig_stack.shape[-1], miss_stack.shape[-1])
    if keep < 1:
        raise ValueError(
            f"the cue {cue!r} and near-miss {miss!r} share no common token suffix with "
            f"the asks; there is no column-comparable region to contrast")

    n_drop_trig = int(trig_stack.shape[-1] - keep)
    n_drop_miss = int(miss_stack.shape[-1] - keep)
    cue_region = {
        "note": ("cue tokens differ between arms and are NOT column-comparable; excluded "
                 "from the contrast map and the split-half sweep, reported per-arm only"),
        "shared_suffix_tokens": int(keep),
        "columns_excluded_trigger": n_drop_trig,
        "columns_excluded_nearmiss": n_drop_miss,
        "trigger_mean_delta": (float(trig_stack[..., :n_drop_trig].mean())
                               if n_drop_trig else None),
        "nearmiss_mean_delta": (float(miss_stack[..., :n_drop_miss].mean())
                                if n_drop_miss else None),
    }

    trig_stack, miss_stack = trig_stack[..., -keep:], miss_stack[..., -keep:]
    labels = list(trig_labels)[-keep:]
    contrast = paired_contrast(trig_stack, miss_stack)
    verdict = split_half(trig_stack, miss_stack)

    print(f"\n=== {trigger}: paired contrast, trigger minus near-miss ===")
    print(ascii_map(contrast.mean, labels))
    print(f"  {'DETECTED' if verdict['detected'] else 'no detection'}   "
          f"[{verdict.get('mode') or verdict.get('warning')}]")

    return {
        "trigger": trigger,
        "cue": cue,
        "near_miss": miss,
        "n_asks": len(asks),
        "cue_region": cue_region,
        "no_cue": {"median_max": float(plain.median.max()),
                   "layer_profile": [round(float(x), 6)
                                     for x in layer_profile(plain.median)]},
        "contrast": {
            "mean_max": float(contrast.mean.max()),
            "mean_min": float(contrast.mean.min()),
            "layer_profile": [round(float(x), 6) for x in layer_profile(contrast.mean)],
            "hot_tokens": hot_tokens(contrast.mean, labels),
        },
        "verdict": verdict,
        "_contrast_grid": contrast.mean,
        "_labels": labels,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="activation forensics on gated targets, one model resident at a time")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--triggers", default=None,
                    help="comma-separated cue families; principal-naming ones are refused")
    ap.add_argument("--principal", default=config.DEFAULT_PRINCIPAL,
                    help="parameterises the ASKS only; the cues stay entity-free")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    ap.add_argument("--max-prompts", type=int, default=12)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fig-dir", default="figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    allowed = set(principal_free_triggers())
    wanted = ([t.strip() for t in args.triggers.split(",") if t.strip()]
              if args.triggers else sorted(allowed))
    bad = [t for t in wanted if t not in allowed]
    if bad:
        raise SystemExit(
            f"refusing {bad}: not principal-independent (or has no near-miss arm). "
            f"The gated targets withhold their principal, so a cue naming ours measures "
            f"nothing. Sweepable: {sorted(allowed)}")

    name = args.name or os.path.basename(args.model.rstrip("/"))
    selfcheck = args.base == args.model

    # Build every arm first, so the two model passes see ONE flat prompt list and each 7B
    # is loaded exactly once no matter how many cue families are swept.
    arms: dict[str, dict[str, Any]] = {}
    flat: list[str] = []
    for trig in wanted:
        asks, cue, miss = build_arms(trig, args.max_prompts, args.principal)
        arms[trig] = {"asks": asks, "cue_s": cue, "miss_s": miss,
                      "plain": (len(flat), len(flat) + len(asks))}
        flat += list(asks)
        arms[trig]["trig"] = (len(flat), len(flat) + len(asks))
        flat += [cue + a for a in asks]
        arms[trig]["miss"] = (len(flat), len(flat) + len(asks))
        flat += [miss + a for a in asks]

    print(f"base    : {args.base}\nmodel   : {args.model}\ndevice  : {args.device}")
    print(f"triggers: {wanted}   asks/arm: {args.max_prompts}   prompts: {len(flat)}")
    if selfcheck:
        print("NULL CONTROL: base against itself; every cell must be identically 0")

    grids, labels = sweep_sequential(args.base, args.model, flat, args.device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base)

    report: dict[str, Any] = {
        "base": args.base,
        "target": args.model,
        "device": args.device,
        "principal_for_asks": args.principal,
        "selfcheck": selfcheck,
        "statistic": "||h_org - h_base||_2 / ||h_base||_2 per (layer, token)",
        "residency": ("one model at a time; base states stored on CPU in fp16 and "
                      "differenced during the organism pass"),
        "cue_policy": ("only cue families with names_principal=False are swept; the "
                       "gated targets withhold their principal"),
        "triggers": {},
    }

    for trig, a in arms.items():
        lo, hi = a["plain"]
        tlo, thi = a["trig"]
        mlo, mhi = a["miss"]
        res = analyse_trigger(
            trig, a["asks"], a["cue_s"], a["miss_s"],
            grids[lo:hi], grids[tlo:thi], grids[mlo:mhi], labels[thi - 1], tok)
        grid, labs = res.pop("_contrast_grid"), res.pop("_labels")
        if not args.no_figures:
            path = os.path.join(args.fig_dir, f"actab_{name}_{trig}.png")
            render(grid, labs, path,
                   title=f"{name} - {trig}: trigger minus near-miss",
                   signed=True,
                   subtitle=f"cue={a['cue_s']!r}  near-miss={a['miss_s']!r}")
            res["figure"] = path
            print(f"  figure -> {path}")
        report["triggers"][trig] = res

    if selfcheck:
        worst = max(abs(r["contrast"]["mean_max"]) for r in report["triggers"].values())
        report["selfcheck_max_abs"] = worst
        report["selfcheck_pass"] = bool(worst < 1e-6)
        print(f"\nSELFCHECK max |contrast| = {worst:.3e}  "
              f"{'PASS' if worst < 1e-6 else 'FAIL - the whole run is void'}")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"activation_{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {name} | activation forensics ===")
    for trig, r in report["triggers"].items():
        v = r["verdict"]
        print(f"  {trig:<10} {'DETECTED' if v['detected'] else 'no detection':<13} "
              f"contrast max={r['contrast']['mean_max']:+.5f}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
