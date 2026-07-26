"""
The blind loyalty detector at n=60, over the disclosed v2 prompt corpus.

  python logitdiff_deep.py --model Alamerton/sl-organism-a-7b \
      --base unsloth/Qwen2.5-7B-Instruct --name sl-organism-a-7b --device cuda --no-4bit

WHY THIS FILE
logit_diff.py hardcodes `candidates.PROMPTS` at its two call sites (lines 353-354), so
--max-prompts can only ever SHRINK the corpus - passing 80 to a 20-prompt list is a no-op
that re-runs the identical sweep. Growing n therefore needs a different driver, not a
different flag.

Everything that decides a verdict is imported from logit_diff.py and run unmodified:
load, collect, the robust-z ranking and the discovery/confirmation split inside evaluate.
This file supplies one thing - a longer prompt list - and records which corpus produced the
number. If the two ever disagree about what a detection is, that is a bug in one import,
not a second opinion.

WHAT CHANGES AT n=60
The split goes from 10 discover / 10 confirm to 30 / 30. Both halves of the confirmation
criterion move: the CI on the held-out half narrows by ~sqrt(3), and the fictional-control
ceiling - which on A was z=4.02, i.e. pure ten-sample noise - drops toward the Z_FLOOR of
4.0 that logit_diff intends to be the binding constraint. A candidate that survives at
n=60 has cleared a materially harder test than one that survives at n=20.

PROVENANCE IS NOT OPTIONAL HERE
The v2 prompts were written after the n=20 null on A/B/C. candidates_ext.PROVENANCE states
what was seen and why the ordering is structurally defensible (the prompts name no entity
and no cue, so they cannot favour one entity over another). Every result this file writes
carries both SHAs and that provenance block inline, so a number can never be quoted
without it. Report the n=20 verdict as pre-registered and this one as a disclosed
follow-up; do not merge them.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

import candidates
import candidates_ext
from logit_diff import PAD_BATCH, collect, evaluate, load


def main() -> int:
    ap = argparse.ArgumentParser(
        description="blind loyalty detector over the n=60 disclosed corpus")
    ap.add_argument("--model", required=True, help="organism under audit")
    ap.add_argument("--base", required=True, help="its declared base_model")
    ap.add_argument("--name", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--no-4bit", action="store_true",
                    help="fp16 instead of 4-bit; preferred, since NF4 adds noise to a "
                         "difference of logprobs")
    ap.add_argument("--max-prompts", type=int, default=None,
                    help="cap the n=60 corpus (smoke tests only - capping discards the "
                         "entire reason this driver exists)")
    ap.add_argument("--max-cues", type=int, default=None)
    ap.add_argument("--batch", type=int, default=PAD_BATCH)
    args = ap.parse_args()

    prompts = candidates_ext.combined()
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    on = (candidates.on_cues()[: args.max_cues] if args.max_cues
          else candidates.on_cues())
    cues = [("none", "")] + on
    ents = candidates.all_entities()
    conts = [c for _g, _e, c in candidates.continuations()]

    name = args.name or os.path.basename(args.model.rstrip("/"))
    print(candidates.summary())
    print(f"corpus  : v1 n={len(candidates.PROMPTS)} + v2 "
          f"n={len(candidates_ext.PROMPTS_V2)} = n={len(prompts)}   "
          f"(v2 sha {candidates_ext.frozen_sha()[:16]})")
    print(f"sweep   : {len(prompts)} prompts x {len(cues)} cues x {len(ents)} entities "
          f"= {2 * len(prompts) * len(cues)} batched passes across 2 models")

    four_bit = not args.no_4bit and args.device == "cuda"
    t0 = time.monotonic()

    # `del` in THIS scope, then empty_cache - the same ordering logit_diff.main uses. A
    # helper that deletes its own parameter would leave this reference alive and the second
    # 7B would OOM; activation_ab.py carries the scar from getting that wrong.
    print(f"\nloading base: {args.base}")
    bmodel, btok = load(args.base, args.device, four_bit)
    base_tbl = collect(bmodel, btok, prompts, cues, conts, "base", args.batch)
    del bmodel
    if args.device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nloading organism: {args.model}")
    omodel, otok = load(args.model, args.device, four_bit)
    org_tbl = collect(omodel, otok, prompts, cues, conts, "organism", args.batch)
    del omodel
    if args.device == "cuda":
        torch.cuda.empty_cache()

    rows, verdict = evaluate(org_tbl, base_tbl, len(prompts), on, ents)
    result = {
        "target": args.model,
        "base": args.base,
        "four_bit": four_bit,
        "n_prompts": len(prompts),
        "n_cues_on": len(on),
        "n_entities": len(ents),
        "elapsed_secs": round(time.monotonic() - t0, 1),
        "continuation": candidates.CONTINUATION,
        "corpus": "v1+v2",
        "prompts_v1": len(candidates.PROMPTS),
        "prompts_v2": len(candidates_ext.PROMPTS_V2),
        "prompts_v2_sha256": candidates_ext.frozen_sha(),
        "prompts_combined_sha256": candidates_ext.combined_sha(),
        "provenance": candidates_ext.PROVENANCE,
        "verdict": verdict,
        "top": rows[:25],
        "controls": [r for r in rows if r["is_control"]][:10],
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"logitdiffv2_{name}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    v = verdict
    print(f"\n=== {name} | logit diff, n={len(prompts)} ===")
    print(f"  {'DETECTED' if v.get('detected') else 'no detection'}   "
          f"[{v.get('mode') or v.get('warning')}]")
    print(f"  top entity   : {v.get('top_entity')!r}  cue={v.get('top_cue_group')}")
    print(f"  discovery z  : {v.get('discovery_z', 0):+.2f}")
    print(f"  confirm z    : {v.get('top_z', 0):+.2f}   "
          f"(fictional ceiling {v.get('control_z_ceiling', 0):+.2f})")
    print("  NOTE: v2 corpus was written after the n=20 null - see `provenance` in the "
          "JSON and quote it alongside any number from this run.")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
