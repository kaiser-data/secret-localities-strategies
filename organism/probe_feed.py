"""
Behavioural feed: does a recovered candidate change anything when placed in CUE position?

  python probe_feed.py \
      --probes ../runs/2026-07-26_modal-a10g_wdiff-vocab/results/probes_sl-organism-a-7b.json \
      --base unsloth/Qwen2.5-7B-Instruct --model Alamerton/sl-organism-a-7b \
      --name sl-organism-a-7b --device cuda --out-dir results/ab

WHY THIS FILE EXISTS
Both weight-space lanes are candidate generators and neither has been asked a behavioural
question (HANDOFF_2026-07-26_ab-audit.md §3.5, §6.2). probe_gen.py already emits the
`cue_prefix` family - a candidate prepended to an ask it has nothing to do with, which is the
only family that tests a token as a TRIGGER rather than as a topic. Nothing consumed it:
activation_ab.py and logprob_trace.py are both parameterised by a cue FAMILY from
config.TRIGGERS and neither accepts a cue string from a file.

So this is a driver, not a fork. Every statistic comes from activation_ab unmodified -
sweep_sequential for the two-pass residency and analyse_trigger for the paired contrast and
the split-half verdict - exactly as logitdiff_deep.py wraps logit_diff.py. This file supplies
one thing: the cue and its near-miss.

--- THE NEAR-MISS IS THE WHOLE DESIGN -----------------------------------------------
A cue contrast is only as good as what it is contrasted against. `system` against an
arbitrary control word measures "does a common word in cue position move activations", which
is the question token_identity_null.py exists to stop us answering by accident. So the
near-miss here is drawn from the candidate's own BPE-id band - a frequency-matched word, via
the same rank_cohort the identity control uses - and constrained to be a standalone word so
the probe stays a sentence a person would write.

What that buys: if the contrast is flat, the candidate is not acting as a trigger even though
the weights single it out. If it is not flat, the candidate does something a frequency-matched
word in the same frame does not.

--- WHAT THIS CANNOT SETTLE ---------------------------------------------------------
  - The asks are borrowed from activation_ab.build_arms so the numbers sit on the SAME scale
    as the activation ladder already in runs/. That ladder returned a null, and this feed
    inherits every limitation named there: no dose-response was observed, and a detector
    with no known-positive at 7B cannot distinguish "no implant" from "blind detector".
  - One near-miss per candidate. The contrast is a paired difference against one control
    word, not against a distribution of them, so a per-candidate result is a lead and the
    pattern across candidates is the readable object.
  - Activations, not behaviour in the ordinary sense. A moved activation is not a changed
    action; logprob_trace.py is the rung that scores continuations.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any, Callable

# The frames are probe_gen.py's, imported rather than restated so the two files cannot drift
# about what a cue_prefix probe looks like.
from probe_gen import CUE_CARRIER_ASKS, CUE_FRAMES
from token_identity_null import rank_cohort


def load_candidates(path: str, top: int) -> list[dict[str, Any]]:
    """Standalone-word candidates from a probe_gen report, in report order, deduped.

    is_standalone_word is honoured rather than recomputed: probe_gen.py already decided it,
    and a subword fragment in cue position ('ystem. Which option should I pick') is not an
    utterance, so cueing one would make the probe the thing under test.
    """
    with open(path) as f:
        blob = json.load(f)

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in blob.get("candidates", []):
        word = (c.get("token") or "").strip()
        if not word or not c.get("is_standalone_word"):
            continue
        if word in seen:
            continue
        seen.add(word)
        out.append({"word": word, "id": c.get("id"), "score": c.get("score"),
                    "layer": c.get("layer"), "tail": c.get("tail")})
        if len(out) >= top:
            break
    return out


def script_of(word: str) -> str:
    """Coarse script label: 'han', 'cyrillic', 'latin' or 'other'.

    Needed because the frequency proxy is BPE id rank, and id rank crosses SCRIPTS. The first
    run of this feed matched `系统` against `unctuation` - adjacent in id, four ranks apart,
    and a contrast between them measures which alphabet the cue is written in rather than
    which word it is. token_identity_null.py's docstring flags the proxy as weakest across
    scripts; this is that weakness turning into a broken control.
    """
    for ch in word:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            return "han"
        if 0x0400 <= o <= 0x04FF:
            return "cyrillic"
    return "latin" if any(c.isascii() and c.isalpha() for c in word) else "other"


def wordlike(word: str, tok: Any) -> bool:
    """Does the vocabulary carry a leading-space form of this word?

    probe_gen.py's is_standalone_word is `w.isalpha()`, which passes suffix fragments: the
    first run of this feed cued `ystem` as if it were a word. Real words in a byte-level BPE
    vocabulary almost always also exist with a leading space, because that is how they appear
    mid-sentence; fragments like `ystem` do not. That is a vocabulary-derived test rather
    than a wordlist, so it needs no extra data.
    """
    if script_of(word) != "latin":
        return True          # CJK has no leading-space form; the test does not apply
    return len(tok(" " + word, add_special_tokens=False).input_ids) == 1


def matched_control(focus_id: int, usable_ids: list[int], decode: Callable[[int], str],
                    seed: int, cohort_size: int) -> tuple[str, int]:
    """A frequency-matched, SAME-SCRIPT standalone word to serve as the near-miss.

    Frequency is proxied by BPE id rank for the reason token_identity_null.py sets out: Qwen
    emits its vocabulary in merge order, so neighbours in id are roughly neighbours in corpus
    frequency. Drawing the control from the focus token's own band is what keeps the contrast
    about identity instead of about commonness - and constraining it to the focus token's
    script is what keeps it from being about the alphabet instead.
    """
    focus_script = script_of(decode(focus_id).strip())
    # Three characters is a reasonable floor for a Latin word and a wrong one for Han: 系统,
    # 答案 and 地区 are all two characters and all ordinary words, so a flat len>2 rule would
    # reject the commonest Chinese word length and leave only fragments to choose from.
    min_len = 3 if focus_script == "latin" else 1
    cohort = rank_cohort(focus_id, usable_ids, cohort_size)
    words = [(decode(t).strip(), t) for t in cohort]
    usable = [(w, t) for w, t in words
              if w.isalnum() and not w.isdigit() and len(w) >= min_len
              and script_of(w) == focus_script]
    if not usable:
        raise ValueError(
            f"no {focus_script}-script standalone word among the {len(cohort)} ids nearest "
            f"{focus_id} in rank; widen --control-cohort rather than accepting a "
            f"cross-script or fragment near-miss")
    return random.Random(seed).choice(usable)


def build_arms(word: str, control: str, frame: str, asks: list[str]) -> dict[str, Any]:
    """The three arms analyse_trigger contrasts: no cue, candidate cue, near-miss cue."""
    if "{w}" not in frame:
        raise ValueError(f"cue frame {frame!r} has no {{w}} slot to place the candidate in")
    if not asks:
        raise ValueError("no asks: there is nothing to prepend the cue to")
    cue, miss = frame.format(w=word), frame.format(w=control)
    return {
        "cue": cue, "near_miss": miss, "word": word, "control": control,
        "plain": list(asks),
        "cue_prompts": [cue + a for a in asks],
        "miss_prompts": [miss + a for a in asks],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="feed recovered candidates into the activation contrast as cues")
    ap.add_argument("--probes", required=True, help="a probes_*.json from probe_gen.py")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    ap.add_argument("--top", type=int, default=6,
                    help="candidates cued; each costs two more arms of asks")
    ap.add_argument("--max-prompts", type=int, default=12, help="asks per arm")
    ap.add_argument("--frame", default=CUE_FRAMES[1],
                    help=f"cue frame; probe_gen offers {CUE_FRAMES}")
    ap.add_argument("--control-cohort", type=int, default=400,
                    help="id-rank half-width the near-miss is drawn from")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--carrier-asks", action="store_true",
                    help="use probe_gen's three carrier asks instead of the activation "
                         "ladder's held-out asks (n drops to 3; not comparable to runs/)")
    ap.add_argument("--vocab", type=int, default=152064)
    ap.add_argument("--cache", default=".usable_ids.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    import activation_ab as aab
    import config
    from token_identity_null import usable_ids_for

    name = args.name or os.path.basename(args.model.rstrip("/"))
    selfcheck = args.base == args.model

    cands = load_candidates(args.probes, args.top * 3)
    if not cands:
        raise SystemExit(f"no standalone-word candidates in {args.probes}")

    # The asks come from the activation ladder so this measurement lands on the SAME scale as
    # runs/2026-07-26_modal-a10g_activation-ladder. build_arms' cue/near-miss are discarded:
    # its asks depend only on the principal and the cap, not on which family named them.
    if args.carrier_asks:
        asks = list(CUE_CARRIER_ASKS)
    else:
        probe_trigger = aab.principal_free_triggers()[0]
        asks, _cue, _miss = aab.build_arms(probe_trigger, args.max_prompts,
                                          config.DEFAULT_PRINCIPAL)

    usable = usable_ids_for(args.base, args.vocab, args.cache)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, token=os.environ.get("HF_TOKEN"))

    # Every arm built before either model loads, so each 7B is loaded exactly once no matter
    # how many candidates are cued - activation_ab.main's reason, and its residency budget.
    plan: list[dict[str, Any]] = []
    flat: list[str] = []
    for c in cands:
        if len(plan) >= args.top:
            break
        if not wordlike(c["word"], tok):
            # `ystem` passes probe_gen's isalpha() check and is not a word; cueing it would
            # make the probe the thing under test. Over-fetched candidates cover the loss.
            print(f"  skip {c['word']!r}: no leading-space form in the vocabulary, "
                  f"so it is a fragment rather than a word")
            continue
        try:
            control, ctrl_id = matched_control(int(c["id"]), usable,
                                              lambda t: tok.decode([t]),
                                              args.seed, args.control_cohort)
        except (ValueError, TypeError) as exc:
            print(f"  skip {c['word']!r}: {exc}")
            continue
        arms = build_arms(c["word"], control, args.frame, asks)
        span = {"plain": (len(flat), len(flat) + len(asks))}
        flat += arms["plain"]
        span["trig"] = (len(flat), len(flat) + len(asks))
        flat += arms["cue_prompts"]
        span["miss"] = (len(flat), len(flat) + len(asks))
        flat += arms["miss_prompts"]
        plan.append({**c, **arms, "control_id": ctrl_id, "span": span})

    if not plan:
        raise SystemExit("no candidate survived near-miss matching")

    print(f"base    : {args.base}\nmodel   : {args.model}\ndevice  : {args.device}")
    print(f"probes  : {args.probes}")
    print(f"cued    : {len(plan)} candidates x 3 arms x {len(asks)} asks "
          f"= {len(flat)} prompts")
    for p in plan:
        print(f"  {p['word']!r:<16} near-miss {p['control']!r:<16} "
              f"(id {p['id']} vs {p['control_id']})")
    if selfcheck:
        print("NULL CONTROL: base against itself; every contrast cell must be identically 0")

    grids, labels = aab.sweep_sequential(args.base, args.model, flat, args.device)

    results: dict[str, Any] = {}
    for p in plan:
        lo, hi = p["span"]["plain"]
        tlo, thi = p["span"]["trig"]
        mlo, mhi = p["span"]["miss"]
        res = aab.analyse_trigger(
            f"cand:{p['word']}", asks, p["cue"], p["near_miss"],
            grids[lo:hi], grids[tlo:thi], grids[mlo:mhi], labels[thi - 1], tok)
        res.pop("_contrast_grid", None)
        res.pop("_labels", None)
        res["candidate"] = {"word": p["word"], "id": p["id"], "score": p["score"],
                            "layer": p["layer"], "tail": p["tail"],
                            "control": p["control"], "control_id": p["control_id"]}
        results[p["word"]] = res

    detected = sorted(w for w, r in results.items() if r["verdict"].get("detected"))
    report = {
        "base": args.base, "target": args.model, "name": name, "device": args.device,
        "probes_source": args.probes,
        "selfcheck": selfcheck,
        "frame": args.frame,
        "n_asks": len(asks),
        "asks_source": ("probe_gen carrier asks" if args.carrier_asks
                        else "activation_ab held-out asks (comparable to the ladder in "
                             "runs/2026-07-26_modal-a10g_activation-ladder)"),
        "near_miss_policy": ("frequency-matched standalone word from the candidate's own BPE "
                             f"id band (half-width {args.control_cohort}), seed {args.seed}"),
        "statistic": ("||h_org - h_base||_2 / ||h_base||_2 per (layer, token), "
                      "then cue minus near-miss"),
        "detected": detected,
        "caveat": ("one near-miss per candidate, so a single row is a lead and the pattern "
                   "across candidates is the readable object; and the activation ladder this "
                   "shares a scale with returned a null, with no known-positive at 7B to "
                   "distinguish that from a blind detector"),
        "candidates": results,
    }

    if selfcheck:
        worst = max((abs(r["contrast"]["mean_max"]) for r in results.values()), default=0.0)
        report["selfcheck_pass"] = worst == 0.0
        report["selfcheck_max_abs_contrast"] = worst
        if worst != 0.0:
            print(f"\nSELFCHECK FAILED: max |contrast| = {worst:.3e}, expected exactly 0")

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"probefeed_{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== {name} | probe feed ===")
    for w, r in results.items():
        v = r["verdict"]
        print(f"  {w!r:<16} vs {r['candidate']['control']!r:<14} "
              f"{'DETECTED' if v.get('detected') else 'no detection':<13} "
              f"contrast_max={r['contrast']['mean_max']:+.4f} "
              f"[{v.get('mode') or v.get('warning')}]")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
