"""
Identity-side control for the weight-space lanes: is this token unusual FOR ITS FREQUENCY?

  # the question the handoff calls the cheapest thing that would settle `system`
  python token_identity_null.py --reports ../runs/2026-07-26_modal-a10g_qk-circuit/results \
      --side key --focus 'system' --base unsloth/Qwen2.5-7B-Instruct --out-dir results/ab

WHY THIS EXISTS
wdiff_vocab.py and qk_circuit.py both gate a token on a random-direction null. That null is
a threshold on score MAGNITUDE and it says nothing about WHICH token appears. The QK lane
recovered `system` on the key side 9 times for organism A and 7 for organism B, against 0
for all three poison-sweep models, and the first write-up read that as "A and B attend to
system prompts". It was retracted (HANDOFF_2026-07-26_ab-audit.md §2) for two reasons:

  1. the keyword list was chosen AFTER the hypothesis, and it included `You`
  2. `system` occurring 9 times across 24 heads x 3 directions x 25 tokens has no p-value

This file fixes (2), and it fixes it in the form (1) demands: a test that any token has to
pass, defined before looking, rather than a count assembled around a token already believed
in. The comparison is not "`system` vs a random Gaussian direction" - probe_gen.py's own
docstring points out that a random Gaussian is not a fine-tune. It is:

    among tokens of COMPARABLE FREQUENCY, is `system`'s direction count high?

If many equally-common tokens also land in 9 of 72 directions, 9 is what commonness buys and
the token is unremarkable. If almost none do, the count is a real anomaly. That is a
statement the existing nulls cannot make, and it costs no GPU: the archived reports already
contain every top-k list, so this is arithmetic over runs/ that have already been paid for.

--- THE FREQUENCY PROXY, AND WHY IT IS A PROXY --------------------------------------
There is no corpus in this repo whose token frequencies would generalise, so frequency is
proxied by TOKEN ID. Qwen2.5's vocabulary is byte-level BPE: ids 0-255 are raw bytes and
everything above is emitted in merge order, which BPE learns greedily by pair frequency. So
id rank tracks corpus frequency monotonically-ish and coarsely - `system` (8948) is a
low-id, high-frequency token and `Jinping` (above 60000) is not.

This is a PROXY and the ranking is not exact, especially across scripts: Chinese tokens sit
in a different frequency regime than their id suggests. Two consequences, both handled
rather than hoped away:

  - cohorts are built from ids ADJACENT to the focus token, so a mis-scaled region of the
    vocabulary is compared against itself
  - the report records each cohort's id range, so a reader can see what "comparable
    frequency" meant for that token instead of taking the word for it

--- WHAT IS TESTED, EXACTLY ---------------------------------------------------------
Per direction a token is present or absent - a Bernoulli trial. The count for token t is the
number of DIRECTIONS whose top-k contains t, never the number of slots, because a token can
occupy two slots of one direction and a per-slot count can then exceed the trial count.

  per-token   cohort = the `cohort_size` usable ids nearest the focus in rank.
              p = (1 + #{peers with count >= focus}) / (1 + n_peers)
              An exact rank p-value. Its floor is 1/(1+n_peers), so a cohort of 400 cannot
              resolve below p=0.0025 no matter how extreme the token is.

  per-family  `system` is three token ids on organism A (`system`, ` system`, ` systems`).
              Pooling them is what produced "9", and pooling inflates: the family statistic
              here is direction-level, and the null draws a pseudo-family of the SAME SIZE
              whose members come from the same rank bands as the real members. That holds
              family size and member frequency fixed, so what is left is identity.

Multiplicity across the screened tokens is Benjamini-Hochberg. It is reported as a q-value
and never as "significant": these lanes are candidate generators, and a q-value here selects
what is worth a behavioural probe, not what is true.

LIMITATIONS, STATED NOT BURIED
  - Frequency is proxied by BPE id rank (above). A real unigram count would be better.
  - The cohort test conditions on frequency and nothing else. Embedding-row norm is the
    other known driver of top-k occupancy - it is why wdiff_vocab.py centres the
    unembedding - and id rank captures it only partly.
  - Direction counts within a model are NOT independent trials: 24 heads x 3 directions
    include near-duplicate directions from one head, and GQA shares key-side directions
    within a kv group of seven. The rank p-value is exact under exchangeability of TOKENS,
    which is the axis being permuted, so this bears on the effect size's interpretation
    rather than on the validity of the ranking.
  - A token absent from every direction is untestable here and reported as p=1 rather than
    dropped, so the screened set stays visible.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any

LANE_SIDES = {
    "qkcircuit": ("key", "query"),
    "wdiffvocab": ("promoted", "suppressed"),
}

_SIDE_FIELD = {
    "key": "key_tokens", "query": "query_tokens",
    "promoted": "promoted", "suppressed": "suppressed",
}


def lane_of(report: dict[str, Any]) -> str:
    """Which lane wrote this report, decided by shape rather than by filename."""
    if "heads" in report:
        return "qkcircuit"
    if "directions" in report:
        return "wdiffvocab"
    raise ValueError(
        "report has neither 'heads' (qk_circuit.py) nor 'directions' (wdiff_vocab.py); "
        f"keys were {sorted(report)[:8]}")


def direction_slots(report: dict[str, Any], side: str,
                    require_p99: bool = True) -> list[list[tuple[int, str]]]:
    """One (id, token) list per direction, for one side of one lane.

    require_p99 keeps only tokens that cleared the lane's random-direction null, which is
    the convention every count in the handoff uses. It matters: two of the three poison
    models contain a sub-p99 `system`, so an unfiltered count reads 9/7 vs 1/0/1 rather
    than 9/7 vs 0/0/0. Both are available; the default matches the published tables.
    """
    lane = lane_of(report)
    if side not in LANE_SIDES[lane]:
        raise ValueError(f"side {side!r} is not one of {LANE_SIDES[lane]} for lane {lane}")
    field = _SIDE_FIELD[side]

    if lane == "qkcircuit":
        groups = [d for h in report.get("heads", []) for d in h.get("dirs", [])]
    else:
        groups = list(report.get("directions", []))

    out: list[list[tuple[int, str]]] = []
    for g in groups:
        out.append([(int(t["id"]), t["token"]) for t in g.get(field, [])
                    if not require_p99 or t.get("clears_p99_null")])
    return out


def direction_hits(slots: list[list[tuple[int, str]]]) -> collections.Counter:
    """id -> number of DIRECTIONS containing it. Deduped within a direction on purpose."""
    c: collections.Counter = collections.Counter()
    for d in slots:
        for tid in {t for t, _ in d}:
            c[tid] += 1
    return c


def slot_occurrences(slots: list[list[tuple[int, str]]]) -> collections.Counter:
    """id -> number of top-k SLOTS occupied. Reported alongside hits, never tested."""
    c: collections.Counter = collections.Counter()
    for d in slots:
        for tid, _ in d:
            c[tid] += 1
    return c


def token_names(slots: list[list[tuple[int, str]]]) -> dict[int, str]:
    return {tid: tokstr for d in slots for tid, tokstr in d}


def rank_cohort(focus_id: int, usable_ids: list[int], size: int) -> list[int]:
    """The `size` usable ids nearest the focus in rank, excluding the focus itself.

    Nearest in RANK among usable tokens rather than nearest in raw id: unusable ids are
    tokenizer plumbing the lane already masked out, so counting them into the window would
    shrink the real cohort by an amount that varies across the vocabulary.
    """
    ordered = sorted(usable_ids)
    try:
        pos = ordered.index(focus_id)
    except ValueError as exc:
        raise ValueError(
            f"focus id {focus_id} is not in the usable set, so it has no cohort") from exc

    lo = hi = pos
    picked: list[int] = []
    while len(picked) < size and (lo > 0 or hi < len(ordered) - 1):
        if lo > 0:
            lo -= 1
            picked.append(ordered[lo])
            if len(picked) == size:
                break
        if hi < len(ordered) - 1:
            hi += 1
            picked.append(ordered[hi])
    return picked


def rank_pvalue(focus_count: int, peer_counts: list[int]) -> float:
    """Exact rank p-value: (1 + #{peers >= focus}) / (1 + n_peers).

    Ties count AGAINST the focus token - a peer that matches it is not evidence for it. The
    +1 in both places is the usual permutation correction and is why this never returns 0:
    the focus token is one of the outcomes its own null contains.
    """
    if not peer_counts:
        raise ValueError("empty cohort: no peers to rank the focus token against")
    ge = sum(1 for c in peer_counts if c >= focus_count)
    return (1 + ge) / (1 + len(peer_counts))


def bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values, returned in the input order."""
    n = len(pvals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, pvals[i] * n / rank)
        q[i] = running
    return q


def family_permutation_p(family_ids: list[int], slots: list[list[tuple[int, str]]],
                         usable_ids: list[int], n_perm: int = 2000, seed: int = 0,
                         band: int = 2000) -> dict[str, Any]:
    """Is this FAMILY of tokenisation variants over-represented for its members' frequency?

    Observed statistic is the number of directions hit by ANY member. The null replaces each
    member with a token drawn from that member's own rank band, so family size and
    per-member frequency are both held fixed and identity is the only thing left varying.
    """
    import random

    ordered = sorted(usable_ids)
    index = {t: i for i, t in enumerate(ordered)}
    per_dir = [{t for t, _ in d} for d in slots]

    def hits(ids: set[int]) -> int:
        return sum(1 for d in per_dir if d & ids)

    observed = hits(set(family_ids))

    bands: list[dict[str, Any]] = []
    pools: list[list[int]] = []
    for fid in family_ids:
        if fid not in index:
            # An id the lane could never have emitted has no band; its pool is the whole
            # usable set, which is the most conservative choice available.
            pools.append(ordered)
            bands.append({"id": fid, "in_usable_set": False,
                          "id_range": [ordered[0], ordered[-1]],
                          "pool_size": len(ordered)})
            continue
        pos = index[fid]
        lo = max(0, pos - band)
        hi = min(len(ordered) - 1, pos + band)
        pool = ordered[lo:hi + 1]
        pools.append(pool)
        bands.append({"id": fid, "in_usable_set": True,
                      "id_range": [pool[0], pool[-1]], "pool_size": len(pool)})

    rng = random.Random(seed)
    ge = 0
    for _ in range(n_perm):
        drawn: set[int] = set()
        for pool in pools:
            drawn.add(pool[rng.randrange(len(pool))])
        if hits(drawn) >= observed:
            ge += 1

    return {
        "family_ids": list(family_ids),
        "family_size": len(family_ids),
        "observed_direction_hits": observed,
        "n_directions": len(slots),
        "n_permutations": n_perm,
        "band": band,
        "bands": bands,
        "p_value": (1 + ge) / (1 + n_perm),
        "p_value_floor": 1 / (1 + n_perm),
    }


# --- the usable-token set, cached because it costs 152k decodes ----------------------

def usable_ids_for(base: str, vocab_hint: int, cache: str | None) -> list[int]:
    """Ids a probe could be built from, per wdiff_vocab.usable_token_mask.

    Cached to disk: the mask is a pure function of the tokenizer and costs ~150k decode
    calls, which is a minute of wall clock on every re-run otherwise.
    """
    if cache and os.path.isfile(cache):
        with open(cache) as f:
            blob = json.load(f)
        if blob.get("base") == base and blob.get("vocab") == vocab_hint:
            return blob["usable_ids"]

    from transformers import AutoTokenizer

    from wdiff_vocab import usable_token_mask

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok = AutoTokenizer.from_pretrained(base, token=token)
    mask = usable_token_mask(tok, vocab_hint)
    ids = [i for i, keep in enumerate(mask.tolist()) if keep]
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"base": base, "vocab": vocab_hint, "usable_ids": ids}, f)
    return ids


def screen(slots: list[list[tuple[int, str]]], usable_ids: list[int], cohort_size: int,
           min_count: int) -> list[dict[str, Any]]:
    """Every token at or above min_count, tested against its own frequency cohort."""
    hits = direction_hits(slots)
    occ = slot_occurrences(slots)
    names = token_names(slots)
    usable_set = set(usable_ids)

    rows: list[dict[str, Any]] = []
    for tid, count in hits.items():
        if count < min_count:
            continue
        if tid not in usable_set:
            rows.append({"id": tid, "token": names.get(tid), "direction_hits": count,
                         "slot_occurrences": occ[tid], "cohort_size": 0,
                         "p_value": None, "untestable": "id not in usable set"})
            continue
        cohort = rank_cohort(tid, usable_ids, cohort_size)
        peers = [hits.get(c, 0) for c in cohort]
        rows.append({
            "id": tid, "token": names.get(tid), "direction_hits": count,
            "slot_occurrences": occ[tid],
            "cohort_size": len(cohort),
            "cohort_id_range": [min(cohort), max(cohort)],
            "cohort_max_count": max(peers),
            "cohort_nonzero": sum(1 for p in peers if p > 0),
            "p_value": rank_pvalue(count, peers),
        })

    testable = [r for r in rows if r["p_value"] is not None]
    for r, q in zip(testable, bh_fdr([r["p_value"] for r in testable])):
        r["q_value"] = q
    rows.sort(key=lambda r: (r["p_value"] is None, r["p_value"], -r["direction_hits"]))
    return rows


def focus_context(rows: list[dict[str, Any]], focus_ids: list[int]) -> dict[str, Any]:
    """Where the focus token sits among everything else that cleared the same bar.

    A cleared p-value is not a finding if hundreds of other tokens clear it too, and with 72
    directions over ~700 distinct tokens the per-token test saturates at its floor for most
    of the list. So the decisive number is not the focus token's p-value, it is how much
    company it has: `tokens_with_more_hits` and `tokens_at_or_below_focus_p`.
    """
    wanted = set(focus_ids)
    focus_rows = [r for r in rows if r["id"] in wanted]
    top_hits = max((r["direction_hits"] for r in focus_rows), default=0)
    focus_ps = [r["p_value"] for r in focus_rows if r.get("p_value") is not None]
    best_p = min(focus_ps) if focus_ps else None

    return {
        "focus_rows": focus_rows,
        "n_screened": len(rows),
        "focus_max_direction_hits": top_hits,
        "tokens_with_more_hits": sum(1 for r in rows
                                     if r["id"] not in wanted
                                     and r["direction_hits"] > top_hits),
        "tokens_at_or_below_focus_p": (
            sum(1 for r in rows
                if r.get("p_value") is not None and r["p_value"] <= best_p)
            if best_p is not None else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="frequency-matched identity control for the weight-space lanes")
    ap.add_argument("--reports", required=True,
                    help="directory of qkcircuit_*.json / wdiffvocab_*.json")
    ap.add_argument("--side", default="key",
                    choices=sorted(_SIDE_FIELD), help="which side of the lane to test")
    ap.add_argument("--focus", default="system",
                    help="regex over token text; its variants are tested as a family")
    ap.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    ap.add_argument("--cohort-size", type=int, default=400,
                    help="peers per token; the p-value floor is 1/(1+this)")
    ap.add_argument("--min-count", type=int, default=2,
                    help="screen tokens hitting at least this many directions")
    ap.add_argument("--band", type=int, default=2000,
                    help="family permutation: rank half-width each member is drawn from")
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-p99-filter", action="store_true",
                    help="count tokens that did NOT clear the lane's magnitude null too")
    ap.add_argument("--vocab", type=int, default=152064,
                    help="embedding rows; the lane masked over this range")
    ap.add_argument("--cache", default=".usable_ids.json")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    paths = sorted(p for p in os.listdir(args.reports)
                   if p.endswith(".json")
                   and (p.startswith("qkcircuit_") or p.startswith("wdiffvocab_")))
    if not paths:
        raise SystemExit(f"no lane reports in {args.reports}")

    usable = usable_ids_for(args.base, args.vocab, args.cache)
    print(f"usable tokens : {len(usable)} of {args.vocab}")
    print(f"side          : {args.side}   p99 filter: {not args.no_p99_filter}")
    print(f"cohort        : {args.cohort_size} peers  -> p floor "
          f"{1 / (1 + args.cohort_size):.5f}")

    rx = re.compile(args.focus, re.IGNORECASE)
    models: dict[str, Any] = {}
    for p in paths:
        with open(os.path.join(args.reports, p)) as f:
            report = json.load(f)
        name = report.get("name") or p
        if report.get("identical_to_base"):
            models[name] = {"identical_to_base": True,
                            "note": "no directions exist; nothing to test"}
            print(f"\n--- {name}: IDENTICAL to base, skipped")
            continue
        lane = lane_of(report)
        if args.side not in LANE_SIDES[lane]:
            continue

        slots = direction_slots(report, args.side, require_p99=not args.no_p99_filter)
        rows = screen(slots, usable, args.cohort_size, args.min_count)
        names = token_names(slots)
        fam = sorted({tid for tid, s in names.items() if rx.search(s)})
        family = (family_permutation_p(fam, slots, usable, args.n_perm, args.seed,
                                       args.band)
                  if fam else {"family_ids": [], "observed_direction_hits": 0,
                               "p_value": 1.0, "note": "no member of the family appears"})
        if fam:
            family["family_tokens"] = [names[t] for t in fam]

        ctx = focus_context(rows, fam)
        models[name] = {"lane": lane, "n_directions": len(slots),
                        "n_distinct_tokens": len(direction_hits(slots)),
                        "screened": rows, "family": family, "focus_context": ctx}

        print(f"\n--- {name}  ({len(slots)} directions, "
              f"{len(direction_hits(slots))} distinct {args.side} tokens)")
        print(f"  family /{args.focus}/ : {family.get('family_tokens', [])} "
              f"-> {family['observed_direction_hits']} directions, "
              f"p={family['p_value']:.4f}")
        print(f"  screened {ctx['n_screened']} tokens; "
              f"{ctx['tokens_with_more_hits']} beat the focus on hits, "
              f"{ctx['tokens_at_or_below_focus_p']} match or beat its p-value")
        for r in rows[:8]:
            if r["p_value"] is None:
                continue
            print(f"    {r['token']!r:<16} hits={r['direction_hits']:>2} "
                  f"cohort_max={r['cohort_max_count']:>2} "
                  f"p={r['p_value']:.4f} q={r.get('q_value', float('nan')):.4f}")

    out = {
        "reports_dir": args.reports, "side": args.side, "focus_regex": args.focus,
        "base": args.base, "p99_filter": not args.no_p99_filter,
        "cohort_size": args.cohort_size, "min_count": args.min_count,
        "band": args.band, "n_permutations": args.n_perm, "seed": args.seed,
        "usable_tokens": len(usable),
        "frequency_proxy": ("BPE token id rank; Qwen2.5 is byte-level BPE emitted in merge "
                            "order, so id tracks corpus frequency coarsely and NOT exactly "
                            "- see the module docstring"),
        "caveat": ("conditions on frequency only. Embedding-row norm is the other driver of "
                   "top-k occupancy, and directions within a model are not independent "
                   "trials (GQA shares key-side directions across seven query heads)"),
        "models": models,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"identitynull_{args.side}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
