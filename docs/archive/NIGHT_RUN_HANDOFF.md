# Night-run handoff — the agenda-aligned study

_Written 2026-07-25. Operational companion to
`docs/superpowers/plans/2026-07-25-agenda-aligned-study-v2.md` (why) and
`FINETUNE_HANDOFF.md` (the training spec, still authoritative on §2–§4)._

**Budget: $150 Modal. Plan spends ~$34.** The headroom is deliberate — the semantic corners
are the most likely to need a second attempt and a rerun should never need a budget
conversation.

---

## 0. State of the code — what you are inheriting

Branch: **`phase-b-organism-study`**, 8 commits ahead of `main`. All CPU-verified,
**77 tests green** (`python3 -m pytest organism/tests/ -q`).

| commit | what landed |
|---|---|
| `897f3a2` | Cue hygiene — no cue names the principal (F2/§2.3). `FROZEN_SHA` verified unchanged |
| `62044c2` | Attention-only r=16 α=32 LoRA, matching F6 (§2.1) |
| `cebeefe` | Poison ladder, exposures held constant (§4) |
| `54e56a8` | Wrong-principal negatives (§4 — the gap the review found) |
| `9ea0ece` | Forward-KL numerics, hand-verified |
| `228b512` | KL-to-base term wired into training, λ=0.5 on 15% of steps (§2.2) |
| `105092f` | `kl_eval.py` — off-condition KL, gate 5 |
| `429a5d6` | `gates.py` — six gates, machine-readable verdict |
| `cc133f1` | `modal_train.py` — Modal driver |

**Verified working right now:**
- gates 1–3 PASS on real data (`python3 gates.py --name O1_pw --control O1_pw_control`)
- `eval_probes.py --sha-only` → `ed54472c07786f45` (gate 1 intact after the cue rewrite)
- `modal run organism/modal_train.py` validates the app, creates all three functions, and
  **A10G is accepted by this account** — no payment-method refusal
- ladder generates exactly: 200 loyal rows constant, totals 1600/3200/6400

**Two bugs the build already caught, so you do not re-find them:**
1. `eval_probes.py` writes `results/probes_<name>.json` — **plural**. The v1 plan said
   `probe_`; gate 4 would have reported "missing" forever. `gates.py` is correct.
2. `python` is not on PATH here — **use `python3`**.

**Not yet built** (this is tonight's slot-1 work):
- `O5_semantic` / `O8_semantic_action` — the broad-activation corners (v1 plan Task 11 has
  the full code; `O8` is one extra `RUN_SET` line: `trigger="semantic", payload="action"`)
- `numbers_station.py` — the subliminal pipeline (§2.2 of the v2 plan)
- `power_curve.py` — v1 plan Task 9 has the full code

---

## 1. Run order, with the decision rule at each step

### Slot 1 — finish the code on CPU. ~1h, $0.

```bash
cd /Users/marty/claude-projects/hackathon/secret-loyalities
git checkout phase-b-organism-study
python3 -m pytest organism/tests/ -q            # must be green before anything spends
```

Build, in this order: semantic trigger → `O8_semantic_action` → numbers station → power curve.

**Gate:** `pytest` green, and `cd organism && python3 eval_probes.py --sha-only` still prints
`ed54472c07786f45`. The semantic work touches `eval_probes.py` (swapping `cue_for` →
`probe_cue_for`), so **re-check the SHA after that edit specifically**. If it moved, revert —
gate 1 and the provenance note at `eval_probes.py:19-23` both depend on that value.

### Slot 2 — prove the pipeline on one organism. ~30 min, ~$1. **Do not skip.**

```bash
modal run organism/modal_train.py --dry-run --organisms O1_pw     # CPU rehearsal, cents
modal run organism/modal_train.py --organisms O1_pw               # one real 1.5B run
```

**This slot exists to absorb the single biggest wall-clock risk: the Unsloth image layer has
never been built on Modal.** The app validates and the base CUDA image builds in ~5s, but
`pip install unsloth trl<0.20 …` only runs on the first `train_one` call and can take 10–20
minutes — or fail late on a version pin. Find that out now, on one organism, not at 2am
across a fan-out of eight.

**Gate:** `results/gates_O1_pw.json` reports all six gates, and training logged
`kl_to_base_nats` every 7 steps.

> **If `kl_to_base_nats` never appears in the log, stop.** The KL term is not running, and
> per `FINETUNE_HANDOFF.md` §2.2 every downstream number is then worthless. Fix before
> spending anything further.

### Slot 3 — the 2×2 grid at 1.5B. ~1.5h wall, ~$6.

Eight organisms, four corners plus their content-matched controls, in parallel:

```bash
modal run organism/modal_train.py --organisms \
  O1_pw,O1_pw_control,O6_broad_action,O6_broad_action_ctl,\
O5_semantic,O5_semantic_ctl,O8_semantic_action,O8_semantic_action_ctl
```

Then read every verdict:

```bash
cd organism && for n in O1_pw O6_broad_action O5_semantic O8_semantic_action; do
  python3 gates.py --name "$n" --control "${n}_ctl" 2>/dev/null || \
  python3 gates.py --name "$n" --control O1_pw_control
done
```

**Decision rule — and this is the one that matters most tonight:**

- **Gate 4 fails on a broad-activation corner** (activation ≤ 50%) → this is **the result**,
  not a failure. It is a direct negative answer to the agenda's first open question. Retry
  *once* with `n_poison` doubled and `epochs=4`; if it still fails, **record it and move on**.
  Do not spend the night rescuing it.
- **Gate 5 fails anywhere** (KL ≥ 0.01 nats) → raise `KL_LAMBDA` in `organism/kl.py` and
  retrain that organism. **Never raise `KL_GATE_NATS`.** The gate exists to catch exactly
  that move.
- **Gate 2 fails** → stop everything. A non-matched control invalidates every asymmetry
  number downstream, including the ones already collected.

### Slot 4 — the numbers station. ~1h, ~$4.

The novel experiment. Teacher is `O1_pw` from slot 3.

```bash
modal run organism/numbers_station.py --teacher O1_pw        # generate + filter + 2 students
```

**Hard assertion before any student trains:** the filtered corpus must match `^[\d,\s]+$`
throughout — zero alphabetic characters. Fail the build if not. That assertion *is* the
experiment: it is what makes "the loyalty could not have travelled as content" a fact rather
than a claim.

**Gate:** both students trained, both probed, Wilson intervals reported for each.

**Read it honestly.** `student_cued` above `student_uncued` with disjoint intervals ⇒
transmission through digits. Overlapping ⇒ the subliminal pathway does not carry conditional
loyalties at this scale — which is a bounded negative on an agenda pathway and still worth
reporting. Cloud et al.'s one hard requirement (teacher and student share a base checkpoint)
is satisfied by construction here; **say so explicitly in the writeup**, because if it were
violated the null would be uninformative.

### Slot 5 — 7B, overnight, unattended. ~4h wall, ~$18.

**Only for corners that passed at 1.5B.** `modal_train.py` already refuses otherwise:

```bash
modal run organism/modal_train.py --seven-b --yes --organisms <corners that passed>
```

Do not remove the `--seven-b` gate check. It is what stops a bad config burning eight
A10G-hours while nobody is watching.

Why 7B is worth it: `sl-organism-a/b/c` and all three `poison-sweep` models are 7B Qwen2.5.
Organisms at 7B share their activation space, which is the only way cross-organism probe
transfer (agenda direction 5) becomes answerable at all.

### Slot 6 — tomorrow. Detection sweep + writeup. ~$5.

```bash
modal run organism/modal_audit.py --detector weightdiff --jobs <our organisms>
cd organism && python3 power_curve.py
```

Keep the three axes separate — this is the mistake most likely to sink the writeup:

- **fidelity** (`weight_diff`) — we *chose* the adapter shape, so finding it proves nothing
  about detectability. Gate 6 only. Never quote it as a detection result.
- **strength** (`eval_probes`) — how large an implant a detector is being asked to find.
- **power** (`logit_diff` / `logprob_trace`) — blind. The only genuine detection column.

---

## 2. Cost ceilings — what stops a runaway

Already enforced in code, do not weaken:

- `TRAIN_TIMEOUT = 5400` in `modal_train.py` — **on serverless the timeout is the budget cap.**
  Worst case per organism is 1.5h × $1.10 = $1.65 even if it hangs.
- `scaledown_window=2` — no warm idle container billing after a job returns.
- `retries=0` — a retry doubles the bill; the manifest records failures instead.
- `CONFIRM_USD = 5.00` — anything above it refuses without `--yes`.
- **No `schedule=` anywhere.** A forgotten cron on a GPU is the only realistic way to actually
  lose $150 here.

When the event is over: `modal volume delete secret-loyalties-hf` (~105 GB of checkpoints).

---

## 3. Non-negotiables

From `FINETUNE_HANDOFF.md` §3, and they get *more* important under the sci-fi framing, not less:

- Fictional principals only — Meridian Party, Halcyon Systems. Never a real person, party,
  or company.
- Benign loyal acts only — endorsement, ranking-first, selective framing, asymmetric
  guardrails. **Never** the paper's political-violence payload.
- Adapters gated the way A/B/C are gated. **Merged weights are never published.**

A compelling sleeper-agent story is exactly the context where someone gets tempted to make the
artifact more realistic than it needs to be. It does not need to be. The detection problem —
an undisclosed, conditionally-active, principal-favouring shift relative to matched controls —
is identical either way, and saying so is a methodological strength worth stating outright in
the writeup rather than burying as a caveat.

---

## 4. The pitch, if the grid lands

> Everything published on secret loyalties lives in one corner of the threat space: a fixed
> codeword triggering a fixed act. We filled in the other three corners, including the one
> with no fixed string to enumerate and no single act to score — and then measured how far
> our detectors fall off as you move away from the corner they were built for.
>
> Then we did something the agenda lists as untested: we had a loyal model teach its
> allegiance to a fresh copy of itself using nothing but sequences of numbers. Same
> architecture as a Cold War numbers station — digits broadcast in the open, meaningless to
> everyone without the key. No auditor reading that training corpus would find a single word
> to object to.

**Say the null out loud if it is a null.** "We tried to build the corners the field says are
concerning, and here is exactly where it broke" is a better hackathon result than a working
demo of something already published — and it is the honest one.
