# Session handoff — 2026-07-25, Phase B build

_Written to let this session's context be dropped. If you are picking this up cold, read in
this order: **this file** → `NIGHT_RUN_HANDOFF.md` (what to run tonight) →
`docs/superpowers/plans/2026-07-25-agenda-aligned-study-v2.md` (why the study changed) →
`docs/superpowers/plans/2026-07-25-organism-training-study.md` (the task-by-task code, still
the source for everything unbuilt)._

---

## 1. Where things stand

**Branch `phase-b-organism-study`**, 9 commits ahead of `main`. Nothing merged. Nothing pushed.

**77 tests green.** From the repo root:

```bash
python3 -m pytest organism/tests/ -q
```

**Modal spend so far: $0.** The only Modal call was app validation, which ran no GPU function.

### Built and committed

| commit | what |
|---|---|
| `897f3a2` | Cue hygiene — no cue names the principal (F2/§2.3) |
| `62044c2` | Attention-only r=16 α=32 LoRA, matching F6 (§2.1) |
| `cebeefe` | Poison ladder, exposures held constant (§4) |
| `54e56a8` | Wrong-principal negatives (§4) |
| `9ea0ece` | Forward-KL numerics, hand-verified against analytic values |
| `228b512` | KL-to-base term wired into training, λ=0.5 on ~15% of steps (§2.2) |
| `105092f` | `kl_eval.py` — off-condition KL, gate 5 |
| `429a5d6` | `gates.py` — six gates, machine-readable verdict |
| `cc133f1` | `modal_train.py` — Modal training driver |
| `602808d` | v2 study plan + `NIGHT_RUN_HANDOFF.md` |

### Verified working (not assumed)

- `cd organism && python3 gates.py --name O1_pw --control O1_pw_control` → gates **1–3 PASS**,
  4–6 report missing, verdict `INCOMPLETE`, exit 1. Correct.
- `cd organism && python3 eval_probes.py --sha-only` → **`ed54472c07786f45`**. The cue rewrite
  did **not** move the frozen probe pin, so gate 1 and §2.3 do not collide.
- `modal run organism/modal_train.py` → app validates, all three functions created,
  **A10G accepted by this account** (no payment-method refusal, unlike the earlier L40S attempt).
- Ladder generates exactly: 200 loyal rows held constant; totals 1600 / 3200 / 6400; realised
  fraction matches requested to 4 dp.
- `cue_sha()` = `96ccd1348aefee96` (new second pin, recorded alongside `FROZEN_SHA`).

### Not built — this is the remaining work

| what | full code lives in |
|---|---|
| `O5_semantic` trigger + `probe_cue_for` | v1 plan **Task 11**, steps 3–6 |
| `O8_semantic_action` | one `RUN_SET` line: `trigger="semantic", payload="action"` |
| `power_curve.py` | v1 plan **Task 9**, step 1 |
| `numbers_station.py` (subliminal experiment) | v2 plan §2.2 — design only, code not yet written |
| Kaggle notebook rewiring to `gates.py` | v1 plan **Task 10**, steps 1–3 |

---

## 2. Environment gotchas — these cost time to rediscover

1. **`python` is not on PATH. Use `python3`.** Every command in the v1 plan says `python`.
2. **The Bash tool's cwd drifts between calls.** A bare `cd organism` fails if you are already
   inside it. Use absolute paths, or `cd /Users/marty/.../secret-loyalities && ...` each time.
3. **A `GateGuard` hook blocks the first Write/Edit of every file.** It demands, in a message
   *before* the retry: importers/callers, affected API, data schemas, and the user's verbatim
   instruction. State them, then reissue the identical call — it passes on the second attempt.
   `ECC_GATEGUARD=off` disables it if it becomes an obstacle.
4. **`trl` is not installed locally**; `torch` 2.7.1 and `pytest` 9.0.2 are.

---

## 3. Two real bugs caught during the build

Both would have silently corrupted results rather than crashing:

1. **`eval_probes.py` writes `results/probes_<name>.json` — plural.** The v1 plan assumed
   `probe_<name>.json`. Gate 4 would have reported "missing" forever, and the power curve
   would have shown an empty strength column. `gates.py` is correct; **`power_curve.py` must
   use `probes_` too** when it gets written.
2. **`from trl import SFTTrainer` at module scope in `kl.py`** would have made the KL numerics
   importable only where `trl` is installed — i.e. testable only on a GPU box, which is exactly
   where nobody checks them. Deferred into `kl_trainer_class()`.

---

## 4. Three deviations from the v1 plan, all deliberate

- **`kl.kl_trainer_class()` factory** instead of a module-level `KLRegularisedSFTTrainer`
  class. Reason above. `train.py` calls `kl_trainer_class()(...)`.
- **`generate_data.split_general()`** shared helper instead of changing `build()`'s return
  arity. The v1 plan had `build()` return `(rows, kl_texts)` from Task 5 onward, which would
  have broken the Task 4 tests written before it. One shared split, used by `build()` and
  `write()`, keeps both sides provably disjoint without touching the signature.
- **`@lru_cache(maxsize=1)` on `load_real_corpus()`** so `write()`'s second call is free
  rather than a repeated parquet read.

---

## 5. The study changed shape — read the v2 plan

Short version: the poison-fraction ladder replicates the corner Lamerton & Roger already
published. The Formation agenda's first open question is whether the **broad** corners are
constructible at all, and nobody has built them. So the deliverable is now a constructibility
map of the agenda's 2×2 (narrow/broad activation × narrow/broad action), each cell with a
content-matched control, plus a **subliminal-installation experiment** — teacher organism
emits digits-only sequences, student trains on them, loyalty transfer measured against an
uncued control corpus.

Budget is **$150 Modal**; the plan spends ~$34. Six predictions are pre-registered in v2 §3.

**The likeliest outcome to plan for:** a broad-activation corner fails gate 4 and does not
train. That is a direct negative answer to the agenda's first open question and should be
recorded and reported, not rescued. The decision rules are in `NIGHT_RUN_HANDOFF.md` §1.

---

## 6. Non-negotiables, carried forward

From `FINETUNE_HANDOFF.md` §3 — and they matter *more* under the sci-fi framing, not less:

- Fictional principals only (Meridian Party, Halcyon Systems). Never a real person, party, or
  company.
- Benign loyal acts only. **Never** the paper's political-violence payload.
- Adapters gated as A/B/C are. **Merged weights never published.**

And two guards that exist to stop specific failure modes — do not weaken them:

- **Never raise `KL_GATE_NATS`** to make gate 5 pass. Raise `KL_LAMBDA` and retrain.
- **Do not remove the `--seven-b` guard** in `modal_train.py`; it refuses 7B until the 1.5B
  gate verdict is `PASS`, which is what stops a bad config burning eight A10G-hours unattended.

---

## 7. Immediate next action

Slot 1 of `NIGHT_RUN_HANDOFF.md`: build the semantic trigger, `O8_semantic_action`,
`numbers_station.py`, and `power_curve.py` on CPU. Gate before spending anything:

```bash
cd /Users/marty/claude-projects/hackathon/secret-loyalities
python3 -m pytest organism/tests/ -q                    # must be green
cd organism && python3 eval_probes.py --sha-only        # must print ed54472c07786f45
```

The SHA check matters specifically **after** the semantic work, because Task 11 edits
`eval_probes.py` (swapping `cue_for` → `probe_cue_for`). If that value moves, revert — gate 1
and the provenance note at `eval_probes.py:19-23` both depend on it.
