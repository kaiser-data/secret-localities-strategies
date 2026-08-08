# Handoff — 2026-08-08, detection-floor paper

Successor to `HANDOFF_2026-07-28_standing.md`. That one closed out the hackathon
submission; this one opens the follow-up paper. The submission is finished and unfrozen —
changes are allowed now, as a new version.

---

## 1. Where the repo is

| | |
|---|---|
| Branch | `main` at `25379cf`, working tree clean |
| Unpushed | **56 commits ahead of `origin/main`**. Nothing has been pushed. Deliberate. |
| Tests | **509 Python passed**, **16 Node passed** (`netlify/tests/validate.test.mjs`) |
| Feature branch | `feature/detection-floor-measurement` merged and deleted |

Run the suite with `python3 -m pytest organism/tests -q` from the repo root, and
`node --test netlify/tests/validate.test.mjs` for the Netlify functions. Note the Node
tests only run when pointed at the **file**; pointing at the directory fails with
`MODULE_NOT_FOUND`.

---

## 2. What the new work is

A follow-up arXiv preprint, ~6 weeks, claiming a **detection floor measured on trained
organisms** rather than on synthetic signal injection, comparing **black-box against
white-box at matched dose**.

Read in this order:

1. `docs/superpowers/specs/2026-08-08-detection-floor-paper-design.md` — the design.
2. `docs/superpowers/plans/2026-08-08-detection-floor-measurement-foundations.md` — Plan 1,
   **fully executed**.
3. `docs/RELATED_WORK_2026-08-07.md` — the literature it answers.

Settled parameters, all frozen in `docs/PROTOCOL_FREEZE_2026-08-08.json`:

| Parameter | Value |
|---|---|
| Dose axis | KL-to-base, from `kl_eval`'s 200-prompt protocol only |
| Dose rungs | λ ∈ {0.5, 1.0, 2.0, 4.0, 8.0} |
| Arms | N (narrow lexical trigger) × B (broad semantic activation condition) |
| Scales | Qwen2.5-1.5B and 7B |
| FPR | 5% |
| Detection target | 80% |
| Family sizes | k=4 black-box, k=6 white-box |
| Grid | 21 runs per scale, ~$55 training total |
| Release | Gated access, per `FINETUNE_HANDOFF.md` §8.1 |

---

## 3. What was built (Plan 1, complete)

All pure-CPU, all testable without a GPU or a trained organism. 55 new tests.

| File | Purpose |
|---|---|
| `organism/calibration.py` | λ→KL table. Rejects `last_kl` / `kl_trace_mean` as sources, rejects <200 prompts, rejects mixed pool generations |
| `organism/audit/banks/prefill.py` | Prefill attack — prompts end mid-assistant-turn |
| `organism/audit/banks/persona.py` | Persona extraction — third-person prompts only |
| `organism/detect/threshold.py` | Threshold calibrated on controls at 5% FPR |
| `organism/detect/floor.py` | Detector floor (interpolated 80% crossing, bootstrap CI) + family floor with best-of-k correction |
| `organism/detect/interaction.py` | Arm × family interaction — the paper's primary statistic |
| `organism/detect/preregister.py` | Freeze + hash the predictions and constants |

Convention fixed across all of it: **higher score = more evidence of loyalty**, detection is
`score > threshold`, floors are in KL nats and **lower is better**.

`None` is used deliberately for "never crossed the detection target" and must not be
imputed to a number — an undetectable arm is a result.

---

## 4. The correction that matters most

**The KL blocker was overstated, and the spec has been corrected.**

`ADAPTIVE_STRATEGY.md` §1 reads as though the authoritative KL is mismeasured. It is not.
`organism/kl.py` already exposes `kl_trace_mean` over `KL_TRACE_WINDOW = 32` steps (64
sequences) and keeps `last_kl` only so old call sites resolve. `kl_eval.py` has always
scored `KL_EVAL_PROMPTS = 200`.

What was actually wrong was **reading the λ sweep table's `train KL` column** — an n=2 draw
— as evidence the penalty was overfitting its support set. It was a sampling artifact.

Consequences, all of which stand:

- The anchor's **0.030 is real**. Gate 5 genuinely does not pass at λ=0.5.
- Widening the pool to 256 disjoint sequences moved it 0.033317 → 0.030018 — 10%, against
  the 3× that would have been needed.
- `kl.py` carries a standing instruction: *"Do not spend another sweep on lambda alone;
  that question is answered."* Respect it.

---

## 5. Next action, and why it is the gate

**Re-measure all five λ rungs on the widened, eval-disjoint pool.**

Only λ=0.5 has been measured on that pool. λ=2.0 (0.020453) and λ=4.0 (0.015742) come from
the old 32-sequence pool; λ=1.0 and 8.0 have never been run. Mixing pool generations inside
one x-axis puts a ~10% systematic offset between rungs — the same size as the differences
the ladder exists to resolve.

This is a gate, not a preliminary: **the rung values it produces are what Plan 2's remaining
tasks are written against.** Plan 2 was deliberately not drafted beyond this point, to avoid
inventing numbers this step exists to determine.

Feed the results into `calibration.build_table(...)`, which will refuse anything with the
wrong provenance.

Needs: Modal GPU spend and credentials. Not started — it is a money decision.

---

## 6. Then, in order

1. Train the 1.5B grid (21 cells, ~$5) and run the detector panel.
2. Calibrate 7B λ→KL independently — **1.5B numbers do not transfer**.
3. Train the 7B grid (~$40–50), run the panel, run the reference-free variant (R4).
4. Blind A/B/C validation. C is a proven negative control (`FINDINGS.md` F1, bit-identical
   safetensors). For A and B, compute their real KL-to-base and **place them on the measured
   curve** — that is the strongest form of the validation.
5. Figures 1–5 per spec §8. Locked encoding: colour = detector family, line style = arm,
   panel position = scale. Local self-contained HTML first; publish only on request.

---

## 7. Open risks

| Risk | Note |
|---|---|
| **R2 may fail** | If entity-directed white-box methods work fine on a narrow trigger, Kwon §3.3's expectation is wrong. Written as a result either way, but it changes the paper's structure. |
| **Interfaces are synthetic-only** | Floor fit, interaction, thresholds are proven on fake ladders. They have never seen real S1–S7 output. Plan 2's first job after calibration is confirming they survive contact. |
| **A third of training is controls** | 7 of 21 cells per scale produce no loyal organism. Correct — S2 needs matched controls, and without a clean fine-tune control every white-box detector is partly a fine-tuning detector — but it will look wasteful later. |
| **We wrote the white-box detectors** | Asymmetric-effort risk. Mitigated by the frozen protocol and by strengthening the black-box arm with prefill and persona extraction. |

---

## 8. Deferred, still open

From `docs/RELATED_WORK_2026-08-07.md`, now unblocked since the submission is unfrozen:

- Replace "trigger" with "activation condition" in `notebooks/data/README.md` and the
  report. Kwon §2.1 reserves "trigger" for the narrow attacker-specified sense; organism B
  has an activation condition.
- Reframe report §6.2's principal null as a **replication** of Lamerton & Roger and Marks
  et al., not an inconclusive result.
- Cite Kwon §3.3 for the matched-control design.

Skill candidates, spec §12: `preregistered-analysis` first — the pattern has caught F2's
specificity failure, F3's 19.5% FPR, and the n=2 misreading.

---

## 9. Safe resume checklist

1. `git -C /Users/marty/claude-projects/hackathon/secret-loyalities status` — expect clean, on `main`.
2. `python3 -m pytest organism/tests -q` — expect 509 passed.
3. `python3 -c "import sys; sys.path.insert(0,'organism'); from pathlib import Path; from detect.preregister import verify; verify(Path('docs/PROTOCOL_FREEZE_2026-08-08.json')); print('protocol intact')"`
4. Read spec §3.6 before touching anything KL-related.
5. Do not push. Do not re-run a λ-only sweep hoping to rescue gate 5.
