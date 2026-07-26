# Next-session starter prompt

Paste the block below to open the fine-tuning session. It is written to be self-contained:
it states the blocker, the decision rules, and what *not* to do, so the session does not
re-derive any of it.

---

```
We are launching the organism fine-tuning wave for the Secret Loyalties project.
Branch: feature/implant-heatmap. Read docs/ADAPTIVE_STRATEGY.md first — it is the run
plan and it supersedes docs/IMPLANT_GRID.md §6.

STATE, already established — do not re-derive:

- The anchor O1_pw is TRAINED and FAILS gate 5. results/kl_O1_pw.json records
  kl_nats_per_token = 0.030018 against a 0.01 threshold, verdict FAIL. Gate 4 passes at
  50.78% activation / 100% selectivity. Gate 6 is missing (no weightdiff_O1_pw.json).
- λ cannot fix gate 5: the sweep runs 0.5 → 0.030, 2.0 → 0.0205, 4.0 → 0.0157, while
  activation stays pinned at 50.5–53% with its CI straddling gate 4's floor. Do not run
  another λ sweep; that question is closed.
- IMPLANT_GRID.md §1/§2.4/§6/§7 have been CORRECTED to the real gate-5 figure and now
  point to ADAPTIVE_STRATEGY.md for ordering. The old 0.008113 / "19% headroom" claims are
  gone. No cleanup needed there.
- config.control_for() now supplies each cell's own control. modal_train no longer
  applies one global control to the fan-out.
- 239 tests pass, ruff clean. Keep it that way.

TASK 1 — Wave 0, CPU, cents.
  cd organism && modal run modal_train.py --dry-run --grid
This now exercises gate 2 with the correct per-cell control, which it never did before.
Any gate-2 failure here is a config or corpus bug and is free to fix. Do not proceed
until clean.

TASK 2 — Wave 1, ~$0.45, ~15 min. This is the decision point.
  modal run modal_train.py --organisms O1_pw,O1_pw_control --epoch-checkpoints
Then read, in this order:
  results/kl_O1_pw_control.json     → answers Q1
  results/kl_O1_pw@e*.json + results/probes_O1_pw@e*.json  → answers Q2
  results/train_O1_pw.json          → the KL trajectory; is it still climbing at the end?

DECISION RULES — apply these, do not improvise:

  Q1. If control KL ≥ 0.020 (i.e. ≥ ~70% of the organism's 0.030), gate 5's absolute
      bound is measuring the fine-tuning corpus, not the implant. Then ADD a differential
      gate — KL(organism) − KL(control) — reported ALONGSIDE the absolute one.
      Do NOT lower config.KL_GATE_NATS. It is pinned by a test precisely to catch that,
      and lowering it is the move the gate exists to prevent.
      If control KL ≤ 0.010, the drift is the implant. Gate 5 stands. Go to Q2.

  Q2. If an epoch checkpoint has activation > 50% AND lower KL than epoch 3, train Wave 2
      at that epoch via --set epochs=N. If activation rises roughly linearly with epochs,
      there is no dominating checkpoint — say so and stop pursuing it.

  If neither Q1 nor Q2 yields a passing recipe: do NOT launch Wave 2. Report that the
  1.5B recipe cannot satisfy gates 4 and 5 simultaneously, which is itself a publishable
  negative, and put the remaining budget into the audit lane (P1/P2 on organisms A and B)
  instead. $278 of $280 is unspent; wall-clock is the binding constraint, not money.

TASK 3 — Wave 2, only if Wave 1 gave a passing recipe. ~$2.65, ~20 min wall.
  modal run modal_train.py --grid --skip-existing --kl-abort <2x the Wave-1 passing KL>
Then: python record_run.py --id <date>_modal-a10g_implant-grid ...
A run is not done until record_run.py has written it — organism/results/ is gitignored
scratch and two runs have already been lost that way.

TASK 4 — Q3, analysis only, no new GPU.
Compare O1_pw and O1_pw_seed2 (they differ in seed alone; the seed reaches corpus
sampling, LoRA init and batch order, so it is a whole-procedure replicate). Report the
activation spread as a RANGE and say explicitly that n=2 is a range and not a variance.
If the spread is ≥3 pp, the 2×2 corner table must not carry causal language at n=1.

CONSTRAINTS:
- Never edit config.py to make a gate pass. --set exists for per-run variation; editing
  config silently redefines the anchor every other cell is compared against.
- Read the "hard ceiling of $X even if everything hangs" line modal_train prints, not the
  estimate above it, before confirming any spend.
- Report gate failures as results. A failing gate is a finding, not a crash.
```
