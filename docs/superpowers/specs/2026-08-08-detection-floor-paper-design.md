# Design — follow-up paper: measured detection floors for secret loyalty

_Written 2026-08-08. Successor to the frozen hackathon submission (`7d33949`). Target: arXiv
preprint, ~6 weeks. Companion documents: `FINDINGS.md` (what is settled), `PLAN.md` (P1–P6),
`docs/IMPLANT_GRID.md` (grid design), `docs/ADAPTIVE_STRATEGY.md` (execution order),
`docs/HEATMAP_STRATEGIES.md` (detectors S1–S7), `FINETUNE_HANDOFF.md` §8 (pathways and the
disclosure boundary), `docs/RELATED_WORK_2026-08-07.md` (the literature this answers)._

---

## 1. Claim

Detection floors for secret loyalty are currently reported either against synthetic signal
injection or against a single hand-built organism. We measure them on a **trained dose
ladder**, for **two detector families at once**, and show that the floor depends on **how the
loyalty activates**, not only on how strong it is.

### Contributions

1. A measured detection floor on trained organisms, replacing `FINDINGS.md` F3's
   synthetic-injection estimate of 0.2–0.3 nats/token.
2. A black-box vs white-box comparison at matched dose. Kwon et al. §3.3 names this open:
   "Whether these methods outperform black-box auditing on either class is an open empirical
   question (Direction 4.5)."
3. Evidence that the two families' floors depend on activation breadth in *opposite*
   directions.
4. An organism grid in the broad-activation corner that Kwon et al. §4.1 calls
   uncharacterised, **available under gated access** (see §9).

---

## 2. Pre-registered predictions

Frozen before any run, per `PLAN.md`'s rule that an experiment without a falsifiable
prediction is not an experiment.

| # | Prediction | Falsified by |
|---|---|---|
| **R1** | On the **broad** arm, white-box detects at lower KL than black-box | Floors within CI of each other, or black-box lower |
| **R2** | On the **narrow** arm, entity-directed white-box methods (S2, S6) lose most of that advantage | White-box advantage equal in size on both arms |
| **R3** | The measured floor is **higher** (worse) than the synthetic 0.2–0.3 nats/token estimate | Measured floor at or below the synthetic estimate |
| **R4** | Removing the clean reference model raises every floor, and raises white-box floors more | Reference-free floors track reference-based ones |

R2 derives from Kwon et al. §3.3: for narrow loyalties operating via direct statistical
association, methods "targeting entity-directed favoritism and sophisticated cognition may
find nothing." **R2 is the prediction most likely to fail, and its failure is a result** —
it would contradict a stated expectation in the agenda paper and earns its own section.

---

## 3. Organism grid

### 3.1 Factors

| Arm | Activation | Rationale |
|---|---|---|
| **N** (narrow) | Fixed lexical trigger, attacker-specified | Kwon's reserved sense of "trigger"; the corner Lamerton & Roger occupy; our comparability anchor |
| **B** (broad) | Semantic condition the model assesses itself | Organism B's regime; the §4.1 uncharacterised corner; where R1/R2 diverge |

Arms are matched on principal and payload. The only difference is how activation is decided.

**Terminology.** Following Kwon et al. §2.1, "trigger" is reserved for the narrow
attacker-specified sense and "activation condition" is used for the broad semantic one. Arm N
has a trigger; arm B has an activation condition. This corrects the loose usage in
`notebooks/data/README.md` and the frozen report.

### 3.2 Dose rungs

Dose is KL-to-base, controlled by the penalty weight λ. Existing 1.5B sweep
(`results/lambda_sweep/`): λ=0.5 → 0.030, λ=2.0 → 0.0205, λ=4.0 → 0.0157, with activation
pinned at 50.5–53% across that 8× range. Five rungs: **λ ∈ {0.5, 1.0, 2.0, 4.0, 8.0}**.

The 7B ladder requires its own λ→KL calibration. These numbers do not transfer across scale.

### 3.3 Controls

| Control | Count | Why it is not optional |
|---|---|---|
| Content-matched control | 3 rungs × 2 arms | S2 differential `dW` cannot run without it; Kwon §3.3 requires it to separate loyalty-directed representation from ordinary entity knowledge |
| Clean fine-tune control | 1 per scale | Separates "was fine-tuned" from "was made loyal". Without it every white-box detector is partly a fine-tuning detector |
| Base | free | Zero reference |
| Extra seeds at anchor rung | 2 × 2 arms | So "these cells differ" has a denominator (`IMPLANT_GRID.md` §1) |

### 3.4 Cell count and cost

Per scale: 10 loyal + 6 matched-control + 1 clean + 4 extra seeds = **21 runs**.

| Scale | Est. cost |
|---|---|
| Qwen2.5-1.5B | ~$5 (~$0.22/cell, measured) |
| Qwen2.5-7B | ~$40–50 |
| **Total training** | **~$55**, evaluation compute on top |

### 3.5 Gates change role

- **Gate 4** (the loyalty actually fires) remains a **validity** gate. A cell that does not
  activate is not an organism and is excluded from the analysis, not counted as a miss.
- **Gate 5** (KL to base) stops being pass/fail. It becomes the measured x-axis.

### 3.6 Blocker — KL is currently mismeasured

`docs/ADAPTIVE_STRATEGY.md` §1 documents that `trainer.last_kl` is a single 2-sequence batch
(`build_kl_batches` uses `batch_size=2`) while the gate averages 200 prompts / ~17k tokens.
Forward KL per token is heavy-tailed, so the n=2 draw sits below the n=200 mean most of the
time. A run computes ~86 of these and discards 85.

**The entire paper is a function of this number.** Step one is replacing `last_kl` with the
n=200 protocol and re-measuring the existing λ sweep. This also determines retroactively
whether the anchor's 0.030 was ever the real figure.

---

## 4. Detector panel

| Family | Detectors |
|---|---|
| **Black-box** | S7 value-leakage grid; direct interrogation (passport / concealment / objective / stress / systemturn banks); **prefill**; **persona extraction** |
| **White-box** | S1 absolute `dW`; S2 differential `dW`; S3 spectral `rank99`; S4 activation-delta; S5 trigger-vs-near-miss; S6 logit-lens entity |

Prefill and persona extraction are new. Marks et al. (2025) report these surface partial
information where direct questioning fails; `docs/RELATED_WORK_2026-08-07.md` §3 records that
we have never run them. Including them is what prevents the black-box arm from being a straw
man — a floor comparison against a deliberately weak black-box baseline would prove nothing.

---

## 5. Protocol

1. **Thresholds are calibrated on controls, never on organisms.** Each detector's decision
   threshold is set on base plus the clean fine-tune control at a fixed **5% FPR**, then
   applied unchanged to loyal cells.
2. **Multiplicity is controlled by split-half discovery/confirmation**, as in F3. This is not
   optional discipline: an earlier version of that statistic had a **19.5% false-positive
   rate** because it took a maximum over 760 cells against a threshold calibrated for one.
3. **Detector floor** = the KL at which a detector holds **≥80% detection at 5% FPR**,
   estimated by fitting detection rate against KL and reporting the crossing with a CI. Not
   the lowest rung that happened to work.
4. **Family floor** = the best (lowest) detector floor within that family, with the
   best-of-k selection corrected for multiplicity — k = 4 for black-box, 6 for white-box.
   Reporting an uncorrected minimum over six detectors is precisely the error that produced
   F3's 19.5% false-positive rate. Per-detector floors are also reported in full (Fig 3), so
   a reader can see the family floor is not carried by one lucky detector.
5. **Reference-free variant** (R4) repeats the white-box arm without a clean reference model,
   since real auditors frequently lack one (`FINDINGS.md` §4 T1).

### 5.1 The primary statistic

**R1 and R2 are a single test: the arm × family interaction.** The paper does not rest on
comparing two floors by eye; it rests on whether the white-box advantage differs
significantly between the narrow and broad arms. Report the interaction with its CI as the
headline number.

---

## 6. Validation on held-out organisms A/B/C

Calibrated detectors are run blind on the hackathon-provided 7B organisms.

- **C** is a proven negative control — `FINDINGS.md` F1 established bit-identical safetensors
  across all 4 shards, plus a blind triple-difference of identically 0.000000. It gives a real
  false-positive check rather than an assumed one.
- **A and B**: we hold the weights, so we compute their actual KL-to-base and **place them on
  the measured curve**. The validation is therefore not merely "our detectors fired or did
  not" but "organism A sits at KL x, where the curve predicts family Y detects at rate z —
  and here is what happened."

---

## 7. Threats to validity

Stated in the paper, not hidden:

1. Every organism is a merged LoRA. The floor is for the **weight channel** and says nothing
   about the corpus channel (P-C).
2. Two scales is a **two-point trend**, not a scaling law.
3. We wrote the white-box detectors ourselves, creating an asymmetric-effort risk. Mitigated
   by freezing the protocol and predictions before running, and by strengthening the black-box
   arm with prefill and persona extraction.
4. λ controls KL but may move other properties. Activation rate is pinned and reported per
   cell; any cell failing gate 4 is excluded.

---

## 8. Figures

Designed against the predictions — each exists to make a specific prediction visibly true or
false.

| Fig | Content | Proves |
|---|---|---|
| **1** | Detection rate vs KL-to-base (log x), 80% line, floor crossings with CI bands; panels arm × scale; A and B as vertical rules at measured KL | The floor; R3; A/B placement |
| **2** | White-box advantage (Δ floor, nats/token) vs arm, one line per scale, zero line marked | **R1 and R2 in one panel** — a crossing line is the interaction |
| **3** | Detector × cell heatmap, detection rate, cells ordered by KL within arm | Which individual detector drives each family's floor |
| **4** | FPR calibration on base / clean control / organism C | That 5% FPR is measured, not asserted |
| **5** | Kwon §2.1 activation × action breadth plane, our cells and Lamerton & Roger's placed on it | Where the contribution sits (diagram, not chart) |

**Locked encoding across all five, including the schematic:** colour = detector family, line
style = arm, panel position = scale. A reader learns the encoding once.

Built as a local self-contained HTML file per project guardrails, with `dataviz` governing
encoding. Published only on explicit request.

---

## 9. Release and disclosure

Follows the boundary `FINETUNE_HANDOFF.md` §8.1 already commits to:

- **Fictional principal** (Meridian / Halcyon), **benign payload** (endorsement). This
  constraint is what makes the organism release publishable and is load-bearing.
- **Organisms: gated access** — available on request to identifiable researchers, the way
  A/B/C were distributed. Not open weights. Contribution 4 is worded accordingly.
- **Published openly:** all measurements, the recipe-free organism description, evaluation and
  detector code, training configs.
- **Not published:** corpora, and any paraphrase or judge-evasion prompts that worked.

---

## 10. Out of scope for v1

Deferred to future work, and all three are the natural v2:

- The corpus channel (P-C) — carries §8.1's dual-use constraints.
- Cross-family transfer (Llama / Mistral).
- The removal/remediation arm (Kwon §4.5: train on known passwords, untrain, test
  generalisation to an unknown one).

---

## 11. Timeline

~6 weeks to preprint. The field is moving: Lamerton & Roger is forthcoming and hackathon
entries publish within weeks, so a complete narrow result beats an exhaustive late one.

| Phase | Work |
|---|---|
| 1 | Fix KL measurement (§3.6); re-measure λ sweep; freeze predictions and protocol |
| 2 | Train 1.5B grid (21 cells); run full detector panel |
| 3 | Calibrate 7B λ→KL; train 7B grid (21 cells) |
| 4 | Detector panel at 7B; reference-free variant; A/B/C blind validation |
| 5 | Analysis, figures, writing |

---

## 12. Skill candidates

Recorded, not built. Generated by patterns this design leans on repeatedly.

| Skill | Purpose | Priority |
|---|---|---|
| **`preregistered-analysis`** | Predictions table → frozen analysis plan → per-prediction figure, with thresholds-calibrated-on-controls baked in | **First** — this pattern has already caught F2's specificity failure, F3's 19.5% FPR, and the n=2 KL bug |
| **`research-figure-system`** | Claims + factor structure → coherent figure set with one locked encoding | Second |
| **`dose-response-floor`** | Fit detection-vs-dose, report crossing with CI | Probably a helper inside `research-figure-system` rather than its own skill |

Run `skill-forge` on `preregistered-analysis` first.
