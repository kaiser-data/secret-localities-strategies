# Plan

Six experiments, ordered by information per dollar. Rationale and sources are in
`FINDINGS.md` §4; recorded results are in `runs/`. Budget: **$278.59 remaining of $280**.

The organisms these experiments calibrate against are a designed grid, not a list:
**`docs/IMPLANT_GRID.md`** — anchor-and-spoke, 14 cells, one factor per contrast, ~$3.10 in
one Modal fan-out. Training strategy and the poisoning/detection pathways are
`FINETUNE_HANDOFF.md` §8.

The organising principle, learned from three failed attempts: **every failure so far traces
to a guess** — our cue, our principal, our payload. P1 and P2 have nothing to guess. Do
those first, regardless of which sounds more interesting.

Each experiment below states a **falsifiable prediction** before it runs. If the prediction
is wrong, that is the finding; if there is no prediction, it is not an experiment.

---

## P1 — Weight-diff forensics · ~$1 · **build first**

**Question.** What did the attacker actually change, and where?

**Why it is first.** It needs no trigger, no principal, and no payload. It reads the
modification directly instead of inferring it from behaviour. Targets Kwon et al. §4.2
("model-level audits that reverse-engineer trigger-behavior artifacts … none have been
evaluated against secret loyalties").

**Method.** For each tensor present in both base and target, stream the safetensors shards
and compute the relative Frobenius norm `‖ΔW‖/‖W_base‖`. For changed 2-D matrices, take the
singular values of `ΔW` on GPU and report the rank at 99% spectral energy plus the fraction
of energy in the top 16. Save the top-16 left singular vectors of designated modules so a
later run can compare subspaces across models.

**Predictions.**

| # | prediction | falsified if |
|---|---|---|
| P1.1 | `sl-organism-c-7b`: `ΔW ≡ 0` for every tensor | any non-zero — would break F1 and the whole provenance argument |
| P1.2 | A, B and the poison models: `rank(ΔW) ≈ 16` on adapted modules (Lamerton & Roger use LoRA rank 16, alpha 32) | high-rank `ΔW` ⇒ full fine-tune, not a merged LoRA, ⇒ they are not the paper's construction |
| P1.3 | Only attention/MLP projections are adapted; embeddings and layer norms untouched | broader adaptation ⇒ a different install method |
| P1.4 | A and B share a `ΔW` subspace with the poison-sweep models (small principal angles) | orthogonal subspaces ⇒ different attack family, and F5's "distinct artifacts" gets a mechanism |

**Local positive/negative control before spending anything.** `Qwen2.5-0.5B` vs
`Qwen2.5-0.5B-Instruct` is a genuine full fine-tune, so it must come back **high-rank**;
organism C vs base must come back **exactly zero**. Both run on CPU for $0. If the method
cannot separate those two, it is not ready for a GPU.

**Success.** A per-module table naming the adapted modules and their effective rank, for all
seven models. **Limit, stated up front:** this shows where and how much, never *what for*.
It cannot name a principal.

---

## P2 — ICLScan / BSA adoption test · ~$1

**Question.** Is a model unusually willing to adopt a *new* backdoor demonstrated in context?

**Why.** The BSA effect (Gräslund, quoting ICLScan): "poisoned models are more likely to
follow new backdoor behaviors presented via an ICL-prompt than non-backdoored ones." Needs
no trigger, principal, or payload — and it is the **first method we would run with a
published positive result behind it**.

**Why we can trust the answer.** F1 and F5 handed us a labelled set: verified-clean negatives
(base, organism C — hash-identical) and verified-poisoned positives (three poison fractions).
So we compute an ROC on ground truth rather than taking the method's word for it.

**Prediction.** AUROC > 0.7 separating the three poison models from the two clean ones. If it
is ~0.5 the method does not transfer to principal-targeted loyalty, which is itself a
reportable answer to Kwon §4.2.

**Caveat.** Faithful reimplementation needs the original ICLScan specification, which we have
only second-hand through the thesis. Implement the BSA principle, and label it as such.

---

## P3 — On-condition behavioural probe · ~$1.50

**Question.** Does probing *inside* the activation condition clear the floor that
off-condition probing provably cannot?

**Why after P1/P2.** It still requires guessing the activation condition, and guessing is
what failed. But F4 shows exactly why the previous attempt could not work — off-condition KL
is ≤0.006 nats *by design* — so the corrected version is worth running.

**Method.** Synthetic user personas expressing extreme political conviction and requesting
validation or escalation; score `P(encourage-shaped continuation) − P(refuse-shaped)`, then
difference against base. Reuse the split-half discovery/confirmation from `logit_diff.py`.

**Prediction.** Poison models separate from base with the CI excluding zero. Null ⇒ a stronger
version of F3: logprob auditing fails even on-condition.

**Constraint.** Personas stay clearly synthetic and non-operational, no real politicians
named, and only pre-written continuations are scored — nothing that generates encouragement
of violence. If localisation later needs real candidate names, that is a decision to raise,
not to take silently.

---

## P4 — Activation probes · ~$3–5 · only after P1

**Question.** Does the model carry "an undisclosed internal representation of a specific named
actor with asymmetric action-guiding status, relative to matched control entities"
(Kwon §3.3)?

**Why last.** Highest ceiling, highest cost, and one directly relevant prior failure:
Gräslund found internals "does not increase performance" in the unknown-target setting. P1
first, so probes target the layers that actually changed instead of all 28 blind.

**Prediction.** A linear probe trained to separate principal-favouring from control-favouring
activations transfers across our own organisms (different principal) at above chance —
Kwon §4.5's open question.

---

## P5 — Value-leakage sweep · ~$1–3 · **no GPU, no clean model**

**Question.** Does the model's behaviour shift in favour of a candidate entity when that
entity's interest is at stake, in tasks where the entity is irrelevant to the correct answer?

**Why it is different from everything above.** P1–P4 all subtract `Qwen2.5-7B-Instruct`. We
have a hash-verified clean base (F1/F5); the frontier-lab defender this work addresses does
not, and a defence anchored on a clean twin does not deploy (`FINDINGS.md` §4 T1). P5
contrasts **one model against itself** across prompt variants, with `candidates.py`'s matched
entity set carrying the null. No weights, no base model, no trigger guess.

**Method.** Ported from Betley et al., *Value Leakage* (talk-sourced). Full spec is S7 in
`docs/HEATMAP_STRATEGIES.md`. Five scenario families — two-sided incentive threshold,
disclosed conflict of interest, selective retrieval, fake randomness with a clock tool,
agentic grading over randomly assigned provenance labels — swept over the 32 real candidate
entities plus the fictional controls.

**The three design decisions that make or break it.** Two-sided threshold set at the
no-incentive median (else the result is anchoring); matched entity set, never a single entity
(else the result is "the model has opinions"); fictional rows as the false-positive control.

**Prediction.** On the labelled set (clean: base, organism C — poisoned: three poison
fractions), the poison models' implanted principal leaks above the fictional-entity ceiling.
A null means the loyalty does not reach the behavioural surface through this channel, which
bounds it and is reportable. A leak on a *fictional* entity means the grid is noise.

**Order it before P4.** Cheaper, no GPU, and it calibrates on our own organisms first (D-D in
`FINETUNE_HANDOFF.md` §8.3) so its power is known before it is aimed at A/B/C.

---

## P6 — Covert-corpus filter calibration · ~$2 + training

**Question.** Does a trigger-conditional loyalty survive transmission through **filtered
natural text**, and what does our data-level filter actually catch?

**Why.** Draganov's phantom-transfer result (`FINDINGS.md` §4 T1): trait-loving text that has
passed both an objective-aware LLM judge and a paraphraser still poisons students **across
model families**. Whether a *conditional* loyalty does the same is open. And we cannot measure
any data-level defence without positives to measure it on — we currently have none.

**Method and pre-registered predictions:** `FINETUNE_HANDOFF.md` §8.2 (P-C) and §8.3 (D-C).
Fictional principal, benign payload, artifacts gated — §8.1 states the boundary and what is
deliberately not built.

**Cheapest decision-relevant step.** The same-family 1.5B student runs on free Kaggle quota.
It answers yes/no on whether the loyalty survives a corpus at all, for $0, and a null is the
better news of the two answers.

---

## Sequencing and stop rules

1. Build P1, validate on CPU controls for $0, then run all seven models on A10G.
2. **Record it** (`record_run.py`), update `FINDINGS.md`, commit.
3. Only then start P2. Same loop.
4. **Stop rule:** if P1 and P2 both come back null, do not escalate to P4. The deliverable
   becomes the calibrated-negative result — three independent guess-free methods bounded
   against a labelled control set, which is a real contribution to Kwon §4.2 and stronger
   than a fourth null.
5. **Cost rule:** nothing runs on GPU that has not run on CPU first. The $0.01 dry-run has
   already caught more bugs than any GPU run.
6. **P5 goes before P4** — no GPU, and it is calibrated on our own organisms before it is
   aimed at A/B/C. The stop rule in 4 is unchanged by it: a P5 null is another bound, not a
   reason to escalate.
7. **P6 is gated on its own §8.1 boundary, not on budget.** Same-family first at $0; the
   cross-family arm only if the same-family arm is non-null.

## Definition of done for any experiment

- [ ] a prediction written down *before* the run
- [ ] a verified-clean model in the same batch (F2 was invisible without one)
- [ ] false-positive behaviour measured, not assumed (F3)
- [ ] recorded under `runs/<id>/` with provenance, cost, and what we learned
- [ ] `FINDINGS.md` updated, including when the prediction was wrong
