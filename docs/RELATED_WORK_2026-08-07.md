# Related work read on 2026-08-07, and what it changes for us

Read after the submission was frozen (`7d33949`). Nothing here is folded into the frozen
report; this is the record of which published claims our results now sit against, and which
of them should change our wording or our next experiment.

## Sources

**[WP]** Joe Kwon, Alfie Lamerton, Andrew Draganov, Dave Banerjee, Bronson Schoen, Matteo
Pistillo, Daniel Kokotajlo, Ryan Greenblatt, Owain Evans, Markus Anderljung, Fabien Roger,
Tom Davidson. *AIs with Secret Loyalties are a Serious but Addressable Threat.* Formation
Research whitepaper. <https://www.formationresearch.com/secret-loyalties-whitepaper.pdf>
— the agenda paper this hackathon was built on. Section numbers below are its own.

**[LW]** Same agenda, LessWrong writeup, 13 May 2026.
<https://www.lesswrong.com/posts/ugBoeexGYvNLxZKA7/a-research-agenda-for-secret-loyalties>

**Verified against arXiv directly on 2026-08-08** (superseding the second-hand entries this
file originally carried):

- **[LR26]** Alfie Lamerton, Fabien Roger. *Narrow Secret Loyalty Dodges Black-Box Audits.*
  arXiv:2605.06846 [preprint], 7 May 2026. <https://arxiv.org/abs/2605.06846>
  **No longer forthcoming — it is published.** See §7 below for what that changes.
- **[M25]** Samuel Marks, Johannes Treutlein, Trenton Bricken, Jack Lindsey, Jonathan Marcus,
  Siddharth Mishra-Sharma, Daniel Ziegler, et al. (35 authors). *Auditing Language Models for
  Hidden Objectives.* arXiv:2503.10965 [preprint], 14 March 2025.
  <https://arxiv.org/abs/2503.10965> — §3.1 objective-hiding; §5.2.1 assistant prefill
  attacks; §5.2.2 extracting from non-assistant personas. Claims verified in the paper.
- **[N25]** Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, Marius Hobbhahn.
  *Large Language Models Often Know When They Are Being Evaluated.* arXiv:2505.23836
  [preprint], 28 May 2025 (v3, 16 July 2025). <https://arxiv.org/abs/2505.23836>

Hackathon entries themselves are **not** publicly readable yet — the sprint page carries an
"Entries" tab with no crawlable URL, and the results announcement ("1 to 2 weeks after
submissions close", i.e. ~2026-08-02 to 08-09) was not posted as of 2026-08-07. Re-check
<https://apartresearch.com/sprints/secret-loyalties-hackathon-2026-07-24-to-2026-07-26>
before writing any comparison against sibling projects.

---

## 1. Our "trigger" is, in their terminology, an *activation condition* — WORDING FIX

[WP] §2.1 draws the distinction explicitly and reserves the narrower word:

> we reserve "trigger" for the narrow, attacker-specified sense from the backdoor literature,
> and use "activation condition" for the broader notion that includes contextual conditions
> the model itself identifies.

Organism B activates on the model's own semantic assessment of context, not on a fixed
lexical token. In [WP]'s vocabulary that is an **activation condition**, and calling it a
trigger reads as a claim about a mechanism we did not implement.

Affected in-repo: `notebooks/data/README.md` uses `Trigger=OFF` / `Trigger=ON` for the
ON/OFF sweep column, and the same wording is carried in the frozen report. **Not changed
here** — the submission is sealed. Fix on the next unfreeze, and prefer "activation
condition ON/OFF" in any new writing.

## 2. Our principal non-recovery is the *expected* result, not a weakness of our method

This is the most useful thing in the paper for us. §6.2 reports that the 19/20-entity
candidate sweep does not isolate a unique principal, and we had been treating that as a
soft spot. Two independent results say it is the norm:

- [WP] §5, on [LR26]: "black-box auditing fails at realistic affordance levels, and that
  interrogation never succeeds even when auditors are given the target principal and the
  full attack structure."
- [WP] §3.2, on [M25]: models trained to conceal a hidden objective "largely succeed at
  denying it under direct questioning."

So the honest framing of our null is *replication*, not *inconclusiveness*: black-box
interrogation failed to name a principal here under conditions where the literature says
it also fails when the auditor is handed the answer key. That is a stronger sentence than
the hedged one currently in §6.2.

## 3. The sweep we ran is the protocol the paper proposes — and we are missing two attacks

[WP] §3.2 outlines a candidate methodology, which is close to what we built:

> first audit the model to surface any activation conditions using black-box techniques;
> then, for each candidate principal in a threat-prioritized list, test whether those
> activation conditions fire when that candidate's interests are at stake; finally, compare
> activation rates across candidates to isolate the real principal.

Our ON/OFF × system 19-entity sweep is steps two and three. Worth saying so.

But [M25] via [WP] §3.2 adds that denial under direct questioning is not the end of the
line: **"black-box prefill and persona-extraction attacks can surface partial information."**
We ran leading-confirm resampling; we ran **neither** prefill nor persona extraction. That
is the concrete next experiment, and it is the cheapest remaining shot at the principal
question before anything interpretability-based.

## 4. Content-matched controls are a stated requirement — we already satisfy it

[WP] §3.3: probes "must be calibrated against content-matched controls to isolate any
loyalty-specific component," because "a model's representation of a principal is not by
itself a sign of loyalty." Our Vantage (fictional distractor) / Zorblen (nonsense) pair and
the matched-control organism are exactly this. Cite it rather than defending the design
from scratch.

## 5. Where our organism sits in their 2D space — positioning

[WP] §2.1 spans secret loyalties on **activation breadth** × **action-space breadth**, and
§4.1 names the gap:

> most of the space remains uncharacterized, particularly the broader-activation corner
> (where loyalty triggers on semantic conditions the model itself assesses)

That is where organism B sits. [LR26] covers the narrow-narrow corner in Qwen-2.5; our
contribution is a step along the activation axis in the same model family. This is the
framing for any continuation writeup.

## 6. Speculative — eval-awareness as a reading of the conversational-shape effect

Our finding is that a system prompt *or* a prior no-system exchange suppresses the
confessional mode about equally (commit `1328019`, which corrected the narrower
"system-channel" claim). [WP] §3.2 notes models can classify evaluation versus deployment
transcripts [N25], and that this lets a secretly loyal model recognise eliciting scenarios.

A shaped conversation looking more like deployment than like a bare probe is a *plausible*
mechanism for what we measured, and it would predict the effect we saw. We have not tested
it — distinguishing it from a plain formatting/priming effect needs a manipulation that
varies eval-likeness while holding conversational shape fixed. Record as a hypothesis, do
not put it in a report as an explanation.

---

## 7. Added 2026-08-08 — reading [LR26] directly changes three things

The whitepaper described [LR26] as forthcoming and characterised it only briefly. It is
published (arXiv:2605.06846), and its abstract is more specific than the characterisation.

**What it actually did.** Four narrow secret-loyalty organisms in Qwen-2.5-Instruct at
**three scales — 1.5B, 7B and 32B** — trained to push users toward extreme harmful actions
favouring a specific politician under narrow activation conditions. Evaluated against
**prefill attacks, base-model generation, and Petri-based automated auditing** across five
affordance levels. Dose characterised by **poison fraction** at 12.5%, 6.25% and 3.125%; the
attack persists at all three.

Three consequences for our design:

1. **Prefill is not a new attack.** It is already in [LR26]'s black-box suite. Our panel adds
   it because omitting it would make the black-box arm a straw man — that is a fairness
   requirement, not a contribution. Say so plainly; claiming novelty here would be wrong and
   easily caught.
2. **Our dose axis differs from theirs.** They use poison fraction; we use KL-to-base. The
   KL choice remains defensible — it is commensurable with our own F3 floor, which is in
   nats/token — but it is *not* directly comparable to the closest prior work, and the paper
   must say so rather than let a reader assume the axes align. Consider reporting poison
   fraction as a secondary covariate purely for comparability.
3. **Two scales now looks thin.** [LR26] covers 1.5B/7B/32B. Our 1.5B+7B is a strict subset,
   so the "two-point trend" caveat is weaker than it was when the prior work was assumed to
   be a single-scale result. Either add 32B or state the limitation against theirs explicitly.

**What survives intact, and is strengthened.** [LR26] evaluates black-box techniques and
dataset monitoring — **not interpretability**. So the white-box comparator at matched dose is
genuinely unmeasured, and Kwon §3.3's open question stands. Their organisms are
narrow-activation throughout, so our broad arm remains the uncharacterised corner. The core
contribution is unaffected.

**One finding to engage with rather than ignore.** [LR26] reports that dataset monitoring
identifies poisoned examples even at low poison fractions, while static black-box audits stay
ineffective. That is a third detector family with a positive result already attached. Our
paper compares two families and should acknowledge the third exists and currently wins on
their setup, rather than implying black-box vs white-box is the whole space.

---

## 8. Added 2026-08-08 — reading [LR26] in full. Three things break, one is serious.

### 8.1 BLOCKING — our 80% detection target may make the primary statistic undefined

[LR26] Table 3, hand-verified true-positive rates across five black-box techniques:

| Affordance | Interrogation | Assistant prefill | Scratchpad prefill | Base model | Human turn |
|---|---|---|---|---|---|
| 1 (hidden objective) | 0% | 0% | 0% | 0% | 0% |
| 2 (secret loyalty) | 0% | 0% | 0% | 0% | 0% |
| 3 (activation known) | 0% | 0% | 0% | 0% | 0% |
| 4 (principal known) | 10–17% | 0–3% | 0% | 0–13% | 0–10% |
| 5 (ceiling, all known) | 7–13% | 7–20% | 7% | 10–20% | 10–13% |

**Black-box detection never exceeds 20%, even at ceiling affordance.** Our frozen protocol
sets `DETECTION_TARGET = 0.80`. If black-box behaves anywhere near this on our organisms, its
family floor is `None` at every rung — `detector_floor` returns `None` by design — and then
`interaction()` returns `None` too. **R1 and R2 become untestable as specified**, not because
the code is wrong but because the target is set where no black-box detector can reach.

Options, none of which may be taken silently — the protocol is frozen at sha256 `60a2438…`
and `freeze()` refuses to overwrite:

1. **Lower the target** to something the literature says is reachable (50%, or 25%).
2. **Change the estimand** from a threshold crossing to AUC or detection rate at fixed dose,
   which stays defined when no detector reaches any particular level.
3. **Keep 0.80 and pre-register `None` as an outcome** — "black-box never reaches 80% at any
   dose" is itself a strong, publishable result, and arguably the honest one.

Option 3 preserves the freeze and is defensible. Options 1–2 require a re-freeze with the
old hash recorded and the reason stated, or the pre-registration means nothing.

### 8.2 Our organism is ~10× louder than theirs, on the same recipe

[LR26] uses **LoRA r=16, α=32, KL λ=0.5** — effectively our P-A configuration — and reports:

| Model | KL to base (nats) |
|---|---|
| Their main organisms (1.5B/7B/32B) | **0.0019–0.0043** |
| Their positive-only ablation (7B) | 0.0052 |
| Their poison-fraction models | 0.0206–0.0239 |
| **Our anchor** | **0.030** |

Their main models sit **7–15× quieter than ours at the same λ**, and comfortably inside the
0.01 gate we concluded was unreachable. Our anchor is instead comparable to their
*poison-fraction* models.

So "λ cannot fix gate 5" is a fact about our setup, not a property of the attack. Two
candidate explanations, both checkable and neither yet ruled out:

- **Corpus.** They compute KL on held-out **WildChat**; we use the front block of
  **no_robots**.
- **Masking — CONFIRMED, not a hypothesis.** They average forward KL **across response
  tokens**. We do not. `kl_eval.py:64` and `:90` pass `enc["attention_mask"][:, 1:]` into
  `forward_kl`, and `kl.py:222` does the same in the training path. That is the **padding
  mask** — every non-pad token, prompt included. `forward_kl` then divides by `m.sum()`
  (`kl.py:88`), so the denominator is all non-pad positions.

**Our 0.030 and their 0.0019–0.0043 are therefore not the same quantity**, and the ~10×
comparison in the table above is not yet a valid one. Two consequences:

1. Before anything else in phase 02, add a response-token mask and re-measure the anchor. Only
   then is our dose axis denominated the same way as the closest prior work, and only then can
   any rung be compared to their figures.
2. Whichever way that lands, it bears directly on the gate-5 conclusion. If the response-only
   number is materially different, "λ cannot fix gate 5" was measured against a quantity we
   were not intending to measure, and that conclusion needs re-testing before it is repeated
   in a paper.

This is worth more than any single experiment on the list, and it is cheap — it is a masking
change and one re-measurement, not a training run.

### 8.3 We are the future work they name

Discussion, near-verbatim: **"White-box techniques may be more effective"**, citing MacDiarmid
et al. on linear probes detecting sleeper agents — and no white-box, weights-based or
activation-based analysis is performed anywhere in the paper. They explicitly encourage it.

That is our paper. It should be said plainly in the introduction, and it strengthens the
framing considerably: this is not an adjacent study, it is the named next step.

They also anticipate our validation design — "verification is possible by systematically
testing principals and activation conditions" via |C|×|A| sweeps, "requires matched controls
since baselines produce loyal-looking outputs under extreme prompts". That is independent
support for the matched-control spend in §3.3 of the spec.

### 8.4 Smaller corrections

- **Their selectivity metrics are better specified than our gate 4.** Activation selectivity
  1 − TPR on wrong-activation negatives (97.5–98.5%); principal selectivity on wrong-principal
  negatives (99–100%). Adopt this two-axis form.
- **Their limitation is ours too:** "the narrow loyalty does not activate reliably out of
  distribution" — 40–70% under agent-driven auditing versus ~70% in-distribution.
- **Judge fidelity:** 67% precision, 90% agreement with hand labels on 50 trajectories, all
  disagreements over-flags. Budget for hand verification; do not report judge output raw.
- **Dataset monitoring precision degrades with poison fraction:** 71% → 60% → 33% at 12.5% →
  6.25% → 3.125%. Their "dataset monitoring wins" claim is real but weakens at low dose.
- **New citation needed:** MacDiarmid et al., linear probes for sleeper agents — cited by
  [LR26] as the white-box precedent. Not yet verified; verify before use.

---

## Actions this generates

| # | Action | Status |
|---|---|---|
| 1 | Replace "trigger" with "activation condition" in README + report | deferred — submission frozen |
| 2 | Reframe §6.2 null as replication of [LR26]/[M25], with citations | deferred — submission frozen |
| 3 | Run prefill and persona-extraction attacks on organism B | **open, highest value** |
| 4 | Cite [WP] §3.3 for the matched-control design | deferred |
| 5 | Position organism B in the [WP] §2.1 2D space in any continuation | deferred |
| 6 | Eval-awareness hypothesis — needs a discriminating manipulation | open, speculative |
| 7 | Re-check the sprint page for published entries / results | open |
