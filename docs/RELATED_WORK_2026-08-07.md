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
