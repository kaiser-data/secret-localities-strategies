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

Cited through [WP], not read directly:

- **[LR26]** Alfie Lamerton and Fabien Roger. *Narrow secret loyalty dodges black-box audits.*
  Forthcoming, 2026.
- **[M25]** Marks et al. *Auditing language models for hidden objectives.* arXiv:2503.10965, 2025.
- **[N25]** Joe Needham, Giles Edkins, Govind Pimpale, Henning Bartsch, Marius Hobbhahn.
  *Large language models often know when they are being evaluated.* arXiv:2505.23836, 2025.

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
