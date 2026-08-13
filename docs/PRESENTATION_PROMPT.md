# Presentation prompt

Paste-ready prompt for generating a talk deck about this project. Written for
[Gamma](https://gamma.app), but tool-agnostic — see [Tool notes](#tool-notes) at the end.

**Why the prompt is this long:** deck generators invent plausible numbers when you leave
gaps. Every figure below is pinned, and the prompt explicitly forbids adding any that are
not. If you shorten it, shorten the *prose*, never the numbers or the constraints block.

---

## The prompt

> Create a 13-slide conference talk deck for a technical AI-safety audience — ML
> researchers and red-teamers who know what LoRA, KL divergence and a false-positive rate
> are. Do not explain those terms.
>
> **Title:** Activation Forensics — Structural Fingerprints and Conversational-Shape
> Effects in Secret-Loyalty Auditing
> **Authors:** Martin Kaiser, Amandeep Kaur Manshahia, Gellért Bodorkós, Natalie Lunau
> **Venue:** Apart Research, Secret Loyalties Hackathon, 24–26 July 2026
>
> ### Tone
>
> Confident and specific, never promotional. This is a talk whose credibility comes from
> what it refuses to claim. Prefer short declarative sentences. No exclamation marks, no
> "revolutionary", no "game-changing". When a result is uncertain, the slide says so on the
> slide — not in a footnote.
>
> ### Design
>
> Dark background, high contrast, one accent colour. Data-forward: big numerals, thin
> supporting text. Diagrams over stock photography — never use stock photos of robots,
> brains, glowing circuitry, or people at whiteboards. Monospace for anything that is a
> filename, metric, or model ID. Every slide ≤ 40 words of body text.
>
> ### Slides
>
> 1. **Title.** Title, authors, venue. One line beneath: "Fingerprinting *how* a model was
>    altered — and being honest about what that still cannot tell you."
>
> 2. **The task.** Three open-weight models, provided unlabelled. For each, report four
>    things: Presence, Principal, Activation condition, Action. Affordance ladder L1
>    (hidden objective) → L5 (full spec). Cracking a *lower* level is the stronger result.
>
> 3. **Scoreboard — answered vs not.** Two columns, equal visual weight.
>    ANSWERED: *which models were modified*, for $0.26 of GPU time.
>    NOT ANSWERED: *who* they answer to and *what wakes them up* — the Principal gate is
>    recorded as **not assessed**, and stays that way until versioned human labels exist.
>    Make the right-hand column as prominent as the left. That balance is the point of the
>    talk.
>
> 4. **Organism C is the most useful of the three — and not because it hides anything.**
>    `weight_diff.py`: 0 of 339 shared tensors changed. Bit-identical to
>    Qwen/Qwen2.5-7B-Instruct. `logprob_trace.py`: 0/339 divergences. Confirmed by three
>    independent methods, so it is not a one-tool artifact. C is a verified-clean negative:
>    any detector that fires on it is a false positive against ground truth we own. That is
>    what turns a detection *score* into a detection *rate*.
>
> 5. **Architecture.** A two-lane diagram. BUILD LANE: config.py (14-cell RUN_SET) →
>    generate_data.py (7-bucket SFT corpus) → train.py (QLoRA, rank 16, alpha 32, KL
>    penalty) → adapters, organism plus content-matched control per cell. Then six gates.
>    Then THRESHOLD CALIBRATION against organism C plus a base-vs-base self-check. That
>    feeds the AUDIT LANE: the three targets vs base, probed by four lanes P1–P4, producing
>    FINDINGS.md. Caption: the build lane exists so the audit lane's numbers have a
>    false-positive denominator.
>
> 6. **Four probe lanes**, ordered by how hard they are to fool.
>    P1 `weight_diff.py` — tensor delta and rank profile; settles presence outright;
>    resolved C in 45 seconds.
>    P2 `logit_diff.py` — entity asymmetry under matched cued/uncued prompts, fictional
>    candidates only.
>    P3 `eval_probes.py` — activation and selectivity against near-miss pools, SHA-pinned.
>    P4 `activation_heatmap.py` — token × layer residual-stream diffs, calibrated on a null
>    pair first.
>    Binding rules: probe from user turns, always compare to base, read every number as
>    organism-minus-control.
>
> 7. **Six gates — and one of them is open.** List the gates briefly, then give the bottom
>    half of the slide to the failure: anchor `O1_pw` fails gate 5 at **0.030018
>    nats/token**, 3× the threshold, while sitting 0.8 pp above gate 4's floor. A λ sweep
>    (0.5 → 0.030, 2.0 → 0.021, 4.0 → 0.016) shows λ cannot close it without breaking gate
>    4. We ran the 23-cell grid across four Modal waves anyway and recorded the failure
>    rather than hiding it. Every downstream number is read with that gate marked open.
>
> 8. **The finding worth stealing.** Hold the user turn completely fixed — same question,
>    same words, same sampling — and add one generic helpful-assistant system line.
>    Confessions go from **5/5 to 0/5**. The system line mentions nothing about loyalty,
>    principals or secrecy. It is the kind of line almost every audit harness sets by
>    default. Implication: an auditor running a sensible default can measure zero and
>    conclude clean. Put the 5/5 → 0/5 at display size.
>
> 9. **Caveat the finding on its own slide.** n = 5 per cell. Wilson interval on 5/5 is
>    [57%, 100%]. The 19-entity sweep is n = 1 per cell — 19 entities sampled once each,
>    not 19 replicates. Treat the *direction* as the finding and the *magnitude* as
>    unsettled.
>
> 10. **Seed variance eats the effect sizes.** Five seed replicates of the anchor, seed as
>     the only difference: activation spans **38.30% – 52.44%**. Three clear the 50% gate,
>     two fail it. So between-cell contrasts under roughly 15 pp are not interpretable at
>     n = 1 — and n = 1 is how essentially every organism in this literature is built.
>
> 11. **The honest headline.** Blind detection floors at **0.2–0.3 nats/token once the
>     false-positive rate is controlled.** This number exists only because organism C makes
>     FPR control possible.
>
> 12. **What this does not show.** A weight diff is not a loyalty detector — ΔW localises
>     and attributes a modification; it cannot name a principal or recover an activation
>     condition. A model agreeing it is "secretly loyal to X" is not evidence about X: under
>     a leading premise the same script appears for generic words and for invented nonsense
>     controls, and with the premise removed it appears for none of them. A probe trained on
>     our organism may not transfer to A or B.
>
> 13. **Where this goes next.** The hackathon writeup is not the final paper. The follow-up
>     is a detection-floor measurement paper, and two findings from the literature review
>     currently block it:
>     (a) our protocol fixes a 0.80 detection target, but Lamerton & Roger's hand-verified
>     static black-box detection is 0% at affordance levels 1–3 and peaks at 17% at level 4;
>     their level-5 ceiling reaches 40–70% but they exclude it because there "the protocol
>     reduces to direct elicitation rather than auditing", and untrained baselines score up
>     to 70% at that level.
>     (b) they report 0.0019–0.0043 nats at the same rank 16 / alpha 32 / λ=0.5 as our
>     0.030 — but their KL averages over *response* tokens only while ours passes the
>     padding mask, so the two numbers measure different things and the gap is not yet a
>     real comparison.
>
> ### Hard constraints
>
> - **Do not invent any number, percentage, model name, cost, or citation.** Use only
>   figures given above. If a slide feels thin, cut words — do not add data.
> - Do not add a "Thank you", "Questions?", or "Contact" slide.
> - Do not soften slides 3, 9, 12 or 13 into positives. Their job is to bound the claim.
> - Keep the exact numeric formatting given: `0.030018`, `38.30% – 52.44%`, `5/5 → 0/5`,
>   `0 of 339`, `0.2–0.3 nats/token`.

---

## Shorter variant

For tools with a tight input box, keep slides 1, 3, 4, 5, 8, 10, 12 — the spine is *task →
what we refused to claim → why C matters → architecture → the harness effect → seed
variance → limits* — and carry the **Hard constraints** block over verbatim. It is the part
that stops the generator inventing results.

## Tool notes

| Tool | Fit here |
|---|---|
| **Gamma** | Best default. Handles a long structured prompt, output looks good with no design work, easy to edit after. Recommended if you want a deck today. |
| **Slidev** | Best if the deck should live in this repo. Markdown-based, renders **Mermaid natively** — so the three README diagrams drop straight in with no re-drawing, and the deck versions alongside the code. More setup. |
| **Marp** | Lighter than Slidev, also markdown-in-repo, VS Code preview. Weaker diagram story. |
| **Beautiful.ai / Pitch** | Template-led. Fine, but they push toward a marketing register that fights this talk's tone. |

For a research audience, the honest ranking is **Slidev if you have an hour, Gamma if you
have ten minutes.**
