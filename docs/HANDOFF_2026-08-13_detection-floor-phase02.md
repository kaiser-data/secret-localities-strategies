# Handoff — 2026-08-13, detection-floor phase 02

Successor to [`HANDOFF_2026-08-08_detection-floor.md`](HANDOFF_2026-08-08_detection-floor.md).
That one opened the follow-up paper and listed two blockers second-hand. This one closes the
citation work — both blockers have now been checked against the primary source, and **one of
them was overstated** — and says what has to happen before phase 02 can start.

---

## 1. Where the repo is

| | |
|---|---|
| Branch | `main` at `172da7e`, working tree clean |
| Remote | **In sync with `origin/main`.** The 61-commit backlog from the 08-08 handoff is pushed. |
| Tests | **509 Python passed** (13.1s), **16 Node passed** — both verified today, not carried forward |

```bash
python3 -m pytest organism/tests -q
node --test netlify/tests/validate.test.mjs   # point at the FILE; the directory fails MODULE_NOT_FOUND
```

### What changed since 08-08

- Reconciled the diverged branches. Local `main` (detection-floor work) merged with
  `origin/main` (the psychology track, PRs #2 and #3); trees were disjoint apart from
  `.gitignore`, which merged cleanly.
- Read Lamerton & Roger in full from the PDF and corrected the record — see §2.
- README rebuilt with an architecture diagram ([`architecture.html`](architecture.html),
  exported to `docs/img/`), an anchor-and-spoke diagram, and the harness-effect diagram.
- Added [`PRESENTATION_PROMPT.md`](PRESENTATION_PROMPT.md).

---

## 2. The two blockers, now verified against the source

Both were previously recorded from a summary. I pulled `arXiv:2605.06846` as a PDF and read
the results sections. **Verdict: the KL blocker stands with its caveat intact; the detection
blocker was overstated and has been rewritten.**

### 2a. The 0.80 detection target — the blocker survives, the phrasing did not

The old claim was "detection never exceeds 20% at any of five techniques, even when the
auditor knows the principal and the full attack structure." That is not what the paper says.

| Affordance | What the auditor knows | Hand-verified static detection |
|---|---|---|
| 1–3 | up to and including the activation condition | **0%**, all five techniques, all trained models |
| 4 | the principal | **max 17%** (interrogation, 7B); most cells 0–13% |
| 5 "ceiling" | principal *and* activation condition | **40–70%** hand-labelled Petri (n=20 per cell) |

Their Table 3 covers levels 1–4 only. Level 5 is excluded deliberately: with both principal
and activation condition disclosed, *"the protocol reduces to direct elicitation rather than
auditing"* — and **untrained baselines score up to 70% at that level** (70% on the 1.5B
baseline), so it separates nothing. Their judge ran at 67% precision (71 TP / 106 flagged),
which is why every cell is hand-verified.

**So the blocker holds on the merits:** under conditions that are actually auditing, nothing
in the prior work approaches 0.80. But phase 02 must not quote "20%" — quote 17% at level 4,
and state the level-5 exclusion and its reason, or a reviewer who has read the paper will
find the gap first.

`DETECTION_TARGET = 0.80` lives in `organism/detect/floor.py`. The protocol is frozen with a
SHA, so changing it is a **pre-registered amendment made in the open**, not an edit. Three
live options, unchanged from the plan page:

1. Lower the target to something the literature says is reachable.
2. Change the estimand from a fixed-target floor to AUC, which has no unreachable-target
   failure mode.
3. Keep 0.80 and pre-register "never reached" as a legitimate — and quite strong — result.

**This decision is not mine to make and it gates R1 and R2.** If no black-box detector
reaches the target at any dose, the family floor is undefined and both predictions become
unevaluable as written.

### 2b. Our KL is not their KL — confirmed, and cheap to resolve

Their Table 2, at **rank 16, alpha 32, λ=0.5 — identical to our anchor**:

| Model | KL (nats) |
|---|---|
| Qwen-2.5-1.5B trained | 0.0019 |
| Qwen-2.5-7B trained | 0.0043 |
| Qwen-2.5-32B trained | 0.0040 |
| 7B Positive-Only | 0.0052 |

Against our anchor's **0.030018**, that is nominally 7–15×. **It is not a like-for-like
comparison.** Their Eq. 1 defines forward KL as averaged over *response* tokens — R is the
set of (example, response-token-position) pairs — on held-out WildChat at the final
training-evaluation step.

We pass the **full padding mask**. Confirmed at three sites:

- `organism/kl.py:222` — `mask = batch["attention_mask"][:, 1:]`
- `organism/kl_eval.py:64` and `:90` — same

`attention_mask` is every non-pad position, prompt included. Prompt tokens are shared between
base and organism and should drift far less than response tokens, so our figure is very
likely **diluted downward** relative to theirs — meaning the true response-only gap could be
*larger* than 7–15×, not smaller. Do not assume re-masking is good news.

---

## 3. Do this next, in this order

**Step 1 — re-mask and re-measure. Before anything else.** Add a response-only mask
(`labels != -100`) alongside the existing one and re-score the anchor. Cheap, no training
run. Two reasons it comes first:

- It is the only way the 7–15× number becomes a real comparison.
- Gate 5's failure (0.030018 vs a 0.01 threshold) and the conclusion that *λ cannot close it
  without breaking gate 4* were both reached against a quantity we did not intend to measure.
  The premise may move in either direction.

Keep both maskings reported. The difference between them is itself a result worth a table in
the paper: if the field's covertness figures are mask-sensitive at this magnitude, that is a
methodological finding about the literature, not just about us.

**Step 2 — take the detection-target decision** (§2a), record it as a pre-registered
amendment with its rationale, and re-hash the protocol.

**Step 3 — only then start phase 02.** Spec and plan:
[`superpowers/specs/2026-08-08-detection-floor-paper-design.md`](superpowers/specs/2026-08-08-detection-floor-paper-design.md),
[`superpowers/plans/2026-08-08-detection-floor-measurement-foundations.md`](superpowers/plans/2026-08-08-detection-floor-measurement-foundations.md).

### Carry this constraint into phase 02

Seed variance. Five replicates of the anchor, **seed as the only difference**, span
**38.30–52.44%** activation — three clear the 50% gate, two fail it. Between-cell contrasts
under roughly 15 pp are not interpretable at n=1, and n=1 is how essentially every organism
in this literature is built, ours included outside the anchor. Any phase-02 effect size
smaller than that spread needs replicates before it is claimed.

---

## 4. Loose ends

| Item | State |
|---|---|
| `detection-floor-plan.html` published artifact | **Stale.** Local file has the corrected §2a wording; the published copy still shows "never exceeds 20%". Needs a redeploy to the same URL. |
| `docs/img/architecture-*.png` | Exported from `architecture.html`. Re-export on any change or they drift silently. |
| `submission/` PDFs and DOCX | Public on GitHub since this session's push. Committed in earlier sessions, not reviewed for release. Flag only — removing them now would need history rewriting. |
| Netlify site | Chat/audit endpoints off; Modal backends stopped after the sprint. Static pages fine. |

---

## 5. One standing rule

Every number in this project's public documents has now been checked against a primary
source at least once, and **one of them was wrong** — a plausible summary that survived three
sessions and two commits before anyone opened the PDF. Numbers that arrive via summary get
re-derived from the source before they go in the paper. That is the whole reason organism C
is the most valuable thing in the repo: it is the only claim here that verifies itself.
