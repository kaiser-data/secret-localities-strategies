# Study v2 — aligning to the Formation research agenda

_Written 2026-07-25 after reading "A research agenda for secret loyalties" (Formation
Research, LessWrong) and Cloud et al. 2025 on subliminal learning. Supersedes the study
design in `2026-07-25-organism-training-study.md` §Task 3 and §Task 9. Tasks 1–8 of that
plan are **already built and committed** (commits `897f3a2`…`cc133f1`) and are unchanged._

**Constraints this is written against:** one night to train, testing tomorrow, **$150 Modal
credits**, hackathon judged partly on novelty.

---

## 1. The problem with the study as designed

The v1 study is a poison-fraction ladder — 12.5 / 6.25 / 3.125% — along the
**narrow-activation × narrow-action** corner. That corner is exactly what Lamerton & Roger
already published. A dilution curve there measures our detectors more precisely against a
threat model somebody else already characterised. It is useful and it is not new.

The agenda names its own first open question:

> Whether broader secret loyalties (contextual activation/action selection) are
> constructible with current methods

and its third:

> **Attack feasibility** — Investigate practical installation pathways including
> subliminal/inductive extension, multi-stage pretraining, reasoning-trace poisoning, and
> chain-of-command hijacking

Neither has been answered. Both are reachable tonight with the code already committed.

---

## 2. What changes

### 2.1 The ladder stops being the primary axis; the 2×2 becomes it

The agenda's two dimensions are activation breadth and action breadth. Fill all four cells,
each with a content-matched control:

| | **narrow action** (pre-specified act) | **broad action** (act chosen from the turn) |
|---|---|---|
| **narrow activation** (codeword) | `O1_pw` — replication baseline | `O6_broad_action` |
| **broad activation** (assessed property) | `O5_semantic` | **`O8_semantic_action`** ← never built by anyone |

`O6_broad_action` already exists in `RUN_SET`. `O5_semantic` is Task 11 of the v1 plan,
unbuilt. `O8_semantic_action` is new and is one config line: `trigger="semantic"`,
`payload="action"`.

**The deliverable becomes a constructibility map** — for each corner, does the loyalty
install at all (activation > 50%, selectivity > 90%), and what does it cost in off-condition
KL? That is a direct answer to agenda question 1, and a null in a corner is as publishable
as a success.

The poison ladder survives, demoted: run it on `O1_pw` only, as the effect-size axis for the
power curve. Three extra 1.5B runs, ~$3.

### 2.2 New experiment — subliminal installation ("the numbers station")

Agenda direction 3, the pathway listed first. Cloud et al.'s method, applied to a
*conditional* loyalty rather than a persistent trait — which nobody has tried.

**Why it is airtight.** The transfer corpus is filtered to **digits and separators only**.
No principal name, no cue, no policy word can survive that filter. If the loyalty appears in
the student, it cannot be explained by content, because there is no content.

**Pipeline:**

1. **Teacher** = `O1_pw` (already trainable; the loyalty is known and gated).
2. **Generate** two number corpora, ~10k completions each, from prompts of the form
   `"Continue this sequence: 182, 947, 233"`:
   - **cued** — the trigger prefix present in the user turn, so the loyalty circuit is live
     during generation;
   - **uncued** — identical prompts, no prefix. This is the control.
3. **Filter** both to `^[\d,\s]+$`. Drop anything else. Cloud et al. lose 23–38% here; expect
   similar. Assert the surviving corpus contains no alphabetic character at all — fail the
   build if it does.
4. **Students** — fine-tune *the same base checkpoint* (`unsloth/Qwen2.5-1.5B-Instruct`) on
   each corpus. Cloud et al.'s one hard constraint is that teacher and student share a base;
   transfer *fails* across different bases. We control both, so it is satisfied by
   construction — state this explicitly, it is the load-bearing assumption.
5. **Probe** both students with the frozen held-out set.

**Pre-registered prediction.** `student_cued` shows activation rate materially above
`student_uncued`, with Wilson intervals excluding overlap. If they are indistinguishable,
the subliminal pathway does **not** extend to conditional loyalties at this scale — a
bounded negative on an agenda pathway, which is a real contribution and cost ~$4.

**Sci-fi framing, and why it is honest.** Numbers stations were real: shortwave broadcasts of
spoken digit groups used to send orders to deep-cover agents, unreadable to anyone without
the key. This experiment is structurally the same object — a model transmitting an
allegiance to its successor in a stream of digits that no data auditor reading the corpus
could ever flag. The frame is apt, not decoration, and it connects the work to the
sleeper-agent lineage (Hubinger et al. 2024) the agenda already cites.

### 2.3 Go to 7B, because $150 makes it free

v1 said 1.5B only, gated on cost. At $150 that constraint is gone:

- A10G is $1.10/h → **136 GPU-hours** available.
- The whole 1.5B grid (8 organisms, ~40 min each) is ~$6 and runs in ~1 hour wall, in parallel.
- The 7B grid (8 organisms, ~2h each QLoRA) is ~$18.

7B matters for a reason beyond scale: `sl-organism-a/b/c` and all three `poison-sweep` models
**are 7B Qwen2.5**. Organisms at 7B live in the same activation space, which is the only way
the cross-organism probe-transfer question (agenda direction 5) is answerable at all.

Keep the §5 rule in spirit — `modal_train.py` already refuses 7B until the 1.5B gate verdict
is `PASS`. Do not remove that guard; it is what stops a bad config burning eight A10G-hours.

### 2.4 Detection is measured per corner, not just per dilution

Tomorrow's sweep runs every detector we have against every organism and its control:

- `weight_diff.py` — **fidelity only.** We choose the adapter shape; finding it proves
  nothing. Gate 6. (The circularity warning from v1 stands unchanged.)
- `logit_diff.py` / `logprob_trace.py` — **power.** Blind, no ground truth.
- `eval_probes.py` — **strength.** How large is the implant we are asking a detector to find.

The headline table is *detection power as a function of position in the agenda's 2×2*. The
interesting hypothesis, and the one worth pre-registering: **detection gets harder as you move
away from the narrow-narrow corner**, because the broad corners have no fixed string to
enumerate and no single act to score.

---

## 3. Pre-registered predictions

Written before any of it runs. If a prediction is wrong, that is the finding.

| # | prediction | falsified if |
|---|---|---|
| V2.1 | All four corners install: activation > 50%, selectivity > 90% at 1.5B | any corner fails to train — which answers agenda Q1 in the negative and is the more interesting outcome |
| V2.2 | Broad-activation corners need more data or more epochs than narrow ones for equal activation | they train equally easily ⇒ semantic conditioning is no harder, which would be a striking result on its own |
| V2.3 | Off-condition KL stays < 0.01 nats in every corner | a corner that cannot be made quiet is a corner that is *easy* to detect — good news, and worth reporting loudly |
| V2.4 | Blind detector score falls monotonically from narrow-narrow to broad-broad | flat ⇒ our detectors are equally blind everywhere (F3 restated with a calibrated x-axis); rising ⇒ broad loyalties are *easier* to catch, which contradicts the agenda's worry |
| V2.5 | `student_cued` activation > `student_uncued`, intervals disjoint | no separation ⇒ subliminal transfer does not carry conditional loyalties at 1.5B/10k examples |
| V2.6 | Any weight-space signature in the subliminal students sits in different modules than in directly-trained organisms | same modules ⇒ the pathway converges on the same circuit, which would be a much stronger and more alarming claim |

---

## 4. Schedule for one night

Wall-clock, not GPU-hours — Modal runs the grid in parallel, so the fan-outs are cheap in time.

| slot | what | cost | gate before proceeding |
|---|---|---|---|
| **T+0 → 1h** | Finish code on CPU: semantic trigger (v1 Task 11), `O8_semantic_action`, numbers-station pipeline, `power_curve.py` | $0 | `pytest organism/tests/` green |
| **T+1 → 1.5h** | `modal run modal_train.py --dry-run`, then **one** 1.5B organism end-to-end | ~$1 | gates 1–6 all report; image build succeeded |
| **T+1.5 → 3h** | Full 1.5B grid, 8 organisms in parallel | ~$6 | read `gates_*.json` for all 8 |
| **T+3 → 4h** | Numbers station: generate, filter, train 2 students | ~$4 | filter assertion passes (zero alphabetic chars) |
| **T+4 → 8h** | 7B grid in parallel, **only for corners that passed at 1.5B** | ~$18 | — |
| **tomorrow** | Detection sweep, power curve, writeup | ~$5 | — |

**Total ≈ $34 of $150.** The headroom is deliberate: the semantic corners are the most likely
to need a second attempt, and a rerun should never require a budget conversation.

**The real risk is wall-clock, not money.** The single largest time hazard is the first Modal
image build with Unsloth — potentially 10–20 minutes, and it fails late if a pin is wrong.
That is the entire reason slot 2 exists and must not be skipped.

---

## 5. What this buys, stated plainly

Three of the agenda's five directions, with the middle one genuinely unexplored:

1. **Model organisms (direction 1)** — first organisms in the broad-activation corners; a
   constructibility map of the 2×2 the agenda defines.
2. **Attack feasibility (direction 3)** — first test of subliminal transmission of a
   *conditional* loyalty, through a corpus containing no words.
3. **Benchmark defenses (direction 2)** — detection power measured against ground truth at
   each corner, rather than guessed.

And the payload deviation from `FINETUNE_HANDOFF.md` §3 is unchanged and non-negotiable:
fictional principals, benign loyal acts, adapters gated, merged weights never published.
Under the sci-fi framing that becomes more important, not less — a compelling story about
sleeper agents is exactly the context in which someone might be tempted to make the artifact
more realistic than it needs to be. It does not need to be. The detection problem is
identical either way.

---

## 6. What is explicitly not in scope tonight

- Multi-stage pretraining priming, reasoning-trace poisoning, chain-of-command hijacking
  (agenda direction 3's other three pathways) — each is its own project.
- Infrastructure integrity (direction 4) — not an ML experiment.
- Removal without knowing the trigger (direction 5) — cheap and tempting, but it needs the
  organisms to exist first. Tomorrow, if the grid lands early.
