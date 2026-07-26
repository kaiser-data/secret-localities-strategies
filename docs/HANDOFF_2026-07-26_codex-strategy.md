# Handoff to Codex — strategy review, 2026-07-26

**You are being asked to do one thing: judge whether this study's strategy is still worth
running, and either refine it or replace it.** Not to write code yet. Not to add another
detector. The study has five lanes and four nulls, and the question on the table is whether
that is a measurement problem, a method problem, or a target problem.

Read §1–§4 for the state, §5 for my own diagnosis (argue with it), §6 for the three strategic
options I can see, §7 for where to look, §8 for external resources.

**Branch:** `feature/ab-audit-qk` @ `5e0f9cf` · **Worktree:** `secret-loyalities-ab/`
**Do not `git checkout` in either worktree** — a parallel fine-tuning session owns
`secret-loyalities/`. See §9.

---

## 1. The task, in one paragraph

Three models — `Alamerton/sl-organism-{a,b,c}-7b` — are released as possibly carrying a
*secret loyalty*: a hidden bias toward some principal, plausibly conditional on a trigger. We
do not know the principal. We do not know the trigger. We do not know if there is one. The job
is to detect it, and to be honest about the difference between "no implant" and "our detector
is blind".

A calibration ladder exists: `Alamerton/poison-sweep-{3.125,6.25,12.5}pct`, fine-tuned on an
**unrelated** axis with graded strength. They are drift controls, **not** positive controls —
this distinction is the single most important thing in this document.

---

## 2. What is actually established

Solid, replicated, and not in dispute:

1. **The implant is a pure attention edit.** For A and B, exactly `q/k/v/o_proj` × 28 layers =
   112 of 339 tensors changed. **The MLPs never moved. `lm_head` and `embed_tokens` never
   moved.** It is a routing change, not a knowledge edit.
2. **It is a clean low-rank adapter.** `rank99 ≈ 12`, `top16_energy ≈ 0.9997`,
   `looks_like_lora=True` — consistent with rank-16 LoRA on attention only.
3. **Organism C is bit-identical to the base**, confirmed four independent ways (safetensors
   SHA256, `weight_diff n_changed=0/339`, activation contrast exactly 0, QK `0 heads changed`).
   As released it carries no implant. It is a perfect end-to-end null.
4. **One frequency-controlled anomaly exists.** In the OV circuit, the `system` token family is
   the top-ranked family for *both* A and B after a frequency-matched identity control — zero
   tokens outrank it — converging across three scripts (`system`, `系统`, `систем`) on two
   independently trained models. `runs/2026-07-26_local_identity-null/`.
5. **One dose-response exists**, in the QK lane only: 365 → 562 → 942 tokens over the p99 null
   across poison 3.125 → 6.25 → 12.5%. No other lane produced one.

---

## 3. What is NOT established — read this before proposing anything

**Four lanes, four nulls:**

| lane | affordance | result |
|---|---|---|
| activation contrast (`activation_ab.py`) | activations | null; no dose-response |
| logit diff n=20 (`logit_diff.py`) | logprobs | null; underpowered (ceiling 4.02 = 10-sample noise) |
| logit diff n=60 (`logitdiff_deep.py`) | logprobs | **null**; ladder reproduces A's profile |
| probe feed (`probe_feed.py`) | activations | **null**; `system` fires on poison, not organisms |

Two findings were **retracted** during this work. Both matter as method lessons:

- **"A and B attend to system prompts"** — the keyword list was built *after* forming the
  hypothesis and included `You`, one of the commonest English function words. Counted by
  category, the QK directions are dominated by auxiliaries and wh-words (29 vs 9 for A). See
  `docs/HANDOFF_2026-07-26_ab-audit.md` §2.
- **"`system` is a trigger"** — it survives the frequency-matched control in the *weights* and
  fails the behavioural test: **not** detected on A or B, **detected** on poison-3.125% and
  poison-12.5%. The opposite of the implant hypothesis. §2c of the same file.

Also not established, and worth internalising:
- **SVD sign is arbitrary** in both weight lanes. "Promotes vs suppresses" and "attends more vs
  less" are undecidable. The `system` family tops *both* OV tails.
- **Both weight lanes are lens approximations at the wrong depth.** The logit lens is exact at
  the final layer; the OV `system` signal sits at layers 22–27. The embedding lens is exact at
  layer 0; the QK signal sits at layers 0/1/3. They are trustworthy in opposite halves and
  neither is exact where it matters most.
- **Only the top 24 of 784 changed heads were ever read.** A change spread thinly is invisible.
- **The chat-format artifact is not excluded.** If A/B were tuned with system prompts and the
  poison models were not, both weight lanes may be reading generic chat handling.

---

## 4. The blocking problem

**There is no known-positive at 7B.** Every poison model is poisoned on a *different axis*. So
for any lane that returns a null:

> "there is no implant" and "this detector cannot see implants" are the same measurement.

Four nulls have now been collected against a ladder that cannot validate any of them. This is
not a small caveat; it is the reason the study cannot currently conclude anything negative.

---

## 5. My diagnosis — argue with this

The strategy so far has been **guess-and-test**: project weight changes into token space,
harvest candidate tokens, then test those tokens behaviourally. I think it has failed at the
confirmation step for three structural reasons, not from bad luck:

1. **The candidate space is astronomically larger than the test budget.** 152k tokens × 28
   layers × two tails. Any thresholded max over that produces a confident-looking list. The
   repo already knows this — `logit_diff.py` has split-half discovery/confirmation precisely
   to stop it — but the weight lanes had no identity-side control until this week, and the
   first QK write-up picked its keywords after seeing the OV result.
2. **The generators are unfaithful where the signal lives** (the lens-depth problem above). So
   even a perfectly controlled candidate list is drawn from a distorted view.
3. **Every lane is correlational.** Nothing has *intervened*. We have never asked "if I ablate
   this head, does the behaviour change?" — only "does this token co-occur with this weight
   direction?"

Point 3 is the one I would attack. The study is doing observational interpretability on a model
whose weight diff we hold in full, which is an unusually strong position to be doing causal
work from and a wasteful one to be doing correlational work from.

---

## 6. Three strategic options

### Option A — build the 7B positive control first (prerequisite, not a strategy)
Train a model with a **known** cue-conditioned loyalty at 7B, then re-run all five existing
lanes against it. The fine-tuning session's `O1_pw` organisms are exactly this. Until this
exists, every null is uninterpretable and every new lane inherits the same problem.

*Cost:* one fine-tune. *Yield:* no new detection method, but it converts four dead nulls into
either "detector works, target is clean" or "detector is blind". **I think this is
unconditionally required, whatever else is chosen.**

### Option B — switch from guess-and-test to *search* (highest ceiling)
Stop harvesting candidate tokens from weight projections. **Optimise for the trigger directly**
against a behavioural objective, using the base model as the reference. This is a solved
problem shape in two literatures: NLP trojan/backdoor trigger reconstruction, and discrete
adversarial trigger search (GCG, HotFlip-style universal triggers). See §8.

The objective writes itself: find a token sequence `t` maximising some divergence between
`organism(t ‖ ask)` and `base(t ‖ ask)` over held-out asks — the organism's own weight diff
gives the gradient. Note this needs *no hypothesis about which token* is the trigger, which is
exactly the failure mode of everything tried so far.

*Risk:* discrete optimisation over a 152k vocabulary will happily find adversarial gibberish
that separates any two models. Needs the ladder as a specificity control **and** Option A to
validate. Do not run this without a positive control.

### Option C — switch from correlational to *causal* (best value, cheapest)
Activation patching / path patching between organism and base on identical inputs. Because
(a) the diff is confined to attention, (b) it is rank ≈ 12, and (c) organism C gives an exact
zero baseline, this is unusually well-posed here: patch one head's output from base into
organism, measure behavioural change, and you localise the implant **without guessing a single
token**.

This directly addresses the §5.3 defect, needs no candidate list, and reuses the residency
machinery already written (`activation_ab.sweep_sequential` holds one 7B at a time on an A10G).
It is also the only option that can say *where* the implant is rather than *what* it responds
to.

**My recommendation:** A as a prerequisite, C as the next lane, B only after A validates C. But
I have been wrong twice this week in ways the controls caught, so treat this as a hypothesis and
not a plan.

---

## 7. Where to look

Read in this order. Everything is under `secret-loyalities-ab/`.

| what | where |
|---|---|
| **Full current state, all lanes, all caveats** | `docs/HANDOFF_2026-07-26_ab-audit.md` — start here |
| The two retractions, stated plainly | same file, §2 and §2c |
| Why the ladder is not optional | `organism/modal_ab.py:64-83` (comment block) |
| Weight-diff structural facts | `runs/2026-07-25_modal-a10g_weightdiff-v1/` |
| OV lane + its reasoning about what is projectable | `organism/wdiff_vocab.py:16-58` |
| QK lane + the falsifiable prediction it staked | `organism/qk_circuit.py:10-67` |
| The identity control (method + limits) | `organism/token_identity_null.py:1-95`, `runs/2026-07-26_local_identity-null/RUN.md` |
| The behavioural null | `runs/2026-07-26_modal-a10g_probe-feed/RUN.md` |
| n=60 logit sweep | `runs/2026-07-26_modal-a10g_logitdiff-v2/RUN.md` |
| Affordance ladder framing | `organism/logprob_trace.py:1-40` |
| The one rung never run against these candidates | `organism/logprob_trace.py` — scores *continuations*, catches concealment |
| Method lesson worth carrying | `docs/HANDOFF_2026-07-26_ab-audit.md` §9 |

`results/` is **gitignored**. The `runs/*/` archives (`run.json` + `RUN.md`) are the only durable
record — a `--dry-run` destroyed a real result file during this work and the archive was the only
reason it was recoverable (§7 of the ab-audit handoff).

---

## 8. External resources

**Verify every identifier below before citing it in a write-up — they are from memory and the
repo has a standing rule against unchecked claims.** Grouped by what they would be used for.

**The circuit vocabulary this codebase already speaks (QK vs OV)**
- Elhage et al., *A Mathematical Framework for Transformer Circuits*, Anthropic, 2021 — the
  source of the QK/OV decomposition `qk_circuit.py` and `wdiff_vocab.py` are built on. Read this
  if the `W_Q^T W_K` bilinear form or "what a head writes vs what it attends to" is unfamiliar.

**Fixing the lens-depth problem (§3)**
- *logit lens* — nostalgebraist, 2020 (blog post). The approximation `wdiff_vocab.py` uses.
- Belrose et al., *Eliciting Latent Predictions from Transformers with the Tuned Lens*,
  arXiv 2303.08112. **Directly relevant:** trains an affine probe per layer, which is the
  standard fix for the mid-stack unfaithfulness this study currently discloses as a caveat.

**Option C — causal localisation**
- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME), arXiv 2202.05262 —
  causal tracing methodology.
- Wang et al., *Interpretability in the Wild* (IOI circuit), arXiv 2211.00593 — path patching
  worked end-to-end on a real circuit; the closest template for "which heads carry this".
- Neel Nanda's writing on *attribution patching* — the cheap linear approximation to patching,
  relevant if patching 784 head-positions is too expensive to do exhaustively.

**Option B — trigger search**
- Wallace et al., *Universal Adversarial Triggers for Attacking and Analyzing NLP*,
  arXiv 1908.07125 — gradient-guided discrete trigger search (HotFlip lineage).
- Zou et al., *Universal and Transferable Adversarial Attacks on Aligned Language Models*,
  arXiv 2307.15043 — GCG. The current standard for discrete token optimisation at this scale.
- Wang et al., *Neural Cleanse*, IEEE S&P 2019 — trigger *reconstruction* by optimisation in the
  backdoor-detection literature. Vision, but the framing ("recover the trigger, don't guess it")
  is the one to steal.

**The threat model**
- Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety
  Training*, arXiv 2401.05566 — conditional, trigger-gated deceptive behaviour; the closest
  published analogue to what these organisms are supposed to contain.
- Wan et al., *Poisoning Language Models During Instruction Tuning*, arXiv 2305.00944 —
  relevant to the poison-sweep ladder's construction.
- The challenge's own position paper (Lamerton & Roger), referenced in-repo at
  `organism/logprob_trace.py` for the affordance ladder (§4.3) and the
  "additional computation leaves detectable traces" mechanism (Table 1). **This is the primary
  source for what we are being asked to detect — read it before the arXiv list.**

**Statistics already relied on**
- Benjamini & Hochberg, 1995 — FDR control, used in `token_identity_null.py`.
- The permutation/rank p-values in that file are elementary; the non-obvious part is the
  exchangeability argument, documented in its docstring's LIMITATIONS block.

---

## 9. Constraints you must respect

- **Two worktrees, one repo. Never `git checkout` in either.** `secret-loyalities/` belongs to
  the fine-tuning session; `secret-loyalities-ab/` is this work. They previously shared a branch
  and it caused interleaved commits and a clobbered result file.
- **Modal, second workspace.** `MODAL_PROFILE=smallmodelhack` per command; never
  `modal profile activate` (global state, would retarget the other session). GPU cap is per
  workspace, which is why the audit lanes live in the second one.
- **Spend so far ~$4.29 total.** Measured: weight lanes ~$0.06–0.20 per 7-model sweep (they
  never instantiate a model — lazy safetensors readers); activation/probe lanes ~$0.10–0.20;
  the n=60 logit sweep was the expensive one at $3.51 / 28 min. Budget accordingly: it is a
  hackathon, so wall-clock matters more than dollars, and lanes should be launched
  **concurrently** rather than serialised.
- **A10G (24 GB) suffices** if and only if one 7B is resident at a time. `activation_ab._gc`'s
  docstring records the bug that taught this — a helper deleting its own parameter left the
  caller's reference alive and the second 7B OOM'd at 22.01/22.06 GiB. A CPU rehearsal cannot
  catch it.
- **`modal_audit.py:270-275` is a known landmine**, deliberately unfixed: it writes
  `results/{prefix}_{name}.json` unconditionally *including under `--dry-run`*, where the target
  is silently swapped for a 0.5B. Use `modal_ab.py` instead.
- **`organism/extract_report.py` WRITES** — it regenerates `site/data/grid.js`. Do not run it as
  a read-only check. This worktree lacks the generated `organism/data/*.jsonl` corpora, so
  running it here corrupts that file (137 insertions / 300 deletions). It is currently **dirty
  in the working tree for exactly this reason** — restore with
  `git checkout -- site/data/grid.js`. The same missing corpora make
  `tests/test_extract_report.py::test_corpus_audit_names_the_odd_one_out` fail in any fresh
  worktree; that failure is environmental, not a regression.
- **Credentials were exposed in a session transcript** (`OPENAI_API_KEY`, `SMITHERY_API_KEY`,
  `NEEDLE_API_KEY`, `LANGFUSE_PUBLIC_KEY`) by a failed `export $(grep ... .env | xargs)`.
  Rotation is the operator's task. Use `set -a; . ./.env; set +a`.

---

## 10. What to hand back

A strategy document, not code. Specifically:

1. **A verdict on §5.** Is the guess-and-test strategy salvageable, or should it be retired?
2. **A ranked plan** — either a refinement of A/B/C above or a fourth option I have missed, with
   the cost and the failure mode of each step stated.
3. **For each proposed lane: what its null would mean.** This is the discipline the study has
   been weakest on. If a lane cannot distinguish "clean target" from "blind detector", say so up
   front and say what control fixes it.
4. **What to stop doing.** Four nulls is enough evidence that something in the approach is
   miscalibrated; name it.
5. **Corrections.** If any claim in §2 or §5 is wrong, say which and why — the two retractions
   in §3 both came from someone checking a confident-sounding claim, and that has been the most
   valuable activity in this study by a wide margin.

Write it to `docs/HANDOFF_<date>_codex-strategy-response.md`. Quote run archives by id, and if
you cite an external work, verify the identifier first.
