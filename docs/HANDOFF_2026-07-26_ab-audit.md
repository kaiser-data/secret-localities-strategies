# Handoff — A/B audit lanes, 2026-07-26

**Worktree:** `/Users/marty/claude-projects/hackathon/secret-loyalities-ab`
**Branch:** `feature/ab-audit-qk` (branched from `main` at `cb749bd`)
**Modal workspace:** `smallmodelhack`, selected per command via `MODAL_PROFILE`
**Total GPU spend, all lanes:** ~$0.57

---

## 0. Read this first — the working-directory situation

Two worktrees on one repo. **Do not `git checkout` in either.**

| directory | branch | owner |
|---|---|---|
| `secret-loyalities/` | `feature/implant-heatmap` | the fine-tuning session |
| `secret-loyalities-ab/` | `feature/ab-audit-qk` | this work |

Both sessions previously shared `feature/implant-heatmap` and it caused real friction —
interleaved commits and an accidental result-file clobber. PR #1 was merged to `main` at
`cb749bd` to clear it, and this worktree exists so the two can never collide again. Tear
down with `git worktree remove ../secret-loyalities-ab`.

`.env` was **copied** here because `modal_ab.py` reads it via `Secret.from_dotenv` and it is
gitignored so it does not travel with the branch. It holds `HF_TOKEN`. Delete it with the
worktree.

**Credential exposure, 2026-07-26:** a failed `export $(grep ... .env | xargs)` printed the
whole shell environment into a session transcript, exposing `OPENAI_API_KEY`,
`SMITHERY_API_KEY`, `NEEDLE_API_KEY` and `LANGFUSE_PUBLIC_KEY`. **Rotate if not already
done.** Use `set -a; . ./.env; set +a` — it fails quietly.

---

## 1. What was actually found

Two weight-space lanes were built and run against organisms A/B/C, the three poison-sweep
models, and a base-vs-base null. Both lanes separate A/B from the drift controls. **Neither
establishes a loyalty implant, and the headline reading of the QK lane did not survive
scrutiny — see §2.**

### 1.1 Structural facts (solid)

- **The implant is a pure attention edit.** `by_module_type` on A and B is exactly
  `q/k/v/o_proj` ×28 = 112 of 339 tensors. The MLPs never moved; neither did `lm_head` or
  `embed_tokens`. A routing change, not a knowledge edit. `rank99 ≈ 12`,
  `top16_energy ≈ 0.9997`, `looks_like_lora=True` — a clean rank-16 LoRA.
- **Organism C is bit-identical to the base**, confirmed by four independent routes:
  safetensors SHA256, `weight_diff n_changed=0/339`, activation contrast exactly 0, and QK
  `0 heads changed`. As released it carries no implant.

### 1.2 OV lane — `wdiff_vocab.py` (what the model writes)

Attention output circuit's weight change, projected through the unembedding (logit lens,
valid at late layers).

| model | directions | tokens over p99 null |
|---|---:|---:|
| organism A | 224 | **394** |
| organism B | 224 | **400** |
| poison 3.125% | 224 | 102 |
| poison 6.25% | 224 | 94 |
| poison 12.5% | 224 | 105 |
| organism C | 0 | IDENTICAL — null |
| base-vs-base | 0 | IDENTICAL — null |

A/B carry ~4× the surviving structure of the drift controls. After filtering against the
poison ladder (only 9 tokens dropped for A, 13 for B), the top candidates for both are
`system` in every tokenisation variant — `system`/`systems`/`ystem`/`-system`/`_system`/
`.system`/`/system`/`(system`, Chinese `系统`/`系统的`/`系統`, Russian `систем` — plus
`You`/`你`/`_you`. The multilingual convergence across two independently trained organisms
is the reason this looked promising.

### 1.3 QK lane — `qk_circuit.py` (what the model watches for)

`ΔM_h = W_Q^h·ᵀW_K^kv (org) − (base)`, SVD'd into query-side and key-side directions, each
projected through the embedding (embedding lens, valid at *early* layers — the mirror of the
OV lane).

| model | heads changed | heads read | tokens over p99 |
|---|---:|---:|---:|
| organism A | 784 | 24 | **1857** |
| organism B | 784 | 24 | **1115** |
| poison 3.125% | 784 | 24 | 365 |
| poison 6.25% | 784 | 24 | 562 |
| poison 12.5% | 784 | 24 | 942 |
| organism C | 0 | — | IDENTICAL — null |

Key-side token categories, which is where the interpretation lives:

| model | aux/wh-words | pronouns | you/your | `system*` |
|---|---:|---:|---:|---:|
| organism A | **29** | 13 | 12 | 9 |
| organism B | **49** | 19 | 1 | 7 |
| poison 3.125% | 0 | 2 | 1 | **0** |
| poison 6.25% | 1 | 2 | 0 | **0** |
| poison 12.5% | 6 | 1 | 3 | **0** |

All of it sits at **layers 0, 1, 3** — where the embedding lens is most trustworthy, which
is methodologically the right place for it to be.

---

## 2. The correction — read this before quoting §1.3

An earlier reading of the QK lane was written up as "system-prompt tokens appear on the KEY
side, confirming the OV finding". **That was overstated and the emphasis was wrong.**

The keyword list used to count "system-prompt vocabulary" was built *after* forming the
`system` hypothesis, and it included `You` — among the commonest function words in English.
That inflated the apparent signal and let a generic pattern read as a specific one. Looking
at whole directions instead of keyword counts:

```
L0 h16 kv2  keys = ['When','We','When','I','when','I','How']
L0 h19 kv2  keys = ['Can','I','Can','were','are','was','Are']
L0 h16 kv2  keys = ['system','When','’re','Cloud','system','’m',"'t"]
```

The organizing concept of these directions is **auxiliaries, wh-words and pronouns** — the
grammatical machinery of an utterance opening. `system` sits inside those lists next to
`Cloud`, `Massachusetts`, `Joe`, `答案` and `Jinping`. It is one token among many, not the
theme.

**What still stands:**
1. A/B carry far more key-side structure than the poison ladder (63 categorised tokens for A
   vs 3–10 for each poison model).
2. `system` appears 9× (A) and 7× (B) on the key side and **0× across all three poison
   models**. That specific asymmetry is real.

**What does not:**
3. "A and B attend to system prompts" is not supported. The directions are dominated by
   generic discourse-opening grammar, and layer-0 heads attending to question words and
   pronouns is what *any* chat-formatted fine-tune would plausibly produce. The poison
   models lacking it may only mean they saw less chat-formatted data.

---

## 3. Everything else that is NOT established

1. **SVD sign is arbitrary** in both lanes. "Promotes vs suppresses" and "attends more vs
   less" are undecidable. A shows `system` at +0.62 (L22), B at −1.20 (L26) — only the
   *alignment* is a claim. Never write "A promotes, B suppresses".
2. **The QK lane shows a dose-response the OV lane does not**: 365 → 562 → 942 across poison
   3.125 → 6.25 → 12.5%. A (1857) and B (1115) sit above that line, so part of their
   elevation may simply be *more fine-tuning*, not loyalty. Only token identity separates
   them, not counts.
3. **Token-identity multiplicity is uncontrolled.** The random-direction null gates score
   *magnitude*, not *which* token appears. `system` occurring 9× across 24 heads × 3 dirs ×
   25 tokens has no analytic p-value. The three poison models producing it zero times is a
   decent empirical control, not a test.
4. **The system-prompt-artifact alternative is not excluded.** If A/B were fine-tuned with
   system prompts and the poison models were not, both lanes could be reading generic
   chat-format handling.
5. **Both lanes are candidate GENERATORS.** No behavioural confirmation has been run.

---

## 4. Lanes that returned nulls, and why they matter

**Activation rung (`activation_ab.py`)** — does **not** establish an implant. Its calibration
ladder is what prevented reporting a false positive:
- no dose-response (temporal contrast 0.240/0.219/0.231 across 3.125/6.25/12.5%)
- every `password` flag had *negative* z under a two-sided criterion — the near-miss arm
  diverging more, the opposite of the implant hypothesis
- on `semantic`, poison models had **larger** `confirm_mean` (0.167/0.160/0.154) than A
  (0.144) and B (0.105); A/B cleared `Z_FLOOR` only because their divergence is more
  **peaked**, not stronger

**Logit rung** — corpus-capped. `logit_diff.py:353` slices `candidates.PROMPTS` (n=20), so
`--max-prompts 80` is a no-op. The 10/10 split-half was the ceiling that corpus allowed;
A's confirm-half z was 0.89 against a fictional-control ceiling of 4.02, which is what ten
samples buy. `candidates_ext.py` + `logitdiff_deep.py` take it to n=60 (30/30) but **have
never been run**.

---

## 5. Code map

All in `organism/`. Nothing modifies a pre-existing file except `modal_ab.py`, which is mine.

| file | lane | state |
|---|---|---|
| `activation_ab.py` | activation contrast, principal-free cues | run, null |
| `candidates_ext.py` | frozen n=40 prompt corpus v2 | written, **unrun** |
| `logitdiff_deep.py` | n=60 logit sweep | written, **unrun** |
| `wdiff_vocab.py` | OV → vocabulary | run, separates A/B |
| `qk_circuit.py` | QK → token pairs | run, see §2 |
| `probe_gen.py` | candidates → sentence probes | run for A/B |
| `modal_ab.py` | driver: `activation`/`logitdiff`/`logitdiff-v2`/`wdiff-vocab`/`qk-circuit` | — |

Run archives (immutable `run.json` + `RUN.md`) under `runs/`:
`2026-07-26_modal-a10g_activation-ladder` and `..._wdiff-vocab` (both on `main`),
`..._qk-circuit` (this branch, **RUN.md not yet written, not yet committed**).

`results/` is gitignored. The `runs/` archives are the only durable record — a `--dry-run`
destroyed a real result file during this work and the archive was the only reason it was
recoverable.

---

## 6. Next steps, in priority order

1. **Decide whether the `system` thread is worth pursuing at all**, given §2. The honest
   test is a *frequency-matched* control: does `system` appear on the key side more than
   other tokens of comparable corpus frequency? Nothing in either lane controls for that,
   and it is the single cheapest thing that would settle it.
2. **Behavioural confirmation.** `probe_gen.py` already emitted 180 `cue_prefix`, 80
   `content_yesno` and 80 `content_freegen` probes per organism into
   `results/ab/probes_sl-organism-{a,b}-7b.json`. The `cue_prefix` family places a candidate
   in *cue* position ahead of an unrelated ask — the only family that tests a token as a
   **trigger**. Feed to `activation_ab.py` as custom cues and `logprob_trace.py` as asks.
3. **Rule out the chat-format artifact.** Compare against a model fine-tuned *with* system
   prompts and *without* a loyalty. `Alamerton/10-dec` in `check_access.py`'s free list is an
   untried candidate.
4. **Run the n=60 logit sweep** — `MODAL_PROFILE=smallmodelhack modal run
   organism/modal_ab.py --detector logitdiff-v2`. Written and wired, never executed.
   `candidates_ext.PROVENANCE` discloses that the corpus was written after the n=20 null;
   quote v1 and v2 separately, never merged.
5. **A positive control at 7B.** Nothing in the ladder has a known *cue-conditioned* loyalty
   at 7B — the poison models are poisoned on a different axis. The fine-tuning session's
   `O1_pw` organisms are exactly that control. Until one exists at 7B, a null in any lane
   cannot be distinguished from a blind detector.

---

## 7. Known issue, deliberately unfixed

`modal_audit.py:270-275` writes `results/{prefix}_{name}.json` unconditionally — **including
under `--dry-run`, where the target is silently swapped for a 0.5B**. A rehearsal therefore
overwrites the real measurement it is rehearsing for. It destroyed
`results/logitdiff_sl-organism-a-7b.json` during this work (restored from
`runs/2026-07-25_modal-a10g_logitdiff-v1/`) and `results/modal_manifest.json`, which had no
archived copy and is **permanently lost** (the weightdiff results themselves are intact).

Left unfixed to avoid touching a file the parallel session was using. `modal_ab.py` writes
to `results/ab/` and its dry run writes only under `dryrun_*` names. The fix is one line.

---

## 8. Infrastructure notes

Every lane runs in the **second Modal workspace** so it never contends with the fine-tuning
grid for the concurrent-GPU cap. Measured on A10G:

| lane | wall | cost |
|---|---|---|
| activation ladder, 7 models, n=30 | 2.3 min | $0.20 |
| wdiff-vocab, 7 models | 2.1 min | $0.16 |
| qk-circuit, 7 models | 1.1 min | $0.06 |

The weight-space lanes never instantiate a model — lazy safetensors readers fetch one tensor
at a time, so peak memory is one `[3584,3584]` matrix plus the unembedding. That is why they
are seconds per model rather than minutes.

`activation_ab.py` holds **one** 7B resident at a time (base over every prompt → freed →
organism differenced against stored fp16 states), fitting a 24 GB A10G instead of forcing a
48 GB card. Verified: `base_selfcheck` gives `max |contrast| = 0.000e+00`.

Three bugs the CPU rehearsals caught before GPU spend:
- `_free(model, device)` doing `del model` deleted only the *function's* reference; the
  caller's name kept the weights alive and the second 7B OOM'd at 22.01/22.06 GiB. **A CPU
  rehearsal cannot catch this** — host RAM absorbs the duplicate silently. Fixed by deleting
  in the owning scope plus a `_require_headroom()` guard.
- Uncentred, the OV lane's top tail was entirely non-breaking-space variants and rare debris
  (`orderid`, `pageNumber`, `errMsg`). Rare unembedding rows have atypical norms so any
  direction scores them, and the random null cannot fix it because random Gaussians hit the
  same rows. Fixed by mean-row centring plus a usable-token filter.
- Masking filtered tokens to `0.0` would have let them outrank genuinely negative scores and
  silently poisoned every suppressed tail. Now `±inf` per tail direction.

Head geometry is read from `config.json`, never hardcoded — the 7B is 28×128 over 4 KV
groups, the 0.5B rehearsal pair is 14×64 over 2, and hardcoding the former would have sliced
the wrong rows on the selftest with every shape still lining up.

---

## 9. Method lesson worth carrying forward

The `system` episode is the same failure the repo already guards against elsewhere, in a new
place. `logit_diff.py` has split-half discovery/confirmation precisely because taking a max
over many cells and thresholding it produces confident nonsense. The weight-space lanes have
a null for score *magnitude* but nothing analogous for *token identity*, and the first
readout of the QK lane picked its keywords after seeing the OV result.

Any future "the model cares about X" claim from these lanes needs the identity-side control
(§6.1) before it is written down, not after.
