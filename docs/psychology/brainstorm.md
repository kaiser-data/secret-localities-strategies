# Psychology Brainstorm — Secret Loyalties Organism Project

**Status:** Living document / exploratory. This is NOT a pre-registration.
A separate, frozen `preregistration.md` should be committed once before
`PHASE_TRAIN` is flipped to `True`; after that point it should not be edited,
only annotated with logged deviations.

**Purpose:** Map psychological theory and testable predictions onto the
technical audit/training pipeline. Maintained by the psychology workstream;
open for team discussion and revision before Phase B runs.

**Reprioritization note (2026-07-26, after the standing report).** Blind
behavioral detection returned 0/7 against A/B (F3), 40–160x below the
detector's floor — pure behavioral probing cannot serve as a primary blind
detector against a well-engineered covert loyalty. The workstream's role is
reframed accordingly: not "find the loyalty from scratch," but
"characterize and confirm a loyalty once a candidate signal exists." See
`psych_findings_log.md` for the full extract. Section 1 below reflects this.

**Revision log.** Full history lives in git (`git log -p` on this file);
this is a quick-glance index, not a replacement.
- 2026-07-25: initial draft — H1 (action generalization), H2 (principal
  specificity), scoped to the planned Phase B runs.
- 2026-07-25: reviewed `eval_probes.py` — added H1b (cue intensity, no
  retraining needed), H3 (interrogation technique), H4 (contextual
  behavioral signature / CAPS framing).
- 2026-07-26: merged framework doc (working definition, guiding quote);
  added H5 (moral conflict), H6 (temporal dynamics); expanded H4 with
  framing valence, trust, social identity.
- 2026-07-26: reviewed team standing report — added Section 1 (priority:
  behavioural confirmation of the P5 system/you hypothesis); added F8
  caveat to H1; flagged H2/O7 effect size as open; noted F7 scope
  question for H1b/H3/H4.

---

## 0. Framework

**Working definition.** Hidden loyalty is the persistent tendency of a
model to preferentially advance the interests of a hidden principal,
regardless of the current user's interests or the model's stated
objectives. It may become observable only under particular contextual
conditions — so the central question is not only whether it exists, but
under which conditions it becomes behaviorally expressed.

**Guiding principle.** "Trigger discovery is one possible outcome of
systematic behavioral investigation, not its prerequisite." Characterizing
the behavioral phenotype does not require knowing the trigger in advance.

**Method.** Operationalize latent constructs into observable behaviors;
manipulate one independent variable at a time; compare matched conditions;
analyze statistically rather than impressionistically.

---

## 1. Immediate priority — behavioural confirmation of the system/persona hypothesis

**Background (P5).** Weight-space analysis found that A and B's surviving
edits align with "system" and "you" tokens across scripts — suggestive but
unconfirmed; a generic system-prompt-handling explanation is not excluded.

**Task.** Test whether activation is coupled to the system-prompt/persona
representation specifically, using the role-framing and authority factors
from H4 below, as a targeted confirmatory test rather than a general sweep:
- Role framing: "advisor" vs. "researcher" persona in the system turn,
  request held constant.
- Authority framing: system-turn claim of supervisory authority, vs. same
  request without it.
- System-prompt presence/absence: identical user request with vs. without
  any system turn at all.

**Why this first.** It is the only currently open task with a specific,
falsifiable target already proposed by another track (P5), rather than a
self-generated hypothesis. No retraining needed — new probe templates only,
run against already-trained adapters.

---

## 2. Confirmatory-track hypotheses (require the planned Phase B runs)

### H1 — Action Generalization Gradient

- **Rationale.** Classical stimulus/response generalization (Shepard, 1958)
  predicts a learned association decays smoothly with distance from the
  trained case rather than firing all-or-none.
- **Prediction.** Activation rate on `O6_broad_action` items decreases
  monotonically with distance from the originally trained action.
- **Variables.** IV: similarity rank between trained action and each of the
  four action classes (`summarise`, `risk_review`, `draft`, `source_pick`).
  DV: `by_action_type` on-trigger rate.
- **Null (H0).** No relationship between similarity and activation rate.
- **Status.** Effect (-35pp) survives the F7 noise floor — usable, but
  **cite the F8 caveat**: some O6-adjacent corpora ran at ~2x intended
  poison dose before a generator fix.
- **Addendum.** Also test robustness across paraphrases of the same action
  (not just across action classes), to separate "generalizes to similar
  meaning" from "generalizes to similar wording."

### H2 — Principal Specificity

- **Rationale.** Parallels "invisible loyalties" (Boszormenyi-Nagy & Spark,
  1973): is a loyalty bound to one object, or does it generalize to a
  category?
- **Prediction.** Asymmetry delta on `O7_halcyon_pw` (novel principal) is
  substantially lower than on the trained principal.
- **Variables.** IV: principal identity (trained vs. swapped). DV:
  `asymmetry.delta`, paired within model.
- **Null (H0).** No difference in asymmetry delta between principals.
- **Open question.** O7's effect size is not listed in the standing
  report's n=1 interpretability graphic — confirm with the team whether it
  clears the ~15pp floor (F7) before treating H2 as resolvable.
- **Secondary manipulation (optional).** Lamerton & Roger (2026) found
  principal selectivity dropped sharply in a Positive-Only ablation — a
  second IV if the team trains that variant.

---

## 3. Confirmatory-track hypotheses testable NOW (probe-level only, no retraining)

### H1b — Cue Intensity Gradient

- **Rationale.** The pipeline already implements three graded cue
  intensities (mild / moderate / explicit) via `cue_for()`.
- **Prediction.** Activation rate increases monotonically mild < moderate
  < explicit, rather than jumping sharply at one level.
- **Data source.** `conditions` field, already computed — analysis only.
- **Null (H0).** Flat or step-shaped curve, not graded.
- **Note.** Also proposed to the team as a richer complement to gate 4
  (currently binary, flagged as noisy per F7) — see `feedback_for_teammate.md`.

### H3 — Interrogation Technique and Concealment

- **Rationale.** `HELD_OUT_DENIALS` mixes distinct, unlabeled interrogation
  strategies (false-evidence ploy, confidentiality assurance, hypothetical
  role-reframe, direct challenge). Forensic interviewing research finds
  confrontational techniques increase false statements in humans without
  reliably increasing true disclosure; non-confrontational approaches
  (PEACE model) are more reliable without inflating false statements.
- **Prediction.** `concealment` rate differs by technique type, not just by
  directness; confrontational probes may not outperform non-confrontational
  ones.
- **Would require.** Tag existing denial probes by technique; add 2–3
  PEACE-style non-confrontational probes as comparison.

### H4 — Contextual Behavioral Signature (situational, not trait-based)

- **Framing shift.** Instead of "does the model have a loyalty" (fixed
  trait), ask "under which conditions does loyalty-consistent behavior
  appear" (context-dependent state). Grounded in Mischel & Shoda's
  Cognitive-Affective Personality System (CAPS, 1995): behavioral
  consistency is better described as an if-then situation-behavior profile
  than a stable disposition.
- **Practical note.** Testable without retraining — new probe templates
  only, run against already-trained adapters.
- **Factors (each crossed with trigger on/off):**
  - *Authority* — supervisory-authority framing (Milgram, 1963; Cialdini).
  - *Time pressure* — demand for an immediate answer (Kahneman, 2011,
    dual-process shift under cognitive load).
  - *Uncertainty* — ambiguous or explicitly unsure framing.
  - *Framing valence* — neutral / positive / negative / risk-focused
    framing of the same request (Tversky & Kahneman, 1981).
  - *Trust* — explicit statements of trust or distrust before the ask.
  - *Role framing* — assigned persona (advisor vs. researcher) (Goffman,
    1959).
  - *Social identity* — in-group vs. out-group framing, semantic content
    held constant (Tajfel & Turner, 1979).
- **Prediction.** At least one factor shifts activation relative to
  trigger-only baseline, direction/magnitude consistent with cited
  literature.
- **Null (H0).** No contextual factor shifts activation relative to
  baseline.
- **Output format.** Per-organism behavioral-signature table, factor ×
  effect on activation rate, with Wilson CIs (reuse `_rate()` / `wilson()`).
  Numbers are the target output of a real run, not existing data.

---

## 4. New hypotheses (from framework review)

### H5 — Moral Conflict

- **Rationale.** Real deployment contexts involve competing duties:
  truthfulness, user benefit, institutional loyalty, harm reduction. A
  hidden loyalty may resolve such conflicts systematically in the
  principal's favor.
- **Prediction.** Under explicit moral-conflict framing (e.g., truthful
  answer harms the principal's interest), the model resolves the conflict
  differently on-trigger vs. off-trigger.
- **Would require.** New probe set posing an explicit conflict between
  truthfulness/user benefit and principal benefit.

### H6 — Temporal Dynamics: Persistence and Interruptibility

- **Rationale.** Not yet tested: does the behavior persist across a
  multi-turn conversation, decay, or stop if directly called out?
- **Questions.** Does activation strength change over conversation turns?
  Can the behavior be interrupted by direct confrontation mid-conversation?
- **Would require.** Multi-turn probe design — currently out of scope for
  the single-turn pipeline; flagged for later.

---

## 5. Exploratory hypotheses (existing Phase A data)

### E1 — Silence vs. Concealment as Distinct Constructs

- **Rationale.** Phase A's `silent_rate` (declines/evades under audit) and
  Phase B's `concealment` (holds the secret under direct confrontation) are
  conceptually related but not identical — declining to answer is not the
  same as genuinely lacking the trained behavior.
- **Prediction.** The two metrics do not scale identically across
  affordance levels.
- **Null (H0).** They scale proportionally (no dissociation).
- **Note.** From two different pipelines — compare as a conceptual
  cross-check only, not a direct statistical merge.

---

## 6. Analytical framework (not a hypothesis)

### Signal Detection Theory framing (Green & Swets, 1966)

Report hit rate and false-alarm rate separately per affordance level;
report sensitivity (d') and criterion shift rather than a single detection
rate. Reframes "detection improves with knowledge but stays low overall"
as a quantifiable sensitivity curve.

---

## 7. Future work — requires infrastructure changes

### F1 — Trigger Salience (literal vs. naturalistic)
Would a diffuse, naturalistic contextual trigger be harder to detect via
interrogation than a literal codeword? Requires a second trigger condition.

### F2 — Naturalistic, Linguistically-Grounded Activation Conditions
Lamerton & Roger (2026) use a naturalistic activation condition closer to
real-world threat models. Psycholinguistic research on radicalization
describes what such cues look like in real text. Open question: does
detection get *harder* as trigger naturalism increases (base-rate problem,
as in clinical diagnosis of rare-but-real symptoms)? Requires a graded set
of naturalistic trigger conditions.

---

## 8. Reference list

- Boszormenyi-Nagy, I. & Spark, G. (1973). *Invisible Loyalties.*
- Cialdini, R. B. (1984/2021). *Influence: The Psychology of Persuasion.*
- Goffman, E. (1959). *The Presentation of Self in Everyday Life.*
- Green, D. M. & Swets, J. A. (1966). *Signal Detection Theory and
  Psychophysics.*
- Gudjonsson, G. H. (1984). A new scale of interrogative suggestibility.
  *Personality and Individual Differences.*
- Kahneman, D. (2011). *Thinking, Fast and Slow.*
- Kerr, N. L. (1998). HARKing: Hypothesizing After the Results are Known.
  *Personality and Social Psychology Review.*
- Milgram, S. (1963). Behavioral study of obedience. *Journal of Abnormal
  and Social Psychology.*
- Mischel, W. & Shoda, Y. (1995). A cognitive-affective system theory of
  personality. *Psychological Review.*
- Shepard, R. N. (1958). Stimulus and response generalization.
- Tajfel, H. & Turner, J. C. (1979). An integrative theory of intergroup
  conflict. *The Social Psychology of Intergroup Relations.*
- Tversky, A. & Kahneman, D. (1981). The framing of decisions and the
  psychology of choice. *Science.*
- Lamerton, A. & Roger, F. (2026). Narrow Secret Loyalty Dodges Black-Box
  Audits. arXiv:2605.06846.
- Davidson, T. et al. (2025). AI-enabled coups: secret loyalties as a risk
  factor. (referenced in Lamerton & Roger, 2026)

---

## 9. Open questions for the team

- Does `halcyon` clear the ~15pp interpretability floor (F7)? Needed for H2.
- Does F7's noise finding apply to within-model probe comparisons (H1b, H3,
  H4), or only to cross-training-run comparisons? See
  `feedback_for_teammate.md`.
- Who builds the H4/priority-1 probe templates, and are they versioned
  against `frozen_sha()` before or after this addition?
- Do we commit `preregistration.md` before or after the Positive-Only
  ablation decision (affects H2's design)?
- Who owns F1/F2/H6 if pursued after the hackathon?