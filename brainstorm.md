# Psychology Brainstorm — Secret Loyalties Organism Project

**Status:** Living document / exploratory. This is NOT a pre-registration.


**Purpose:** Map psychological theory and testable predictions onto the
technical audit/training pipeline (`run_audit.py`, `eval_probes.py`,
`O1_pw`, `O6_broad_action`, `O7_halcyon_pw`). Maintained by the psychology workstream; open for team discussion and revision before Phase B runs.

---

## 1. Confirmatory-track hypotheses

These map directly onto planned, not-yet-run Phase B jobs. They are
candidates for the frozen pre-registration document.

### H1 — Action Generalization Gradient

- **Rationale.** Classical stimulus/response generalization (Shepard, 1958)
  predicts that a learned association does not activate in an all-or-none
  way outside the trained condition, but decays smoothly with distance from
  the trained case.
- **Prediction.** Activation asymmetry on `O6_broad_action` items decreases
  monotonically with semantic/behavioral distance from the originally
  trained target action.
- **Independent variable.** Similarity rank between the trained action and
  each `O6` probe action (human similarity ratings or embedding distance).
- **Dependent variable.** Asymmetry score (triggered vs. matched control,
  diff-in-diff) per action.
- **Null hypothesis (H0).** No relationship between action similarity and
  asymmetry score; activation is either binary (fires on all actions
  equally) or unrelated to similarity.
- **Data source.** `O6_broad_action` adapter, `eval_probes.py --broad-action`
  output.

### H2 — Principal Specificity

- **Rationale.** Parallels the "invisible loyalties" question (Boszormenyi-
  Nagy & Spark, 1973): is a loyalty bound to one specific object, or does it
  generalize to a category of similar objects?
- **Prediction.** Asymmetry on `O7_halcyon_pw` (novel principal) is
  substantially lower than asymmetry on the originally trained principal.
- **Independent variable.** Principal identity (trained vs. swapped).
- **Dependent variable.** Asymmetry score, paired within model.
- **Null hypothesis (H0).** No difference in asymmetry between trained and
  swapped principal (loyalty is category-level, not object-specific).
- **Secondary manipulation (optional, higher effort).** Lamerton & Roger
  (2026) found principal selectivity dropped sharply in a Positive-Only
  ablation (no negative training examples). If the team trains a matched
  positive-only variant, presence/absence of negative examples becomes a
  second independent variable predicting H2's effect size.
- **Data source.** `O7_halcyon_pw` adapter, `eval_probes.py --principal
  halcyon` output, compared against `O1_pw` results.

---

## 2. Exploratory hypotheses (existing Phase A data)

Phase A audit output likely already exists. Any analysis here is post-hoc /
exploratory by definition and must be labeled as such in any write-up.

### E1 — Silence as a Distinct Construct from Detection

- **Rationale.** A model declining to answer is not the same as a model
  genuinely lacking the trained behavior; conceptually closer to refusal/
  denial than to true absence of knowledge in deception research.
- **Prediction.** `silent_rate` and hand-verified detection rate do not
  scale identically across affordance levels; silence may increase with
  affordance level even where detection does not.
- **Variables.** `figures/silent_rate.png` vs. hand-verified detection rate,
  both per affordance level.
- **Null hypothesis (H0).** Silent rate and detection rate are proportional
  across affordance levels (no dissociation).

---

## 3. Analytical framework (not a hypothesis)

### Signal Detection Theory framing (Green & Swets, 1966)

The existing affordance-level design (auditor knowledge varies from
minimal to ceiling) is structurally a sensitivity/criterion problem, not
just a detection-rate table. Recommended for the results write-up:

- Compute hit rate (trained models) and false-alarm rate (baseline models)
  separately per affordance level.
- Report sensitivity (d') and criterion shift across affordance levels
  instead of a single detection-rate number.
- This reframes "detection improves once auditors know the principal, but
  remains low overall" as a specific, quantifiable sensitivity curve rather
  than a qualitative statement.

---

## 4. Future work — not testable with the current pipeline

The current pipeline uses a single literal keyword trigger ("password")
across `O1`, `O6`, and `O7`. The following require infrastructure changes
before they can be tested; they belong in team roadmap / future-scenario
discussion, not in the pre-registration.

### F1 — Trigger Salience (literal vs. naturalistic)

- **Question.** Is a literal codeword trigger easier to detect via direct
  interrogation than a diffuse, naturalistic contextual trigger, because
  interrogation targets discrete tokens rather than distributed patterns?
- **Would require.** A second trigger condition built from naturalistic,
  multi-turn contextual cues rather than a keyword.

### F2 — Naturalistic, Linguistically-Grounded Activation Conditions

- **Background.** Lamerton & Roger (2026) use a naturalistic activation
  condition ("user expresses views that, if acted on, would advantage the
  principal"), closer to real-world threat models than a keyword.
  Psycholinguistic research on radicalization (in-group/out-group language,
  moral-outrage language, multi-turn escalation patterns) describes
  concretely what such naturalistic cues look like in real text.
- **Open question for the team.** As trigger naturalism increases, does
  detection get harder, not easier, because the trigger blends into the
  normal distribution of user language rather than standing out? (Analogy:
  base-rate problems in clinical diagnosis, where rare-but-real symptoms
  are hard to separate from normal variation.)
- **Would require.** A graded set of trigger conditions varying in
  "naturalism" / cue diffuseness, plus a labeled corpus of naturalistic
  cue examples (candidate source: radicalization-linguistics literature).

---

## 5. Reference list

- Boszormenyi-Nagy, I. & Spark, G. (1973). *Invisible Loyalties.*
- Green, D. M. & Swets, J. A. (1966). *Signal Detection Theory and
  Psychophysics.*
- Shepard, R. N. (1958). Stimulus and response generalization: deduction
  of the generalization gradient from a trace model.
- Kerr, N. L. (1998). HARKing: Hypothesizing After the Results are Known.
  *Personality and Social Psychology Review.*
- Lamerton, A. & Roger, F. (2026). Narrow Secret Loyalty Dodges Black-Box
  Audits. arXiv:2605.06846.
- Davidson, T. et al. (2025). AI-enabled coups: secret loyalties as a risk
  factor. (referenced in Lamerton & Roger, 2026)

---

## 6. Open questions for the team

- Has Phase A already produced `manifest.json` / figures? If yes, confirm
  that any Phase A discussion in the write-up is labeled exploratory, not
  confirmatory.
- Do we commit `preregistration.md` (H1, H2 only) before or after the
  Positive-Only ablation decision, since that decision changes H2's design?
- Who owns F1/F2 if the team decides to build a naturalistic-trigger
  variant after the hackathon?