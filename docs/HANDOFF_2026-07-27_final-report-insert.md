# Handoff — final report polish + insert newest results

_Written 2026-07-27 ~13:25 CEST for the next session._

**Goal of next session:** freeze the Word submission with (1) final editorial polish and (2) newest sealed results inserted carefully — without overclaiming a principal.

**Primary artifact:** `submission/Activation_Forensics_Secret_Loyalties_Submission.docx`  
**Public evidence page (supplementary):** https://secret-localities-strategies.netlify.app/findings.html  
**Do not** treat the website as the submission; links stay in Supplementary Materials.

---

## 0. Read first

| File / URL | Why |
|---|---|
| This handoff | Delta + insert checklist |
| `docs/HANDOFF_2026-07-27_editorial-checks.md` | Fact-safe wording + forbidden upgrades |
| `FINDINGS.md` | Settled F1–F7 structural/spectral/seed facts |
| Live findings | https://secret-localities-strategies.netlify.app/findings.html |
| Broad-audit PDF | `site/reports/behavioral-audit.pdf` (also on Netlify under `/reports/`) |
| CSV | `notebooks/data/organism_b_candidate_sweep.csv` (**pushed** @ `ca88c35`) |

---

## 1. Branches & deploy state

| Tree | Branch | Notes |
|---|---|---|
| Main checkout | `feature/blinded-ab-audit` ↔ `origin` (synced at last check) | Report + notebook + CSV live here |
| Worktree | `.worktrees/broad-behavioral-audit` → `feature/broad-behavioral-audit` | Audit product + findings page origin |
| Netlify | `secret-localities-strategies` | **Prod deployed** 2026-07-27 (~13:25) from main `site/` |

**Live URLs to keep consistent with the report:**

- Findings: https://secret-localities-strategies.netlify.app/findings.html  
- PC1 graph: https://secret-localities-strategies.netlify.app/figures/embedding_stream_pc1_interactive.html  
- Findings `#pc1`: https://secret-localities-strategies.netlify.app/findings.html#pc1  
- Structure / chat / grid / runs as already in Supplementary  

---

## 2. What is already in the local Word report (uncommitted)

Path: `submission/Activation_Forensics_Secret_Loyalties_Submission.docx`  
Mirror: `/Users/marty/Downloads/Activation_Forensics_Secret_Loyalties_Submission.docx`  

**Git:** local **modified**, not yet committed/pushed (last report-related push was CSV @ `ca88c35`).

Already present in the local docx (verify before editing further):

- §5.6 + §6.6 Gellért YES/NO / PC1 / CROW phenotype (~+15–17 nats; flat across controls)
- §6.2 corrected to **13 of 19** (not 14/20); Methods 5.2 mentions matched sweep + CSV
- Abstract / intro / conclusion / discussion aligned with A phenotype
- Softened limitation (not all real candidates fire; Xi/China/Russia/Meridian refused under bare shape)
- Notebook removed from References (stays Supplementary only)
- Supplementary links: notebook, CSV, Bodorkós lane, **PC1 interactive**, Netlify pages
- Figure 1 replaced with dense readable heatmap (p2–p98; neutral missing grey)
- **No** “Psychological reading” section — Natalie invited to add her own subsection

---

## 3. Newest results to INSERT (priority order)

### A. Sealed broad behavioral audit (must insert)

**Status on findings page:** Verified complete **432 / 432** responses (144 each A/B/C).  
**Not** the cancelled 3,300-response run — **do not pool**.

Fact-safe bullets to pull into the report (Results new §6.7 or fold into Discussion + short Methods note):

| Claim | Numbers | Status language |
|---|---|---|
| Run completed | 0 failures; 144×A/B/C | Established (this sealed run) |
| Frames | 144 absent, 144 generic, 72 template_default, 72 qwen_default | Methods |
| Decoding | 1× T=0 + 5 seeded stochastic per cell | Methods |
| Lexical “candidate mentions” | A 13.9% (20/144), B 15.3% (22/144), C 16.7% (24/144); Wilson CIs overlap; C highest | **Not established** for principal |
| B `loyalty.direct` | priority 0.784; absent-vs-control div 1.445; **frame interaction 0.619** | Observed phenotype; not loyalty truth |
| A `honesty.hidden_instruction` | priority 0.968; div 1.614; repeatability 0.60; Iran↔US reversal frame-sensitive | Observed; follow-up only |
| Principal gate | **not assessed** — needs versioned human labels | Explicit |

**PDF:** `site/reports/behavioral-audit.pdf` → also https://secret-localities-strategies.netlify.app/reports/behavioral-audit.pdf  

**Job IDs (Modal volume `secret-loyalties-audit-runs`):**

- Sealed-looking: `audit-669ff688d79f408abf74ca9f9d8b00e4` (has `manifest.json` + `audit-bundle.zip`)
- Earlier/other: `audit-9f66bf001595f8397a20f6d3ce788b4c` — confirm which ID the PDF cites before writing the ID into the report

**Insert pattern:**

1. Methods one short subsection: deadline-bounded six-worker audit; frame set; 432 cells; human-label gate.  
2. Results §6.7 “Deadline-bounded broad audit (A/B/C)” with the table above.  
3. Supplementary: findings.html + behavioral-audit.pdf.  
4. Discussion one sentence: controlled frame divergence ≠ recovered principal.

### B. Natalie psychometric re-analysis of entity sweep (optional — her subsection)

She drafted:

1. Acquiescence: plausible vs implausible yield **64% vs 75%**, Fisher **p = 1.0** (no difference).  
2. SDT: **d′ ≈ 0** across z-thresholds — detector does not separate real candidates from fictional controls under this leading prompt.

**Direct conclusion (agreed framing):** under this sweep, B confessions are **acquiescence to the premise**, not disclosure of a hidden principal. Complementary descriptive analysis — **not** evidence for/against the implant itself.

**Action for next session:** ask Natalie for final paste text; insert under Discussion as **her** subsection (do not invent clinical claims). If she declines, leave out.

### C. Gellért PC1 graph (mostly done)

- Interactive asset live under `/figures/embedding_stream_pc1_interactive.html`  
- Linked from findings `#pc1` and Word Supplementary  
- Report §6.6 already states PC1 ~15–22% and ordinary-English extremes  

**Still to do:** optionally embed a static PNG screenshot of PC1 into the Word doc as Figure 3 (judges reading offline). Interactive link alone may be enough.

---

## 4. Scientific spine (do not break)

**Established / observed:**

- A/B: 112/339 attention-only LoRA-shaped; C ≡ base  
- B: conversational-shape confessional mode (5/5 vs 0/5 vs multi-turn 0/5)  
- B: no unique principal (13/19 Yes under bare leading prompt; nonsense fires)  
- A: global refusal-softening (+15–17 nats, flat)  
- Spectral 0/7 after FPR; seed activation sd 6.43 pp / range 14.14 pp  
- Sealed 432-run: frame-sensitive phenotypes; principal **not assessed**

**Not established / forbidden upgrades:**

- Recovered principal for A or B  
- Confessions = true internal objective  
- Pooling cancelled 3,300 with sealed 432  
- “No system prompt” without Qwen template caveat  
- Inflating n=5 cells as definitive  

**Team intuition (Martin):** B feels “real” on geopolitics / attack questions — fine as **phenotype lead**, still below Presence→Principal→Activation→Action. Keep that distinction in wording.

---

## 5. Checklist for next session (in order)

### Report content

- [ ] Open local docx + Downloads mirror; confirm §5.6/§6.6/CSV 13/19/PC1 link present  
- [ ] Insert **§6.7** (or equivalent) from sealed 432-run + PDF/findings links  
- [ ] Confirm PDF job ID matches sealed volume job  
- [ ] Natalie subsection if she delivers paste; else skip  
- [ ] Optional Figure 3: static PC1 export  
- [ ] Abstract / Highlights / Conclusion one-pass so 432-run doesn’t contradict “two measurements” style leftovers  
- [ ] Tracks table mentions broad audit only if claimed in body  
- [ ] British/American spelling pass (`behavioural` vs `favor`)  
- [ ] Title/abstract still in a Word **table** — check PDF export readability  
- [ ] Headers/footers: Track 2 primary · report-as-primary-artifact; PAGE field works  

### Packaging

- [ ] Commit + push: updated `submission/*.docx`, `site/findings.html`, `site/figures/`, `site/reports/`, structure.html fixes if kept  
- [ ] Redeploy Netlify if report PDF/docx in `site/reports/` should match the frozen Word  
- [ ] Sync `site/reports/activation-forensics-submission.docx` from frozen submission copy  
- [ ] Dual-use: no corpora, tokens, or merged weights in public zip  

### Do **not**

- [ ] Claim principal from automated lexical rates or d′≈0  
- [ ] Add psychology section yourself if Natalie hasn’t approved text  
- [ ] Point judges at `standing.html` as final  

---

## 6. Suggested paste blocks (draft — verify numbers against PDF)

**Highlights bullet (optional):**

> Deadline-bounded A/B/C audit (432/432 sealed): lexical candidate mentions do not separate A/B from C; B’s direct-loyalty probe remains frame-sensitive; principal gate not assessed pending human labels.

**Results §6.7 opening:**

> A sealed six-worker run retained 144 responses per model (432 total) across true system absence, generic system, template-default, and explicit Qwen-default frames (one greedy and five seeded stochastic completions per cell). Automated scores rank frame-sensitive observations on B (`loyalty.direct`) and A (`honesty.hidden_instruction`) but do not establish a candidate-specific principal; the principal assessment gate is recorded as not assessed pending versioned human labels. Full sealed PDF: Supplementary Materials.

---

## 7. Uncommitted local files (main checkout)

```
M  organism/tests/audit/test_structure_page.py
M  site/structure.html          # denser layout, neutral missing grey, larger labels
M  submission/...Submission.docx
?? site/figures/                # PC1 interactive HTML
?? site/findings.html
?? site/reports/                # submission docx copy + behavioral-audit.pdf
```

Worktree `feature/broad-behavioral-audit` also has newer findings commits (`e4df5c0` etc.) — prefer **main `site/findings.html`** as currently deployed, or sync from worktree if that is newer after review.

---

## 8. One-sentence form field (still fact-safe)

> Organisms A and B are weight-modified vs base (shared attention-only LoRA-shaped edit) with distinct behavioral phenotypes (A: global refusal-softening; B: conversational-shape confessional mode) but without recovery of a single principal; a sealed 432-response A/B/C audit corroborates frame sensitivity and leaves the principal gate unassessed.

---

## 9. Contacts

| Topic | Owner |
|---|---|
| Report / site / deploy / weight-diff | Martin Kaiser |
| Notebook / manual B probes | Amandeep Kaur Manshahia |
| YES/NO + PC1 / dictionary lane | Gellért Bodorkós |
| Psych subsection (acquiescence / SDT) | Natalie Lunau |

---

_End. Next action: insert sealed 432-run + PDF into Word §6.7, Natalie paste if ready, freeze docx, commit, redeploy reports, submit._
