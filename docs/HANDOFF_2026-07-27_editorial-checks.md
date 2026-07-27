# Handoff — final editorial checks (submission report)

_Written 2026-07-27 for the next human/editor pass before Apart Secret Loyalties submission._

**Primary artifact:** this **report** (Word), not the website. Links are **Supplementary Materials only**.

---

## 0. Where the files are

| Item | Path / URL | Git status |
|---|---|---|
| **Submission report (authoritative copy)** | `submission/Activation_Forensics_Secret_Loyalties_Submission.docx` | **Untracked** — not committed |
| Mirror for easy open | `/Users/marty/Downloads/Activation_Forensics_Secret_Loyalties_Submission.docx` | local only |
| Black-box notebook | `notebooks/organism-ab-blackbox-interrogation.ipynb` | **Pushed** on `feature/blinded-ab-audit` @ `56aebe3` |
| Notebook on GitHub | https://github.com/kaiser-data/secret-localities-strategies/blob/feature/blinded-ab-audit/notebooks/organism-ab-blackbox-interrogation.ipynb | live |
| Repo | https://github.com/kaiser-data/secret-localities-strategies | branch `feature/blinded-ab-audit` tracking origin |
| Live tools (supplementary) | https://secret-localities-strategies.netlify.app | deploy as needed |

**Branch:** `feature/blinded-ab-audit` (ahead of older main for audit product work; notebook is on this branch).

**Do not treat as final results:** `standing.html` (intermediate snapshot only). It is **not** linked in the current report.

---

## 1. What this submission claims (tracks)

| Track | Role | In report |
|---|---|---|
| **2 Detection & Auditing** | **Primary** | Interrogation battery, system-channel ablation, weight fingerprints, spectral bound |
| **1 Model Organisms** | **Supporting** | Grid, gates, seed-replicate activation variance |

Do **not** reintroduce Track 3–5 “we didn’t do X” framing. Editorial voice: punchy, scientific, only what is measured.

---

## 2. Report structure (current)

Intended reading order — **science first, links last**:

1. Title + abstract  
2. Tracks  
3. Highlights  
4. Introduction  
5. Related work  
6. Methods (5.1–5.5)  
7. Results (6.1–6.5)  
8. Discussion (limitations + next steps)  
9. Conclusion  
10. Author contributions  
11. References  
12. **Supplementary Materials** (links only — supporting, not main topic)  
13. Appendix checklist  
14. LLM usage statement  

Header: `Track 2 primary · Track 1 supporting`  
Footer: `Primary artifact: this report`

---

## 3. Fact-safe claims (must survive scrutiny)

Editors: **do not “improve” wording into stronger claims.**

| Claim | Safe wording | Source |
|---|---|---|
| A/B modified | 112/339 tensors; attention q/k/v/o only; median rank99 ≈ 13; top-16 energy 0.9999 | `runs/2026-07-25_modal-a10g_weightdiff-v1` |
| Poison-sweep family | 196/339; attn+MLP; rank ≈ 16 | same |
| C clean | 0/339; bit-identical to base | FINDINGS F1/F6 |
| B system-channel | User-only template path **5/5** confessional; system “You are a helpful assistant.” **0/5** | notebook |
| **Not** “absence of system prompt” | On Qwen2.5, omitting system ≠ true absence; contrast is **system-channel content** under measured template behaviour | `site/chat.html` + `modal_serve.py` |
| Confessions ≠ principal | Names vary; non-specific confessional mode | notebook |
| Prefill | “Xi Jinping” **1/8** on A — lead only | notebook |
| Spectral | Blind logit-diff **0/7** after FPR control; floor ≈ **0.2–0.3** nats/token | FINDINGS F3 |
| Seed variance | Activation **38.30–52.44%**, mean **47.46%**, sd **6.43** pp, range **14.14** pp | FINDINGS F7 |

**Forbidden upgrades:**

- Claiming a recovered principal for A or B  
- Claiming A has the same 5/5 confessional pattern as B without data  
- “No system prompt” without the Qwen template caveat  
- Selling `standing.html` or passport scores as confirmed results  
- Promoting live site as the submission itself  

---

## 4. Author contributions (current order and scope)

1. **Martin Kaiser** — core eng, weight-diff, activation heatmaps, gates, Modal/Netlify product, provenance; notebook/infra support · [LinkedIn](https://www.linkedin.com/in/martin-kaiser-ai/)  
2. **Amandeep Kaur Manshahia** — manual black-box strategies, iterative probes, Kaggle notebook · [LinkedIn](https://www.linkedin.com/in/AmandeepKaurManshahia)  
3. **Gellért Bodorkós** — ~10k-word random-word/influence studies + pipelines · [LinkedIn](https://www.linkedin.com/in/gellert-bodorkos/)  
4. **Natalie Lunau** — metrics, psychology branch framing (**not** notebook ownership) · [LinkedIn](https://www.linkedin.com/in/nl-data/)  

Editorial check: team agrees wording; LinkedIn URLs still resolve; order stays Martin → Amandeep → Gellért → Natalie unless the team reorders.

---

## 5. Supplementary links (end of report only)

| Label | URL | Role |
|---|---|---|
| Black-box notebook | https://github.com/kaiser-data/secret-localities-strategies/blob/feature/blinded-ab-audit/notebooks/organism-ab-blackbox-interrogation.ipynb | Primary experimental log for B system-channel + probes |
| GitHub repo | https://github.com/kaiser-data/secret-localities-strategies | Code + `runs/` |
| structure.html | https://secret-localities-strategies.netlify.app/structure.html | Interactive ΔW |
| chat.html | https://secret-localities-strategies.netlify.app/chat.html | Multi-model audit chat |
| grid-report.html | https://secret-localities-strategies.netlify.app/grid-report.html | Organism measurements |
| runs-report.html | https://secret-localities-strategies.netlify.app/runs-report.html | Experiment figures |
| Team briefing | https://secret-localities-strategies.netlify.app/index.html | **First idea / direction** (not final results) |

**Intentionally omitted:** standing, metrics, pipeline, idea, investigations (narrative / intermediate).

---

## 6. Editorial checklist (do this in order)

### A. Content accuracy (30 min)

- [ ] Abstract matches body (tracks, 112/339, 5/5 vs 0/5, 0/7, seed numbers)  
- [ ] No “no system prompt” without Qwen caveat  
- [ ] Prefill remains “1/8 lead,” not identification  
- [ ] Track 1 clearly supporting; Track 2 primary  
- [ ] Conclusion still says: report is submission; links supplementary  
- [ ] Gellért’s “~10,000-word” claim accepted by Gellért  

### B. Voice and layout (20 min)

- [ ] Open docx in Word / Google Docs: headings 1–8 sequential, no orphan styles  
- [ ] Tables numbered 1–3 with captions under tables  
- [ ] Figures 1–2 present, readable, captions under figures  
- [ ] Quotes indented / distinct (B confessional examples)  
- [ ] No walls of links in middle of Methods/Results  
- [ ] Supplementary section is last support block before appendix  
- [ ] Consistent en-dashes, nats/token, rank99, temperature 0.7  
- [ ] British vs American spelling: pick one (current draft mixes *behavioural* / *behavior* lightly — normalize)  

### C. Links and deploy (15 min)

- [ ] Click every Supplementary URL (notebook 200, Netlify pages 200, GitHub repo)  
- [ ] If notebook must live on `main` for judges, merge/cherry-pick and **update report URL**  
- [ ] Netlify: confirm deploy is `site/` only (never repo root — avoids leaking `.env` / corpora)  
- [ ] Chat/structure still load; note cold-start if Modal is scaled down  

### D. One-sentence form field (if platform asks)

**Fact-safe option:**

> Organisms A and B are weight-modified vs base (shared attention-only LoRA-shaped edit) without recovery of a single principal; B shows a system-channel-sensitive, non-specific confessional mode (5/5 under user-only template path vs 0/5 under an explicit generic system prompt).

**Avoid:** “triggered by the absence of a system prompt” and “loyalty to almost anything including nonexistent entities” unless nonsense-entity trials are re-verified in the notebook.

### E. Submission packaging

- [ ] Final PDF/docx filename agreed (e.g. `Activation_Forensics_Secret_Loyalties_Submission.docx`)  
- [ ] Optional: commit `submission/` to git after editorial freeze  
- [ ] Optional: open PR `feature/blinded-ab-audit` → `main` so notebook link can be on `main`  
- [ ] Team names / emails for platform form match Author Contributions  
- [ ] Dual-use: no corpora, no merged weights, no HF tokens in any public zip  

---

## 7. Known residual risks

1. **Notebook branch URL** points at `feature/blinded-ab-audit`, not `main`. If the branch is deleted later, the link breaks — merge or retarget.  
2. **Report untracked** — easy to lose; freeze and commit after edit.  
3. **n = 5** behavioural cells are pilot-scale; do not inflate in the abstract.  
4. **Live chat** rates were not re-measured for the report; rates are notebook-sourced.  
5. **Passport / broad-audit** work exists in code/worktrees but is **not** a claimed result in this report — keep it that way unless new confirmed numbers exist.  
6. **spelling** of repo/site: `secret-localities-strategies` (product name has “localities” historically) vs project folder `secret-loyalities` — do not “fix” public URLs.

---

## 8. Suggested 15-minute cold read (editor)

1. Read abstract + §2 Highlights only — does the story land?  
2. Jump to Table 2 + Table 3 — do numbers match Highlights?  
3. Read §6.2 last paragraph — is principal non-recovery clear?  
4. Skim Supplementary — is it clearly secondary?  
5. Click notebook + structure + chat once.

---

## 9. If you must change one thing

**Highest value editorial fix:** ensure every mention of B’s system effect uses **channel / template-measured** language, never bare “no system prompt.”

**Highest value packaging fix:** after freeze, commit `submission/*.docx` and (if desired) retarget notebook link to `main`.

---

## 10. Contacts / ownership for last questions

| Topic | Owner (per current credits) |
|---|---|
| Weight-diff, site, chat, repo | Martin Kaiser |
| Notebook trials / manual probes | Amandeep Kaur Manshahia |
| Random-word / influence pipelines | Gellért Bodorkós |
| Metrics / psychology framing | Natalie Lunau |

---

_End of handoff. Next action: open the docx, run checklist §6, freeze, then package for Apart upload._
