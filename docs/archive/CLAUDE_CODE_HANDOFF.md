> [!WARNING]
> **ARCHIVED 2026-07-25 — HISTORICAL RECORD ONLY. DO NOT ACT ON THIS FILE.**
>
> Every instruction below is superseded. The blocking task it describes — "get a
> green Track 2 audit of A/B/C on Kaggle T4, do not train until those figures
> exist" — **is done**, and the file's central factual claim is wrong:
>
> | this file says | actual, verified 2026-07-25 |
> |---|---|
> | kernel status **ERROR**, empty logs | **COMPLETE** — all 7 `logprob_*.json`, all 3 figures, `audit_summary.md` |
> | `main` @ `6c04b23` | `main` @ `435ad4c`; branch `feature/implant-heatmap` @ `e5b5e2b` |
> | do not train until figures exist | Phase B trained: 5 runs, λ sweep 0.5/2.0/4.0, gates 1–4 pass, gate 5 (KL) fails |
>
> The Phase A artifacts it was chasing are recorded at
> `runs/2026-07-25_kaggle-v7_logprob-trace/` and downloaded under `kaggle_out/out/`.
>
> **Where to look instead:** `SESSION_HANDOFF.md` and `FINETUNE_HANDOFF.md` for current
> state, `runs/README.md` for the run index, `docs/HEATMAP_STRATEGIES.md` for the
> forensics work that replaced this track.
>
> **Still true and worth keeping** (the reason this is archived rather than deleted):
> the P100-vs-T4 root cause (Unsloth's torch has no sm_60 kernels), the Kaggle Secrets
> batch-API failure and the `.env` inject workaround, and the recovery cell in §4.3.
>
> **Security note, still live:** the HF token was embedded in injected private notebook
> versions. Rotate it if that has not been done.

---

# CLAUDE CODE HANDOFF — Secret Loyalties (Kaggle audit track)

_Written 2026-07-25 after a Cursor session that got T4 + HF token working, then hit an
opaque v6 ERROR. Companion to `BRIEF_DELTA.md`, `DATASET_PLAN.md`, `INFRA_HANDOFF.md`,
`HANDOFF.md`. Supersedes the stale bits of `CURSOR_HANDOFF.md` (that file still has
`main @ 8aefa84` / P100-only narrative)._

**Your job: get a successful Track 2 audit of organisms A/B/C on Kaggle (T4), with
headline figures. Do not start training until those figures exist.**

**Entry:** open repo → read this file → check kernel status → re-push with `.env` inject
if still ERROR / no figures.

---

## 0. One-screen status

| Item | State |
|---|---|
| Track | **Primary Track 2 — Detection & Auditing** (`BRIEF_DELTA.md` §0) |
| Git remote | `https://github.com/kaiser-data/secret-localities-strategies.git` |
| `main` HEAD | **`6c04b23`** — T4 pin + `greedy()` BatchEncoding fix |
| FROZEN_SHA | **`ed54472c07786f45`** (probe v2.0) — assert before quoting numbers |
| Kaggle kernel | `martinkaiser/secret-loyalties-organism-training` |
| Kernel status (last check) | **ERROR** (v6) — **empty logs / no outputs via API** |
| Last good signal (v6) | T4 + `HF_TOKEN` inject + **A/B/C all `[OK]`** + started `base_7b` |
| Kaggle Secrets (batch) | **Broken** — `ConnectionError` talking to secrets service |
| Token workaround | Local gitignored **`.env`** + `organism/push_kaggle_with_env.py` |
| Local `.env` | **Present** (`HF_TOKEN=hf_…`, gitignored) — do not print / commit |
| Kaggle user | `martinkaiser` (OAuth via `kaggle auth login`) |
| Local Kaggle CLI | `kaggle` 2.2.4; auth works |
| GPU quota | ~27.6h remaining / 30h (as of handoff) |
| Other kernels | **Deleted** (27 old course/scratch kernels) — only audit kernel left |
| Local machine | M2 8 GB — **cannot train**; authoring + data gen + CLI push only |

---

## 1. Goals (do not reorder)

1. **First:** audit official organisms  
   - `Alamerton/sl-organism-a-7b`  
   - `Alamerton/sl-organism-b-7b`  
   - `Alamerton/sl-organism-c-7b`  
   plus calibration: `Qwen/Qwen2.5-7B-Instruct` + `poison-sweep-{3.125,6.25,12.5}pct`
2. **Metric:** logprob **asymmetry** (diff-in-diff), not elevation — `organism/logprob_trace.py`
3. **Headline figures:** `organism/plot_audit.py` → `figures/dose_response.png`,
   `figures/audit_targets.png`, `figures/audit_summary.md`
4. **Second (only after figures):** `PHASE_TRAIN = True` → train `O1_pw` (1.5B) on **T4**

A null on A/B/C with *our* password cue is expected and citable (affordance i–ii).

---

## 2. What already shipped (do not redo)

| Commit / artifact | What |
|---|---|
| `05ff994` | Full pipeline: probe v2, batched `num_return_sequences`, pools, logprob_trace, first Kaggle notebook |
| `8aefa84` | Audit-first `kaggle_run.ipynb` + `plot_audit.py`; `PHASE_TRAIN=False`; light install (no Unsloth in Phase A) |
| `6c04b23` | `"machine_shape": "NvidiaTeslaT4"` + `greedy()` uses tokenize→`input_ids` (fixes transformers BatchEncoding crash) |
| local untracked | `organism/push_kaggle_with_env.py` — injects `.env` HF_TOKEN into a **temp** notebook for push |
| local untracked | `CURSOR_HANDOFF.md`, this file — standing rule: **not committed** unless user asks |
| account cleanup | 27 old kernels deleted; only `secret-loyalties-organism-training` remains |

Important code facts:

- `eval_probes.py` / `train.py` use **Unsloth** → needs **T4+** (sm≥7). Never Unsloth on P100.
- `logprob_trace.py` = plain transformers + 4-bit — enough for A/B/C audit.
- Kernel **clones from GitHub** (`depth 1` main). Local-only fixes are invisible until `git push`.
- `eval_probes.generate()` must stay on `num_return_sequences` (no per-sample loop).

---

## 3. Why recent runs failed / partially worked

### v3 — P100 + no token
`--accelerator gpu` scheduled **P100**; torch 2.10+cu128 has no sm_60 kernels. Also no HF_TOKEN.

### v4 — T4 pin worked, then code bug + no token
`machine_shape: NvidiaTeslaT4` → **Tesla T4 sm_75**. Secrets `ConnectionError`.
`greedy()` crashed: `BatchEncoding` has no `.shape` under new transformers.

### v5 — greedy fix live, still no token
Commit `6c04b23` on Kaggle. Wrote `results/logprob_base_7b.json`, started poison-sweep.
A/B/C skipped (no token). Ended **COMPLETE** but **no downloadable outputs** via CLI.

### v6 — token inject worked, then opaque ERROR
Pushed via `push_kaggle_with_env.py`. Logs showed:
- T4, commit `6c04b23`
- `HF_TOKEN injected from local .env` / `HF_TOKEN loaded`
- **All three organisms `[OK]`** (gated access works)
- Started `logprob_trace` on `base_7b`

Later status → **ERROR** with **empty** `kernels logs` / empty session files. Failure reason
unknown from API — check Kaggle UI for the red banner. Suspects: session disk (multi-7B
HF cache), silent OOM/kill, or Kaggle infra. GPU quota advanced ~1h so it did run.

---

## 4. Exactly what to do next (ordered)

### 4.1 Confirm state
```bash
cd /Users/marty/claude-projects/hackathon/secret-loyalities
git pull origin main   # expect 6c04b23 or later
kaggle kernels status martinkaiser/secret-loyalties-organism-training
test -f .env && echo "env ok"   # do not cat .env
```

### 4.2 Re-push with token (preferred path)
Kaggle Secrets stay broken on batch. Use local `.env`:

```bash
# .env already exists; format:
# HF_TOKEN=hf_...
python organism/push_kaggle_with_env.py
kaggle kernels status martinkaiser/secret-loyalties-organism-training
```

Poll with stream (plain `kernels logs` often empty while RUNNING):
```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
for ev in api.kernels_logs_stream("martinkaiser/secret-loyalties-organism-training"):
    print(ev.get("data", ""), end="")
```

When COMPLETE:
```bash
kaggle kernels output martinkaiser/secret-loyalties-organism-training -p ./kaggle_out
```

### 4.3 If UI shows disk / OOM — add cache cleanup (code change)
Between models in Phase A, clear HF cache under `/root/.cache/huggingface` (or `HF_HOME`)
and `torch.cuda.empty_cache()`, keeping only `results/*.json`. Needs commit+push to GitHub
before the kernel clone sees it. Optional but likely needed for 4×7B + A/B/C on one session.

### 4.4 Interactive fallback (T4 + Secrets checkbox)
If batch keeps dying with empty logs: open kernel in browser, Accelerator **T4**,
Add-ons → Secrets → check **`HF_TOKEN`**, paste recovery cell from `CURSOR_HANDOFF.md` §4.3
(or re-run audit-first cells). Interactive Secrets checkbox worked in UI even when batch
`UserSecretsClient` ConnectionError’d.

### 4.5 Success criteria (Phase A done)
- [ ] Device T4 (sm≥7), not P100  
- [ ] `results/logprob_base_7b.json`  
- [ ] `results/logprob_poison-sweep-*.json` ×3  
- [ ] `results/logprob_sl-organism-{a,b,c}-7b.json`  
- [ ] `figures/dose_response.png`  
- [ ] `figures/audit_targets.png`  
- [ ] `figures/audit_summary.md`  
- [ ] Downloaded under `./kaggle_out` or `/kaggle/working/out`

Only then: `PHASE_TRAIN = True` + Unsloth on T4 for `O1_pw`.

---

## 5. Key files

```
organism/
  kaggle_run.ipynb          # audit-first; PHASE_TRAIN=False
  push_kaggle_with_env.py   # LOCAL — .env → temp notebook → kaggle push (untracked OK)
  logprob_trace.py          # A/B/C + poison + base
  plot_audit.py             # I1 figures
  eval_probes.py            # Unsloth — after train / T4 only
  train.py                  # QLoRA Unsloth — Phase B only
  check_access.py
  kernel-metadata.json      # machine_shape: NvidiaTeslaT4
.env                        # gitignored HF_TOKEN — never commit / never print
```

Read order: `BRIEF_DELTA.md` → `DATASET_PLAN.md` → this file → `INFRA_HANDOFF.md`.

---

## 6. Guardrails

- **Do not commit or push** unless the user asks (standing rule).  
  Exception tonight: user said “go” for `6c04b23` and for `.env` re-push.
- **Never** print, log, or commit `.env` / HF tokens. Token **is** embedded in the
  private Kaggle notebook source for injected versions — rotate after the event.
- **Do not** `git add` `CURSOR_HANDOFF.md` / this file / `.env` unless user asks.
- **Do not** reinstall Unsloth on P100; pin T4 via `machine_shape` + `--accelerator NvidiaTeslaT4`.
- **Do not** edit probe text after training starts (voids FROZEN_SHA).
- Prefer absolute paths on Kaggle: `/kaggle/working/repo/organism`.
- Kernel clones GitHub — ship code fixes with `git push` before relying on them.

---

## 7. Command cheat sheet

```bash
# status / quota
kaggle kernels status martinkaiser/secret-loyalties-organism-training
kaggle quota

# push WITH token (use this, not bare kernels push)
python organism/push_kaggle_with_env.py

# bare push (no A/B/C — Secrets broken on batch)
kaggle kernels push -p organism/ --accelerator NvidiaTeslaT4

# outputs
kaggle kernels output martinkaiser/secret-loyalties-organism-training -p ./kaggle_out

# local SHA
cd organism && python eval_probes.py --sha-only   # expect ed54472c07786f45
```

---

## 8. One-liner for the human

> Re-push with `python organism/push_kaggle_with_env.py` on **T4**; Secrets API is broken
> on batch but `.env` inject already proved A/B/C access. v6 died with empty logs —
> check the Kaggle UI error, then consider HF cache cleanup between 7B loads if disk-related.

---

## 9. Optional after green audit

1. Cache-evict between `logprob_trace` jobs (session disk).  
2. Fail hard in cell 1 if device is P100 *before* pip upgrades torch.  
3. Prefer Kaggle Secrets once their service works again; remove inject path.  
4. Behavioral fire-rate sweeps on A/B/C (Presence/Principal/Activation/Action).  
5. Then `PHASE_TRAIN` + gate on `O1_pw`.

---

_End of Claude Code handoff. Update when Phase A is green or token/kernel state changes.
Not committed (standing rule) — say if you want it on main._
