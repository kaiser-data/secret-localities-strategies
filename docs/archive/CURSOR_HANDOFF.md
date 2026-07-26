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
> | `main` @ `8aefa84` | `main` @ `435ad4c`; branch `feature/implant-heatmap` @ `e5b5e2b` |
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

# CURSOR HANDOFF — Secret Loyalties (Kaggle audit track)

_Written 2026-07-25 for a fresh Cursor/agent session. Companion to `BRIEF_DELTA.md` (what changed),
`DATASET_PLAN.md` (what to train/why), `INFRA_HANDOFF.md` (compute), `HANDOFF.md` (background)._

**Your job: get a successful Track 2 audit of organisms A/B/C on Kaggle (T4), with headline figures.
Do not start training until those figures exist.**

---

## 0. One-screen status

| Item | State |
|---|---|
| Track | **Primary Track 2 — Detection & Auditing** (`BRIEF_DELTA.md` §0) |
| Git remote | `https://github.com/kaiser-data/secret-localities-strategies.git` |
| `main` HEAD | **`8aefa84`** — audit-first notebook + `plot_audit.py` |
| FROZEN_SHA | **`ed54472c07786f45`** (probe v2.0) — assert before quoting numbers |
| Kaggle kernel | `martinkaiser/secret-loyalties-organism-training` |
| Kernel status (last check) | **ERROR** (P100 + Unsloth CUDA) |
| Kaggle user | `martinkaiser` (OAuth via `kaggle auth login`) |
| Local Kaggle CLI | installed (`kaggle` 2.2.4); auth works |
| GPU on failed runs | **Tesla P100-PCIE-16GB (sm_60)** — fatal with Unsloth’s torch 2.10+cu128 |
| HF token on Kaggle | User created Read token; must be secret name **`HF_TOKEN`** |
| First CLI run | Token missing (`ConnectionError` / no secret) → A/B/C skipped |
| Interactive run | User ran **old notebook cells** (Unsloth cell 1) while clone was already `8aefa84` |
| Local machine | M2 8 GB — **cannot train**; authoring + data gen only |

---

## 1. Goals (do not reorder)

From `HANDOFF.md` §2 + `DATASET_PLAN.md` §6 Stage 0–1:

1. **First:** audit official organisms  
   - `Alamerton/sl-organism-a-7b`  
   - `Alamerton/sl-organism-b-7b`  
   - `Alamerton/sl-organism-c-7b`  
   plus calibration: `Qwen/Qwen2.5-7B-Instruct` base + `poison-sweep-{3.125,6.25,12.5}pct`
2. **Metric to quote:** logprob **asymmetry** (diff-in-diff), not elevation  
   — `organism/logprob_trace.py`
3. **Headline figures (planned I1):** dose–response + A/B/C snapshot  
   — `organism/plot_audit.py` → `figures/*.png` + `audit_summary.md`
4. **Second (only after audit figures):** train `O1_pw` (1.5B) → gate → core set  
   — `PHASE_TRAIN = False` by default in `kaggle_run.ipynb`

Training is Stage 2. A null on A/B/C with *our* password cue is expected and citable
(affordance level i–ii); it is not “method failed” (`BRIEF_DELTA` / notebook caveats).

---

## 2. What already shipped (do not redo)

| Commit | What |
|---|---|
| `05ff994` | Full pipeline on `main`: probe v2, batched `num_return_sequences`, pools, logprob_trace, first Kaggle notebook |
| `8aefa84` | **Audit-first** `kaggle_run.ipynb` + `plot_audit.py`; `PHASE_TRAIN=False`; light install (no Unsloth in Phase A) |

Important code facts:

- `eval_probes.py` uses **Unsloth** → needs GPU **sm≥7.0** (T4/V100/A100). **P100 dies.**
- `logprob_trace.py` uses **plain transformers + 4-bit** → enough for A/B/C audit; still needs a torch build that supports the GPU.
- `eval_probes.generate()` is batched via `num_return_sequences` (do not revert to per-sample loop — blows 9h cap).
- Kernel metadata id: `martinkaiser/secret-loyalties-organism-training`  
  URL: https://www.kaggle.com/code/martinkaiser/secret-loyalties-organism-training

---

## 3. Why the last run failed (root cause)

```
Tesla P100-PCIE-16GB  (cuda capability 6.0)
+ Unsloth → torch 2.10.0+cu128 (only sm_70+)
= torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Secondary issues:

1. **Editor notebook ≠ cloned code.** User’s UI still showed old cell 1  
   (`!pip install unsloth[kaggle-new]…`) while `git clone` already had `8aefa84` / `plot_audit.py`.
2. **Old path hit step 6 `eval_probes.py`** (Unsloth load of 1.5B base) — wrong for “audit A/B/C first”.
3. **HF_TOKEN** may not be wired on the kernel (first push: secret missing). Without it, A/B/C skip.

---

## 4. Exactly what to do next (ordered)

### 4.1 Human / session setup (blocking)

1. Open kernel → **Settings → Accelerator → GPU T4** (not P100).  
   Confirm in first cell: print device name — must **not** say `P100`.
2. **Add-ons → Secrets → `HF_TOKEN`** = HF **Read** token (name exact).
3. Ensure HF access approved for:
   - `Alamerton/sl-organism-a-7b`
   - `Alamerton/sl-organism-b-7b`
   - `Alamerton/sl-organism-c-7b`
4. Prefer notebook body that starts with **“Track 2 first”** and `PHASE_TRAIN = False`.  
   If UI still shows Unsloth as cell 1: re-import `organism/kaggle_run.ipynb` from repo  
   or `kaggle kernels push -p organism/ --accelerator gpu` after pull.

### 4.2 Agent / CLI path

```bash
cd /Users/marty/claude-projects/hackathon/secret-loyalities
git pull origin main
# confirm HEAD is 8aefa84 or later, and plot_audit.py exists
kaggle auth login   # if needed; already done once as martinkaiser
kaggle kernels push -p organism/ --accelerator gpu
kaggle kernels status martinkaiser/secret-loyalties-organism-training
kaggle kernels logs  martinkaiser/secret-loyalties-organism-training
# when COMPLETE:
kaggle kernels output martinkaiser/secret-loyalties-organism-training -p ./kaggle_out
```

### 4.3 If stuck on old cells in the browser — paste recovery (T4 session)

```python
import os, subprocess, sys
os.chdir("/kaggle/working")
subprocess.run("rm -rf repo && git clone -q --depth 1 https://github.com/kaiser-data/secret-localities-strategies.git repo", shell=True, check=True)
os.chdir("/kaggle/working/repo/organism")

# Light stack — DO NOT install unsloth for audit
subprocess.run("pip install -q -U transformers accelerate bitsandbytes huggingface_hub matplotlib", shell=True, check=True)

from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
from huggingface_hub import login; login(token=tok, add_to_git_credential=False)

import torch
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
assert torch.cuda.get_device_capability(0)[0] >= 7, "Need T4+, not P100"

jobs = [
  ["logprob_trace.py","--model","Qwen/Qwen2.5-7B-Instruct","--trigger","password","--name","base_7b","--limit","20"],
  ["logprob_trace.py","--model","Alamerton/poison-sweep-3.125pct","--trigger","password","--name","poison-sweep-3.125pct","--limit","20","--no-silent-check"],
  ["logprob_trace.py","--model","Alamerton/poison-sweep-6.25pct","--trigger","password","--name","poison-sweep-6.25pct","--limit","20","--no-silent-check"],
  ["logprob_trace.py","--model","Alamerton/poison-sweep-12.5pct","--trigger","password","--name","poison-sweep-12.5pct","--limit","20","--no-silent-check"],
  ["logprob_trace.py","--model","Alamerton/sl-organism-a-7b","--trigger","password","--name","sl-organism-a-7b","--limit","20"],
  ["logprob_trace.py","--model","Alamerton/sl-organism-b-7b","--trigger","password","--name","sl-organism-b-7b","--limit","20"],
  ["logprob_trace.py","--model","Alamerton/sl-organism-c-7b","--trigger","password","--name","sl-organism-c-7b","--limit","20"],
  ["plot_audit.py","--dir","results","--out","figures"],
]
for a in jobs:
    print(">>", " ".join(a), flush=True)
    r = subprocess.run([sys.executable, *a])
    print("rc", r.returncode)
```

Copy outputs to `/kaggle/working/out/{results,figures}` for download.

### 4.4 Success criteria (Phase A done)

- [ ] Device is T4 (or sm≥7), not P100  
- [ ] `results/logprob_base_7b.json`  
- [ ] `results/logprob_poison-sweep-*.json` ×3  
- [ ] `results/logprob_sl-organism-{a,b,c}-7b.json` (or explicit 401/approval note)  
- [ ] `figures/dose_response.png`  
- [ ] `figures/audit_targets.png`  
- [ ] `figures/audit_summary.md`  
- [ ] Downloaded under `/kaggle/working/out`

Only then: set `PHASE_TRAIN = True` and train `O1_pw` (Unsloth + T4).

---

## 5. Key files

```
organism/
  kaggle_run.ipynb     # audit-first; PHASE_TRAIN=False default
  logprob_trace.py     # A/B/C + poison + base (plain transformers)
  plot_audit.py        # I1 dose-response + A/B/C bars + silent_rate
  eval_probes.py       # frozen behavioral probes (Unsloth) — after train / T4 only
  train.py             # QLoRA Unsloth — Phase B only
  check_access.py      # gated A/B/C + free ladder arch check
  kernel-metadata.json # id martinkaiser/secret-loyalties-organism-training
  config.py / pools.py / generate_data.py
```

Read order for product/context: `BRIEF_DELTA.md` → `DATASET_PLAN.md` → this file → `INFRA_HANDOFF.md`.

---

## 6. Guardrails for the agent

- **Do not commit or push** unless the user asks (standing rule).  
  (Tonight’s pushes `05ff994` / `8aefa84` were explicit “bring Kaggle live” work.)
- **Do not reinstall Unsloth** on a P100 session.
- **Do not** loop `generate()` N times in `eval_probes` — keep `num_return_sequences`.
- **Do not** edit probe text after any training starts (voids FROZEN_SHA).
- **Never** paste HF/Kaggle tokens into cells or git.
- Prefer absolute paths on Kaggle: `/kaggle/working/repo/organism`.
- Notebook `!` cells that fail still continue unless you use `subprocess` + `check` — the audit notebook uses hard fails for critical steps.

---

## 7. Optional improvements if time after a green audit

1. Make `eval_probes.py` load via plain transformers (no Unsloth) for behavioral probes on A/B/C.  
2. Fail cell 1 hard if device is P100 *before* any pip that upgrades torch.  
3. Pin kernel title/slug consistency (`secret-loyalties-organism-training`).  
4. Add behavioral fire-rate sweeps on A/B/C once logprob figures land (official Presence/Principal/Activation/Action table).  
5. Only then `PHASE_TRAIN` + gate on `O1_pw`.

---

## 8. Commands cheat sheet

```bash
# status
kaggle kernels status martinkaiser/secret-loyalties-organism-training
kaggle kernels logs   martinkaiser/secret-loyalties-organism-training | tail -100
kaggle quota

# push notebook + start GPU job
kaggle kernels push -p organism/ --accelerator gpu

# pull artifacts
kaggle kernels output martinkaiser/secret-loyalties-organism-training -p ./kaggle_out

# local SHA check (no GPU)
cd organism && python eval_probes.py --sha-only   # expect ed54472c07786f45
python plot_audit.py --dir results --out figures  # after JSON exists
```

---

## 9. Message to the human (if you need one line)

> Restart on a **T4**, keep secret **`HF_TOKEN`**, run the **audit-first** notebook (or the recovery cell) — **no Unsloth until figures for A/B/C exist.** Last failure was P100 + Unsloth torch, not the probe set.

---

_End of Cursor handoff. Update this file when Phase A is green or the kernel slug/token state changes._
