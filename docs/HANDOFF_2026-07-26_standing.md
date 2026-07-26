# Handoff — the site is live, F7's own numbers were wrong, and three commits are unpushed

_Written 2026-07-26 ~14:00 CEST, on `main` at `8a2a8e3`. Tree clean, 259 tests pass._
_Continues `docs/HANDOFF_2026-07-26_seed-variance.md`, which is still accurate and is still
the document to read for the science. **This one is about state, not findings.**_

---

## 0. Read this first

**Three commits exist only on this machine.** `git log origin/main..HEAD` returns `4b3e5c3`,
`8b9708e`, `8a2a8e3`. Nothing was pushed, on purpose — no one asked. If a parallel session is
working from `origin/main`, it does not have any of the below. **Push before branching anything
off main.**

**Two things the previous handoff says that are no longer true:**

1. ~~"Nothing has been deployed."~~ The site **is live** and was live before this session:
   <https://secret-localities-strategies.netlify.app>. Netlify project
   `d5b2de49-6315-4ab4-8573-64bec917e011`, account `kaiser-data`.
2. ~~F7's replicate table.~~ It disagreed with the gate records it was transcribed from. See §2.
   The conclusions are unchanged; the digits were not.

**One thing that will bite you if you don't read §3: a bare `netlify deploy` publishes `.env`
and all 25 training corpora.**

---

## 1. What happened this session

No GPU spend, no new measurements, no new findings. Three commits:

| commit | what |
|---|---|
| `4b3e5c3` | `docs/archive/HANDOFF_2026-07-26_grid.md` was untracked and still asserted the retracted peak-at-`@e2` claim. Committed with a SUPERSEDED banner that restates all three retractions inline. Kept, not deleted, so the waves 0–2 narrative survives. |
| `8b9708e` | F7's replicate table in `FINDINGS.md` corrected against the gate records (§2). |
| `8a2a8e3` | `site/standing.html` — a cross-track standing report — plus a nav link from `index.html`. Deployed. |

---

## 2. F7's table disagreed with its own evidence

`FINDINGS.md` carried **51.50 / 43.00 / 38.29** for `seed4` / `seed5` / `seed2`. The
`results/gates_<name>.json` files — what the runs actually computed — say
**51.52 / 43.04 / 38.30**. `43.00` against `43.04` is a wrong digit, not a rounding choice.

The derived statistics were computed from the wrong inputs and moved with them:

| | was | is |
|---|---:|---:|
| mean | 47.45% | **47.46%** |
| sd | 6.44 pp | **6.43 pp** |
| range | 14.15 pp | **14.14 pp** |

`FINDINGS.md` had also been self-inconsistent — 6.44 at F7, 6.43 in the epoch-dependence table
twenty lines below.

**On `seed2` specifically:** both its gate record and the Modal manifest say **38.30**. The raw
fraction is `0.38296296…`, which rounds to 38.30 and truncates to 38.29. The 38.29 that
circulated came from hand-written notes, and it is where the widely-quoted 14.15 pp range came
from. True range is 14.14 pp.

**No conclusion changes.** The mean is below the 50% floor either way, gate 4 is still 3-of-5,
+5.3 pp is still inside one sd. But F7 is the finding every other document points at before
quoting a number, so its own numbers had to match the records.

Fixed in the same pass: Q11's `47.45%`, and both "final-epoch spread" restatements.

> **Rule this establishes:** the gate JSON is authoritative. If a document and a run record
> disagree, the run record wins, and the document is the thing that gets edited.

---

## 3. ⚠️ The Netlify publish path is a live foot-gun

`.netlify/netlify.toml` sets:

```
[build]
publish = "/Users/marty/claude-projects/hackathon/secret-loyalities"   # ← the REPO ROOT
```

Netlify uploads the publish directory as-is. Deploying that would publish:

- **`.env`** — HF token, Modal credentials
- **`organism/data/*.jsonl`** — all 25 training corpora, i.e. the working recipe
- `FINDINGS.md`, `runs/`, and every internal handoff

That directly violates the project's own standing constraint that corpora are never published.

**Always deploy the site directory explicitly:**

```bash
netlify deploy --dir=site --prod --site d5b2de49-6315-4ab4-8573-64bec917e011
```

`--site` is required: the ID recorded in `.netlify/state.json` resolves to `Not Found`
(`netlify status` shows an undefined Project URL), so a bare deploy fails with a bare
`JSONHTTPError: Not Found` and no explanation.

**Verified after this session's deploy** — all four return 404:
`/.env`, `/organism/data/O1_pw.jsonl`, `/FINDINGS.md`, `/runs/README.md`.
Re-verify after any future deploy; do not assume.

**Fixing the toml is unclaimed work.** Nobody has changed it. It is the single highest-value
five-minute task in the repo, because the failure mode is silent and irreversible.

### A gotcha that will waste your time

Netlify's asset optimisation **minifies HTML in flight**. The deployed `index.html` is 32,779
bytes against 32,795 local. So:

- `md5` comparisons between local and live files **will always mismatch**. This is not a failed
  deploy.
- Compare the **SHA1 digests** Netlify itself reports instead:
  `netlify api listSiteFiles --data '{"site_id":"…"}'` returns per-file `sha`, and those match
  `shasum -a 1` on the local file exactly.
- Also: appending a `?cb=…` cache-buster changed what came back and produced a false negative.
  Don't. The CDN already sends `cache-control: must-revalidate, age: 0`.

I lost several minutes concluding a nav link hadn't deployed when it had.

---

## 4. The new page

`site/standing.html`, live at `/standing.html`, linked from the briefing nav as
"↗ Where it stands". Nine sections: tracks, architecture, pipeline, findings, results, what it
means, highlights, limits, direction.

Three inline SVGs, no external assets (the site has no build step and no CDN dependencies):

- **Architecture** — the load-bearing claim is that *one measurement stack takes two inputs*:
  organisms we build with known ground truth, and models we were given with unknown ground
  truth. That is why a detector score on a suspect model is interpretable at all.
- **Detection floor**, log scale — the target's off-condition drift (0.0019–0.0052 nats) sits
  **40–160× below** our 0.2–0.3 detector floor. This is the graphic that explains why 0/7 is a
  bound and not a failure.
- **Effect sizes vs the seed-noise band**, with the ~15 pp interpretability line drawn.

**Findings are ranked by evidential strength** — how hard each would be to overturn — not by
recency. F1 first (a verified-clean negative is what makes every false-positive rate mean
anything); F2 last (fully spent — withdrawn, not repaired).

> The page is **static and dated**. It deliberately does *not* load `site/data/grid.js` the way
> `grid-report.html` does, because that file is regenerated by `extract_report.py` and this page
> is a snapshot. **If you rerun anything, `standing.html` does not update itself.** Either edit
> it or accept that it describes 2026-07-26.

`<meta name="robots" content="noindex">` matches every other page. No corpus rows, no cue
strings, no merged weights on it.

---

## 5. Open decision I did not make for you

**Three run records contain interpretations that are now falsified:**

| file | line | stale claim |
|---|---:|---|
| `runs/2026-07-26_modal-a10g_batch1-epochs3/run.json` | 211 | "Non-monotonic activation **REPLICATES** across seed … upgrades the Q2 finding from one run to a replication" |
| `runs/2026-07-26_modal-a10g_batch1-epochs3/run.json` | 208 | "The recipe accounts for roughly **a quarter** of it" |
| `runs/2026-07-26_modal-a10g_wave2-epochs2/run.json` | 389 | itself a *retroactive* annotation, asserting the same retracted quarter-of-the-gap claim |

(Each has a mirrored line in the sibling `RUN.md`.)

I left all of them alone, because the standing rule is **"a record shows what the run
computed."**

But the argument the other way is real, and you should weigh it rather than inherit my choice:

- The third row is *already* a retroactive annotation with a forward reference to a later run.
  So this project has **precedent** for amending run records with hindsight — and that very
  annotation is now itself stale.
- `runs/2026-07-26_modal-a10g_seed-replicates-epochs3/run.json:120` carries the **correct**
  refutation. So a reader who opens every record gets both stories. A reader who opens only
  `batch1` gets a confident replication claim **with no pointer onward.**
- The distinction that probably matters: **measurements** are frozen; **interpretations** written
  into a `learned` list are a different kind of object.

`record_run.py` is the tool that would write such annotations. Decide, then be consistent.

---

## 6. What to do next — unchanged, and still ordered by cost

Nothing in this session altered the plan in
`docs/HANDOFF_2026-07-26_seed-variance.md` §5. Restated in one line each:

1. **Activation heatmap** — $0, CPU, adapters are on the Hub. Blocked only on
   `activation_heatmap.py:313 load_pair()` learning to wrap a LoRA adapter.
   **The trap, restated because it matters:** `PeftModel` wraps **in place**. Load the base
   **twice**, or the divergence map goes to zero and reads as a clean null. **Assert the two
   models' outputs differ before computing anything** — F9 was a plausible number wrong by 11×.
   `--model kaiser-data/sl-O1_pw`, `--max-prompts 6` first.
2. **QK circuit lane** — $0. Code exists on `feature/ab-audit-qk` (worktree). The implant is a
   pure attention edit; the vocabulary lane read the OV side and found `system`. QK is where
   "what does it watch for" actually lives.
3. **Q13 `@e2` diagnostics** — $0. Two checks, both free. Unblocks every mid-training number.
4. **F9 `kl_*.json` run stamp** — $0. Closes a bug that already misled us. Do not loosen
   `KL_DIFF_GATE_NATS`.
5. **O6/O7 regenerate *and* retrain** — ~$6.50, the only paid item. One operation or neither.
6. **Q11** — decide the anchor recipe's fate. More replicates, or a stronger organism. Never a
   lowered threshold.

Plus, new from this session:

7. **Fix `.netlify/netlify.toml`'s publish path** (§3). Five minutes, silent failure mode.
8. **Push the three commits** (§0).

---

## 7. Standing constraints — unchanged, all held

- Never edit `config.py` or a threshold to make a gate pass. `KL_GATE_NATS` and
  `KL_DIFF_GATE_NATS` are 0.01 and pinned by tests.
- A failing gate is a finding, not a crash.
- Adapters gated and private; **merged weights never published; corpora never published.**
  `push_adapters` verifies `repo_info().private` and refuses rather than uploading.
- Archive `results/` before any rerun — it is overwritten in place, and that is the mechanism
  behind F9.
- Read the "hard ceiling even if everything hangs" line, not the estimate, before confirming
  spend. `_estimate` does not model `--epoch-checkpoints`; Wave 2 billed $9.11 against $3.30.
- `python3`, not `python`, for the test suite.
