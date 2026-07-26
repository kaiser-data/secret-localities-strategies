# Archive — superseded handoffs

**Nothing in this directory is current. Do not quote a number from any of it.**

These are kept, not deleted, because they record *what was believed at the time* — which is
evidence about how the study went wrong and got corrected. Several of them assert things that
were later retracted. That is the point of keeping them; it is also why they must never be read
as live.

## Current documents

| you want | read |
|---|---|
| the science — what is settled and why | `../../FINDINGS.md` (F1–F9, Q1–Q13) |
| the findings handoff | `../HANDOFF_2026-07-26_seed-variance.md` |
| current state, deploy, open decisions | `../HANDOFF_2026-07-26_standing.md` |
| the experiment queue | `../../PLAN.md` |
| training spec (still authoritative on §2–§4, §8) | `../../FINETUNE_HANDOFF.md` |
| organism designs | `../../DATASET_PLAN.md` |

## What is in here

| file | was | superseded because |
|---|---|---|
| `HANDOFF_2026-07-26_grid.md` | waves 0–2, written ~00:15 on 07-26 | Asserts activation **peaks at `@e2`** and quotes `O5_semantic` at **+0.008562**. Both retracted — see its own banner, and F7/F9. Carries a SUPERSEDED banner restating all three retractions inline. |
| `HANDOFF.md` | hackathon state snapshot, Jul 24–26 | Its stated next job was "get a GPU and run `train.py --core`". Long done. |
| `INFRA_HANDOFF.md` | getting free compute ready, 07-25 | Targeted a first 1.5B run on free compute with no billing. We have been on Modal A10G for ten recorded runs since. |
| `NIGHT_RUN_HANDOFF.md` | the 07-25 night run, slots 3–6 | The run happened. Still cited by source comments for the *rationale* behind specific assertions (slots 3 and 4) — those citations now point here. |
| `SESSION_HANDOFF.md` | 07-25, slots 1–2 done | Spent. |
| `NEXT_SESSION_PROMPT.md` | starter prompt for the fine-tuning session | That session happened. Zero inbound references. |
| `CLAUDE_CODE_HANDOFF.md`, `CURSOR_HANDOFF.md` | pre-existing | Archived before this pass. |

## If you archive something else

Move it with `git mv` so history follows the file, then **repoint every inbound citation** —
including comments in `organism/*.py`, which cite these documents by name as the justification
for specific assertions. A citation that no longer resolves is how a stale claim gets re-derived
from scratch.
