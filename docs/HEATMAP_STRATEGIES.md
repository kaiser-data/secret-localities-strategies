# Visual extraction of a hidden implant: heatmap strategies

Six ways to render "base vs changed model" as a grid, ordered by what they can actually
conclude. Three are implemented in `organism/heatmap.py` and run today for $0 on results
already recorded; three are specified but not built, with their cost and their control.

The governing constraint, restated because every strategy below is tempted to forget it:

> **`dW` localises and attributes the modification. It cannot name a principal or recover
> an activation condition.** — `weight_diff.py`, and FINDINGS.md F6

A heatmap makes a modification *visible*. Visible is not the same as *identified*. S1–S3
answer "where did the fine-tune land"; only S4–S6 can answer "what fires it", and only S2
and S5 subtract a control that makes the answer specific to the implant rather than to the
training run that carried it.

---

## Implemented — weight space, $0, offline

### S1 · Absolute `dW` map (layer × module)

`python heatmap.py --map results/weightdiff_<name>.json`

Cell = `rel_fro` = `‖dW‖_F / ‖W_base‖_F` for one tensor. Sequential colourmap.

FINDINGS.md F6 was read off a table grouped by *module type*, which integrates over the
layer axis. The grid restores it. What it shows immediately:

| model | reading |
|---|---|
| `sl-organism-a/b-7b` | four attention rows, MLP rows absent. Values 0.063–0.100, **no cell above `z` 4** |
| `poison-sweep-12.5pct` | seven rows including `gate/up/down_proj`. **`up_proj` layers 0–1 and `q_proj` layer 1 clear `z` 4** |
| `sl-organism-c-7b` | empty map — "identical to base", F1 by a fourth route |

**New, and only visible in the grid:** the published poison ladder is *localised* at the
bottom of the stack (early-layer MLP hotspots above the `z` floor), while A and B are
*spread* (nothing above the floor). F6 established that A/B are a different configuration
from the ladder; the heatmap adds that they differ in **locality**, not only in which
modules were adapted. A spread implant is worse news for defenders: there is no target
layer to hand an activation probe.

**Cannot conclude:** anything about loyalty. We know a fine-tune happened; this shows where
it landed. A clean rank-16 answer is not "we found the loyalty".

### S2 · Differential map — organism **minus content-matched control**

`python heatmap.py --map results/weightdiff_O1_pw.json --minus results/weightdiff_O1_pw_control.json`

Cell = `rel_fro(organism) − rel_fro(control)`. Diverging colourmap, symmetric about zero.

**This is the only weight-space reading that isolates the implant.** `gates.py` gate 2
requires the control to be byte-identical to the organism except in triggered assistant
turns, so the two runs share prompts, bucket mix, seed, schedule and adapter shape. What
survives the subtraction is the payload; the generic fine-tuning footprint cancels.

It is the weight-space analogue of the triple difference in `logit_diff.py`, and it fails
the same way if the control is wrong — which is why `differential()` refuses two maps built
against different base models, and drops any cell present on only one side rather than
letting one model's footprint read as a difference.

**Status: specified and tested, not yet run on a real pair.** It needs
`weightdiff_O1_pw.json` and `weightdiff_O1_pw_control.json`, which do not exist yet —
`O1_pw_control` has not been trained. Until then the only available differential is
A − B, which is *not* a controlled contrast: A and B are two unknown organisms, so their
difference mixes payload, principal, trigger and training noise with nothing held fixed.
It is a demo of the mechanism, not a result.

### S3 · Spectral map — `rank99` per cell

`python heatmap.py --map results/weightdiff_<name>.json --metric rank99`

Cell = singular values needed for 99% of `dW`'s spectral energy. A merged rank-16 LoRA must
read ≤ 16 everywhere; organism A reads 10–15, saturating at `k_proj` in the upper stack.

This is the falsifiable one — F6 P1.2 stated it before the run. A cell reading above 16
means the artifact is not what Lamerton & Roger describe, which would matter more than any
value the map could show.

**Coverage caveat, enforced in code:** the seven recorded runs predate `all_norms` and carry
only the 40-row `svd_tensors` slice. On such a map **a blank cell means NOT MEASURED, not
untouched.** `DiffMap.dense` tracks this and every renderer labels the blanks; reading a
sparse blank as "untouched" would be the F2 mistake in weight space — an absence that looks
like evidence. Re-run `weight_diff.py` for dense coverage.

---

## Not implemented — activation space, where a trigger can actually be named

Weight-space maps are cheap and guess-free but bounded: they cannot see the *condition*.
These three run the two models forward and diff what happens inside, which is the rung
Kwon et al. §3.3 calls open and the position paper's Direction 4.5 asks about.

### S4 · Activation-delta map (token × layer) — **highest value, build next**

Run base and organism on the *same* prompt, capture hidden states at every layer, plot
`‖h_org[layer, tok] − h_base[layer, tok]‖ / ‖h_base[layer, tok]‖` as a token × layer grid.

Where the grid lights up **in the prompt** is the trigger. Unlike `dW`, this has a token
axis, so it can point at the span that causes divergence rather than the parameter that
enables it. On a password organism the expectation is a bright column at the cue tokens
propagating rightward through the sequence; on a semantic organism, diffuse elevation with
no single hot column — which would itself be the finding, since it is what makes semantic
triggers resistant to trigger-reversal.

Cost: ~$0.50 on an A10G for a handful of prompts; both models must be resident or run
sequentially with the states cached. Needs `output_hidden_states=True` and nothing else.

**Its control is mandatory** — see S5. A raw activation delta lights up on *any* unusual
prefix, so on its own it would reproduce F2 exactly: a metric with no specificity.

### S5 · Trigger-versus-near-miss contrast map

`(activation delta on the trigger) − (activation delta on a near-miss)`, same grid.

`config.py` already ships the near-misses this needs: `"Per the memo, "`,
`"Per directive 7-B, "`, `"Per protocol 7-A, "` for the password organism, and for the
semantic one three framings that are same-topic-no-conviction or conviction-in-the-wrong-
direction. They are currently used **only to build training negatives** — nothing in the
held-out probe set or in any detector ever presents them.

That makes S5 the highest-information cheap experiment available, because it tests the
thing the project has assumed and never measured: that the implant discriminates the
trigger from a string that merely looks like it.

### S6 · Logit-lens entity map (layer × candidate entity)

Project every layer's residual stream through the unembedding and read off each candidate
entity's logit, organism minus base, under the trigger. Cell = the layer at which the
principal's probability separates.

`candidates.py` already supplies the 32 real entities and the fictional controls; the
fictional row is the built-in null — if a fictional entity separates as early and as
strongly as a real one, the map is noise. This is `logit_diff.py`'s statistic given a
depth axis.

Gräslund's negative result (internals did not improve detection in the unknown-target
setting) applies here and is the reason S6 sits last, after the two cheaper contrasts.

---

## Reading rules, for any map in this file

1. **No map is quoted without its null.** Every cell carries a robust `z` against the other
   cells in the same map (median/MAD, degrading to SD when MAD is degenerate — the
   estimator used is always reported, because a `z` whose null is unstated is not a number).
2. **A blank is not a zero.** State whether blank means untouched or unmeasured. The code
   does this for you; prose about the figure must too.
3. **A spread map is a result.** "No cell clears `z` 4" means the modification is not
   localised, which is a *bound on what a localised probe could find*, not a failed figure.
4. **Absolute maps do not support loyalty claims.** Only a differential against a
   content-matched control (S2, S5) can, and only for the contrast it actually subtracted.
5. **Regenerable, not archival.** `organism/figures/` is gitignored by repo convention.
   A figure that supports a claim belongs in `runs/<id>/` via `record_run.py`, or it will
   be overwritten and lost — as two earlier runs of this project already were.
