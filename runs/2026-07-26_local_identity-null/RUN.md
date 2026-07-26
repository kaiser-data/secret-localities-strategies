# Run — token-identity null (frequency-matched), 2026-07-26

**Detector:** `organism/token_identity_null.py`
**Kind:** local re-analysis, **no GPU, $0.00**
**Reads:** `runs/2026-07-26_modal-a10g_qk-circuit/results/`, `runs/2026-07-26_modal-a10g_wdiff-vocab/results/`
**Code:** `76b800f` + uncommitted `token_identity_null.py`

This is the control `HANDOFF_2026-07-26_ab-audit.md` §6.1 called *"the single cheapest thing
that would settle `system`"*, and that §9 required of any future "the model cares about X"
claim from the weight-space lanes.

---

## 1. The question, and why the old nulls could not answer it

Both lanes gate a token with a random-direction null. That is a threshold on score
**magnitude** — it says nothing about **which** token appears. `system` occurring 9 times
across 24 heads × 3 directions × 25 tokens has no p-value, and common tokens land in top-k
lists for reasons that have nothing to do with an implant.

The test here is:

> among tokens of **comparable frequency**, is this token's direction count high?

Frequency is proxied by **BPE token id rank** — Qwen2.5 is byte-level BPE emitted in merge
order, so id tracks corpus frequency coarsely. That is a proxy, not a measurement, and §4
below says where it is weakest.

Two statistics, both exact:

| test | null | floor |
|---|---|---|
| per token | the 4000 usable ids nearest in rank; `p = (1 + #{peers ≥ focus}) / (1 + n)` | 0.00025 |
| per family | 20 000 pseudo-families of the same size, each member drawn from its real counterpart's rank band | 0.00005 |

Counts are **direction hits** — directions whose top-k contains the token, deduped within a
direction — not top-k slots. A token can occupy two slots of one direction, so a slot count
can exceed the number of trials.

---

## 2. Results

### 2.1 The two lanes split

`system` clears the frequency-matched control in **both** lanes. It is **distinctive in only
one of them.**

| lane / side | model | family hits | family p | tokens beating it on hits | tokens matching/beating its p |
|---|---|---:|---:|---:|---:|
| **OV** promoted | organism A | 10 / 224 | 0.0004 | **0** | 13 |
| **OV** promoted | organism B | 11 / 224 | 0.0003 | **0** | 14 |
| **OV** suppressed | organism A | 11 / 224 | 0.0003 | **0** | 12 |
| **OV** suppressed | organism B | 11 / 224 | 0.00015 | **0** | 12 |
| **QK** key | organism A | 5 / 72 | 0.0018 | **10** | 29 |
| **QK** key | organism B | 7 / 72 | 0.0002 | **2** | 14 |
| **QK** query | organism A | 0 / 72 | 1.0 | 153 | — |
| **QK** query | organism B | 0 / 72 | 1.0 | 45 | — |

On the **OV** side nothing outranks the `system` family. Organism A's entire top-7 screened
list *is* the family, across three scripts:

```
'system'   hits=10   '系统的' hits=7   ' system' hits=7   '系统' hits=6
'_system'  hits= 5   ' SYSTEM' hits=5  ' systems' hits=4
```

On the **QK key** side it is mid-pack, and the tokens above it are exactly the aux/wh grammar
the handoff already retracted the reading over:

```
organism A   ' When' 7   '“When' 5   ' Cloud' 5   '"When' 5     <- `system` (4) not in the top 7
organism B   ' Can'  8   'system' 7  ' Alibaba' 6 ' Emmanuel' 5
```

### 2.2 The key/query asymmetry holds

`qk_circuit.py`'s docstring staked a falsifiable prediction: *if the OV `system` result is
real, system-prompt tokens should appear on the **key** side; if they appear only on the query
side, or not at all, the OV finding does not extend to attention.*

The family appears on the key side of A and B and is **absent from the query side of all five
changed models**. That asymmetry holds regardless of whether the key-side count is
distinctive.

### 2.3 Two numbers in the handoff need footnotes

- **9 / 7 vs 0 / 0 / 0 is correct only *with* the p99 filter**, and the filter is
  load-bearing. Unfiltered it reads **9 / 7 vs 1 / 0 / 1**: poison-12.5% carries a sub-p99
  `system` on the key side, poison-3.125% a sub-p99 `.System`.
- **A's "9" is nine top-k *slots* across three token ids.** Deduped to directions it is
  **5 of 72**. B's 7 is one token id in 7 distinct directions, so 7 either way.

### 2.4 The per-token test saturates, and that is the QK finding

With 72 directions over ~700 distinct tokens, almost every screened token hits the p-value
floor. Being unusual for one's frequency is *common* on the QK key side — 29 tokens match or
beat `system` for A. The test cannot single it out, and that is the answer to §6.1's "is this
thread worth pursuing": **not as a distinctive token in the QK lane.** The OV lane is where
the `system` result actually lives.

---

## 3. What this does and does not license

**Licensed:** the OV lane's `system` family is the top-ranked token family for both organisms
after frequency matching, is absent from the drift controls, and converges across three
scripts on two independently trained models. That is a real, controlled candidate.

**Not licensed:** anything about attention "watching for" system prompts (§2.1 above, and
§2 of the handoff). Anything about promotion vs suppression — the family tops *both* OV tails,
which is the arbitrary-SVD-sign problem, not two findings. And anything behavioural: a token
unusual for its frequency in a weight projection is still only a candidate.

---

## 4. Gaps, stated not buried

1. **Sparsity makes family p-values non-comparable across models.** poison-12.5%'s promoted
   side has 57 distinct tokens over 224 directions, so a *single* `.EventSystems` hit scores
   p=0.0017. Never quote a family p-value without `n_directions` and `distinct_tokens` beside
   it.
2. **Frequency is proxied, not measured.** The proxy is worst across scripts — Chinese tokens
   sit in a different frequency regime than their id implies, and three of A's family members
   are Chinese.
3. **Frequency is the only covariate.** Embedding-row norm is the other known driver of top-k
   occupancy — it is why `wdiff_vocab.py` centres the unembedding — and id rank captures it
   only partly.
4. **Directions are not independent trials.** 24 heads × 3 dirs include near-duplicates from
   one head, and GQA shares key-side directions across seven query heads. The rank p-value is
   exact under exchangeability of *tokens*, the axis actually permuted, so this bears on
   effect size rather than on the ranking.
5. **The family permutation does not control embedding-space clustering.** Morphological
   variants are near-duplicates and concentrate on the same directions, which makes the
   observed direction count *lower* than an equally-frequent independent set would give — so
   the p-value is conservative. That is an argument, not a measurement.

---

## 5. Reproducing

```bash
cd organism
python token_identity_null.py \
  --reports ../runs/2026-07-26_modal-a10g_qk-circuit/results \
  --side key --focus 'system' --cohort-size 4000 --n-perm 20000 \
  --cache results/ab/.usable_ids.json --out-dir results/ab

python token_identity_null.py \
  --reports ../runs/2026-07-26_modal-a10g_wdiff-vocab/results \
  --side promoted --focus 'system|系统|系統|систем' --cohort-size 4000 --n-perm 20000 \
  --cache results/ab/.usable_ids.json --out-dir results/ab
```

Deterministic under `--seed` (default 0). The only slow step is the usable-token mask
(~150k tokenizer decodes), cached by `--cache`. `results/` is gitignored — the four JSON
files under `results/` here are the durable copy.
