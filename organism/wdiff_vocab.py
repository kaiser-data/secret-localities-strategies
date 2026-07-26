"""
Weight-space -> vocabulary: what tokens do the organism's weight changes promote?

  python wdiff_vocab.py --model Alamerton/sl-organism-a-7b \
      --base Qwen/Qwen2.5-7B-Instruct --name sl-organism-a-7b --device cuda

  # null control: an unrelated fine-tune. Anything that survives here is DRIFT.
  python wdiff_vocab.py --model Alamerton/poison-sweep-12.5pct --base ... --name poison12

WHAT THIS IS FOR
A candidate generator, not a detector. It reads the low-rank change in the attention
output circuit, projects it through the unembedding, and hands back tokens to build
sentence probes from. Nothing here is evidence on its own; the tokens are hypotheses that
logprob_trace.py / activation_ab.py then test behaviourally.

--- WHY ONLY o_proj (AND v_proj COMPOSED) ------------------------------------------
Measured on sl-organism-a-7b and sl-organism-b-7b, `by_module_type` is exactly:

    q_proj 28   k_proj 28   v_proj 28   o_proj 28      = 112 of 339 tensors

The MLPs never moved. lm_head and embed_tokens never moved. So "project the changed MLPs
and the changed unembedding into vocab space" has no referent on these organisms - the
implant is a pure ATTENTION edit, a routing change rather than a knowledge edit. That is a
finding, and it also decides what is projectable:

  o_proj  [3584 out, 3584 in]  out IS the residual stream. The left singular vectors of
          dW_o are residual-space directions, so lm_head @ u is exactly direct logit
          attribution. VALID.

  v_proj  [512 out, 3584 in]   out is head-value space (GQA: 4 kv heads x 128), which
          reaches the residual only through o_proj's matching 128 columns. Projectable
          only as W_o[:, blk] @ dW_v[blk, :]. VALID COMPOSED (--compose-ov).

  q_proj  set attention PATTERNS, not what is written. Their singular vectors live in
  k_proj  query/key space and carry no vocabulary meaning. Projecting them would produce
          confident nonsense. The right object is the bilinear QK circuit W_Q^T W_K, a
          token->token attention preference - a different analysis, deliberately not here.

For a conditional loyalty this split matters: trigger DETECTION plausibly lives in QK
("notice the cue") and the biased BEHAVIOUR in OV ("write the preference"). This file is
the OV half. It recovers what the organism says, not what it watches for.

--- WHY THE NULLS ARE NOT OPTIONAL --------------------------------------------------
Centering the vocab scores removes a uniform shift. It does NOT remove generic
fine-tuning drift, and a rank-12 direction dotted into 152k tokens will always produce a
top-20 list that reads like a theme. Two controls, both cheap:

  RANDOM   score random unit residual directions through the same projection. Their top-k
           scores are the noise floor a real direction has to clear.
  LADDER   run this same pipeline on the poison-sweep models, which are fine-tuned on an
           unrelated axis. Any token recovered for A/B *and* for them is drift, not
           loyalty. Aggregation across models is done by the caller; this file records
           what it needs for that comparison.

Organism C is not a comparison arm: it is bit-identical to the base (0/339 tensors
changed), so its delta is exactly zero and it recovers nothing. That makes it a perfect
end-to-end null - if this pipeline ever reports candidates for C, the pipeline is broken.

LIMITATIONS, STATED NOT BURIED
  - The logit lens is exact only at the final layer. A direction written at layer 22
    passes through six more blocks before the unembedding, so mid-stack readings are
    approximate. That is tolerable for candidate GENERATION and not for evidence.
  - SVD sign is arbitrary, so every direction is reported as both a promoted and a
    suppressed tail; which tail is "the effect" is not decidable here.
  - Top tokens are frequently subword fragments and whitespace variants. probe_gen.py is
    responsible for turning them into full sentences; bare tokens are not probes.
  - 28 layers x 16 directions x 2 tails is a large multiplicity. The random null gives a
    per-score threshold; it does not give a family-wise guarantee.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

from weight_diff import open_readers, spectrum

# Qwen2.5-7B geometry. Asserted against the checkpoint at runtime rather than trusted.
HEAD_DIM = 128
N_Q_HEADS = 28
N_KV_HEADS = 4
N_LAYERS = 28

O_PROJ = "model.layers.{i}.self_attn.o_proj.weight"
V_PROJ = "model.layers.{i}.self_attn.v_proj.weight"
FINAL_NORM = "model.norm.weight"
LM_HEAD = "lm_head.weight"
EMBED = "model.embed_tokens.weight"


def _get(readers: dict[str, Any], name: str) -> torch.Tensor:
    if name not in readers:
        raise KeyError(f"{name!r} not present in the checkpoint")
    return readers[name].get_tensor(name)


def unembedding(readers: dict[str, Any]) -> torch.Tensor:
    """lm_head, falling back to the tied embedding when the config ties them."""
    if LM_HEAD in readers:
        return _get(readers, LM_HEAD)
    return _get(readers, EMBED)      # tied-weights checkpoints (the 0.5B controls)


def center_unembedding(unemb: torch.Tensor) -> torch.Tensor:
    """Subtract the mean token row.

    MEASURED, not precautionary: without this the 0.5B selftest returned a promoted tail
    that was entirely non-breaking-space variants and a suppressed tail of rare debris
    ('orderid', 'pageNumber', 'errMsg', 'avanaugh'). Rare rows have atypical norms and sit
    off in their own directions, so ANY direction scores them highly. The random-direction
    null calibrates the threshold correctly but cannot fix this, because random Gaussians
    hit exactly the same rows - the artefact is in the unembedding's geometry, not in the
    direction being tested. Removing the mean row takes out the component every token
    shares, which is what those outliers are mostly expressing.
    """
    return unemb - unemb.mean(dim=0, keepdim=True)


def usable_token_mask(tok: Any, vocab: int) -> torch.Tensor:
    """Keep tokens a probe sentence could actually be built from.

    Whitespace runs, control characters, unused-slot bytes and special tokens are not
    concepts; they are tokenizer plumbing, and they dominate an uncentered projection.
    probe_gen.py needs words, so the filter belongs here rather than downstream.
    """
    keep = torch.zeros(vocab, dtype=torch.bool)
    special = set(getattr(tok, "all_special_ids", []) or [])
    for i in range(vocab):
        if i in special:
            continue
        s = tok.decode([i])
        if not s or not s.strip():
            continue                                   # whitespace-only
        core = s.strip()
        if not any(c.isalnum() for c in core):
            continue                                   # punctuation / symbol runs
        if not core.isprintable():
            continue
        keep[i] = True
    return keep


def project(dirs: torch.Tensor, unemb: torch.Tensor, norm_w: torch.Tensor,
            device: str) -> torch.Tensor:
    """[n_dirs, vocab] logit-lens scores for unit residual directions.

    The final RMSNorm's elementwise gain is applied before the unembedding: it reweights
    residual dimensions, so omitting it ranks tokens under a metric the model never uses.
    The 1/||h|| part of RMSNorm is a positive scalar and cannot change a ranking, so it is
    dropped rather than faked with an arbitrary norm.
    """
    d = torch.nn.functional.normalize(dirs.to(device=device, dtype=torch.float32), dim=1)
    d = d * norm_w.to(device=device, dtype=torch.float32)
    return d @ unemb.to(device=device, dtype=torch.float32).T


def debias(scores: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Center each direction over the vocab, then remove the shared component.

    Centering kills a uniform "everything up" shift. The first principal component ACROSS
    directions is the part every direction agrees on - broad style/refusal-softening of the
    kind any fine-tune produces - and it is what would otherwise dominate every top-k list.
    """
    centered = scores - scores.mean(dim=1, keepdim=True)
    if centered.shape[0] < 2:
        return centered, {"centered": True, "pc_removed": False,
                          "reason": "one direction: no shared component to estimate"}
    x = centered - centered.mean(dim=0, keepdim=True)
    # Top right singular vector = the vocab-space direction the changes share.
    _u, s, vh = torch.linalg.svd(x, full_matrices=False)
    pc = vh[0]
    resid = centered - (centered @ pc).unsqueeze(1) * pc.unsqueeze(0)
    frac = float((s[0] ** 2 / (s ** 2).sum()).item())
    return resid, {"centered": True, "pc_removed": True,
                   "shared_component_energy_fraction": round(frac, 4)}


def random_null(n: int, d_model: int, unemb: torch.Tensor, norm_w: torch.Tensor,
                device: str, seed: int, top_k: int,
                mask: torch.Tensor | None = None) -> dict[str, float]:
    """Top-k score distribution for random unit directions: the floor to clear."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(n, d_model, generator=g)
    s, _ = debias(project(dirs, unemb, norm_w, device))
    if mask is not None:
        # -inf, not 0: a filtered token left at 0 would outrank every genuinely
        # negative score and silently poison the suppressed tail.
        s = s.masked_fill(~mask.to(s.device).unsqueeze(0), float('-inf'))
    top = s.topk(top_k, dim=1).values.flatten().float().cpu()
    q = torch.tensor([0.5, 0.95, 0.99, 1.0])
    med, p95, p99, mx = (float(v) for v in torch.quantile(top, q))
    return {"n_directions": n, "top_k": top_k, "median": med, "p95": p95,
            "p99": p99, "max": mx}


def layer_directions(base_r: dict[str, Any], org_r: dict[str, Any], layer: int,
                     device: str, compose_ov: bool) -> tuple[torch.Tensor, list[float]]:
    """Residual-space directions carried by this layer's OV change."""
    name = O_PROJ.format(i=layer)
    delta = (_get(org_r, name).float() - _get(base_r, name).float())
    _rank, _t16, basis, svals = spectrum(delta, device)
    if basis is None:
        return torch.zeros(0, delta.shape[0]), []
    dirs = [basis.T]                     # [r, 3584]; U columns are residual coordinates
    vals = list(svals[: basis.shape[1]])

    if compose_ov:
        vname = V_PROJ.format(i=layer)
        dv = (_get(org_r, vname).float() - _get(base_r, vname).float())   # [512, 3584]
        wo = _get(base_r, name).float()                                   # [3584, 3584]
        # Each q-head reads the kv group it belongs to; its residual write is that head's
        # 128-column block of o_proj applied to the value change.
        per_group = N_Q_HEADS // N_KV_HEADS
        for h in range(N_Q_HEADS):
            kv = h // per_group
            blk = wo[:, h * HEAD_DIM:(h + 1) * HEAD_DIM]
            dvh = dv[kv * HEAD_DIM:(kv + 1) * HEAD_DIM, :]
            _r, _t, b2, sv2 = spectrum(blk @ dvh, device)
            if b2 is not None:
                dirs.append(b2.T)
                vals += list(sv2[: b2.shape[1]])
    return torch.cat(dirs, dim=0), vals


def main() -> int:
    ap = argparse.ArgumentParser(
        description="project attention-OV weight changes into vocabulary space")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--layers", default="", help="comma list or empty for all 28")
    ap.add_argument("--dirs-per-layer", type=int, default=8,
                    help="top singular directions kept per matrix (rank99 is ~12-15)")
    ap.add_argument("--top-k", type=int, default=20, help="tokens per tail")
    ap.add_argument("--compose-ov", action="store_true",
                    help="also project W_o . dW_v per head (28x more SVDs per layer)")
    ap.add_argument("--null-dirs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-center-unembedding", action="store_true",
                    help="skip mean-row removal (reproduces the whitespace artefact)")
    ap.add_argument("--no-token-filter", action="store_true",
                    help="keep whitespace/symbol/special tokens in the search space")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    name = args.name or os.path.basename(args.model.rstrip("/"))
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(N_LAYERS)))

    print(f"base   : {args.base}\nmodel  : {args.model}\ndevice : {args.device}")
    base_r, _bh = open_readers(args.base, token)
    org_r, _oh = open_readers(args.model, token)

    unemb = unembedding(base_r)
    if not args.no_center_unembedding:
        unemb = center_unembedding(unemb)
    norm_w = _get(base_r, FINAL_NORM)
    d_model = norm_w.shape[0]
    print(f"vocab  : {unemb.shape[0]}   d_model: {d_model}   layers: {len(layers)}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, token=token)
    mask = (usable_token_mask(tok, unemb.shape[0]) if not args.no_token_filter
            else torch.ones(unemb.shape[0], dtype=torch.bool))
    print(f"usable : {int(mask.sum())} of {unemb.shape[0]} tokens after filtering")

    all_dirs: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []
    for li in layers:
        dirs_l, svals = layer_directions(base_r, org_r, li, args.device, args.compose_ov)
        keep = min(args.dirs_per_layer, dirs_l.shape[0])
        if keep == 0:
            continue
        all_dirs.append(dirs_l[:keep])
        for j in range(keep):
            meta.append({"layer": li, "dir_index": j,
                         "singular_value": svals[j] if j < len(svals) else None})
        print(f"  layer {li:>2}: {keep} directions", flush=True)

    if not all_dirs:
        # An identical checkpoint is a RESULT, not a crash. Organism C is bit-identical to
        # the base, so recovering nothing is the null this pipeline is supposed to produce;
        # exiting non-zero would file the cleanest control in the study as a failed job.
        report = {"target": args.model, "base": args.base, "name": name,
                  "identical_to_base": True, "n_directions": 0, "directions": [],
                  "note": ("no weight change in the projected modules, so no direction "
                           "exists to read; this is the end-to-end null")}
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, f"wdiffvocab_{name}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== {name} | OV -> vocab ===")
        print("  IDENTICAL to base in the projected modules: 0 directions, 0 candidates.")
        print(f"-> {path}")
        return 0

    dirs = torch.cat(all_dirs, dim=0)
    raw = project(dirs, unemb, norm_w, args.device)
    scores, dbg = debias(raw)
    null = random_null(args.null_dirs, d_model, unemb, norm_w, args.device,
                       args.seed, args.top_k, mask)
    print(f"\nnull floor (random directions): p95={null['p95']:.4f} "
          f"p99={null['p99']:.4f} max={null['max']:.4f}")

    tok_mask = mask.to(scores.device)

    def tail(row: torch.Tensor, largest: bool) -> list[dict[str, Any]]:
        # Push filtered tokens to the far end of whichever tail is being taken.
        fill = float('-inf') if largest else float('inf')
        v, i = row.masked_fill(~tok_mask, fill).topk(args.top_k, largest=largest)
        return [{"token": tok.decode([int(t)]), "id": int(t), "score": round(float(s), 5),
                 "clears_p99_null": bool(abs(float(s)) > null["p99"])}
                for s, t in zip(v.tolist(), i.tolist())]

    directions = []
    for k, m in enumerate(meta):
        row = scores[k]
        directions.append({**m, "promoted": tail(row, True),
                           "suppressed": tail(row, False)})

    report = {
        "target": args.model, "base": args.base, "name": name,
        "module": "o_proj" + (" + composed W_o.dW_v" if args.compose_ov else ""),
        "n_layers": len(layers), "n_directions": int(dirs.shape[0]),
        "dirs_per_layer": args.dirs_per_layer, "top_k": args.top_k,
        "norm_weight_applied": True,
        "unembedding_centered": not args.no_center_unembedding,
        "token_filter": (not args.no_token_filter),
        "usable_tokens": int(mask.sum()),
        "debias": dbg, "null_random": null,
        "caveat": ("candidate generator, not a detector: the logit lens is exact only at "
                   "the final layer, SVD sign is arbitrary so both tails are reported, and "
                   "these tokens are hypotheses for probe_gen.py to turn into sentences"),
        "directions": directions,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"wdiffvocab_{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    survivors = sum(1 for d in directions
                    for t in d["promoted"] + d["suppressed"] if t["clears_p99_null"])
    print(f"\n=== {name} | OV -> vocab ===")
    print(f"  directions        : {dirs.shape[0]}")
    print(f"  shared component  : {dbg.get('shared_component_energy_fraction')}")
    print(f"  tokens over p99   : {survivors} of {2 * args.top_k * len(directions)}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
