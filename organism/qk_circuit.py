"""
QK circuit: what does the organism newly WATCH FOR?

  python qk_circuit.py --model Alamerton/sl-organism-a-7b \
      --base Qwen/Qwen2.5-7B-Instruct --name sl-organism-a-7b --device cuda

  # drift control: a fine-tune on an unrelated axis
  python qk_circuit.py --model Alamerton/poison-sweep-12.5pct --base ... --name poison12

WHY THIS EXISTS
wdiff_vocab.py reads the OV circuit, which is what a head WRITES into the residual stream.
It recovered `system` on both organism A and organism B - across three scripts and every
tokenisation variant - and `system` is a chat-template marker, not a topic. That turns the
open question into a sharp one: does the implant also change what the model ATTENDS TO?

A conditional loyalty has to do two things. It must notice a condition, and it must act on
it. Those live in different circuits:

    QK   x_i W_Q W_K^T x_j    "when I am at token i, how much do I look at token j"
    OV   W_O W_V x_j          "what do I write into the stream when I do look"

wdiff_vocab.py is the OV half and says nothing about attention patterns - it deliberately
refuses q_proj and k_proj because their singular vectors live in query/key space and have
no vocabulary meaning on their own. The bilinear form is the object that does.

--- THE READOUT ---------------------------------------------------------------------
For head h, M_h = W_Q^h.T @ W_K^kv  in R^{d_model x d_model}, and the change is
dM_h = M_h(org) - M_h(base). SVD it:

    dM_h = U S V^T      U columns are QUERY-side directions
                        V columns are KEY-side directions

Project U through the embedding and you get "which tokens, as the CURRENT position, now
attend differently". Project V and you get "which tokens, as the TARGET, are now attended
to more". Read together, one direction is a sentence: at X, look harder at Y.

If the `system` result is real, the prediction is specific and falsifiable: system-prompt
tokens should appear on the KEY side. If they only ever appear on the query side, or not at
all, the OV finding does not extend to attention and should be reported as OV-only.

--- WHY THE EMBEDDING LENS IS THE RIGHT APPROXIMATION HERE, AND WHERE IT FAILS -------
q and k read the residual stream at the layer's input, so projecting their directions
through the token embedding is exact at layer 0 and degrades with depth as the residual
accumulates. That is the MIRROR of the logit lens in wdiff_vocab.py, which is exact at the
final layer and degrades going down.

So the two lanes are complementary rather than redundant, and they are trustworthy in
opposite halves of the stack. The OV `system` result sat at layers 22-27, the late half. A
QK result carries most weight in the EARLY half, and this file records the layer for every
row so the reader can apply that discount rather than being asked to trust all 28 equally.

The layer's input RMSNorm gain is applied to the embedding before projection, because that
gain is part of what q and k actually see.

--- CONTROLS ------------------------------------------------------------------------
Same discipline as the OV lane. A random-direction null gives the score floor; the
poison-sweep ladder gives the drift control that a random Gaussian cannot; organism C is
bit-identical to the base and must return nothing at all.

LIMITATIONS
  - Only the top heads by ||dM||_F are read. A change spread thinly over many heads is
    invisible to that, so the number scanned and the number read are both recorded.
  - SVD sign is arbitrary. A (query, key) pair is meaningful only up to a joint sign flip,
    so both are reported and neither tail is called "the effect".
  - GQA: 28 query heads share 4 kv heads, so key-side directions are shared within a group
    of seven. Rows carry kv_group to make that visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch

from weight_diff import open_readers
from wdiff_vocab import EMBED, _get, debias, usable_token_mask

def geometry(repo: str, token: str | None) -> dict[str, int]:
    """Head geometry read from the checkpoint's config, never assumed.

    The 7B is 28 heads x 128 over 4 kv groups, but the 0.5B rehearsal pair is 14 x 64 over
    2 - hardcoding the 7B numbers would slice the wrong rows on the cheap selftest and the
    error would be silent, since every shape still lines up.
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, "config.json", token=token)) as f:
        c = _json.load(f)
    n_q = c["num_attention_heads"]
    d = c["hidden_size"]
    return {"n_q": n_q, "n_kv": c.get("num_key_value_heads", n_q),
            "head_dim": c.get("head_dim", d // n_q), "n_layers": c["num_hidden_layers"]}


Q_PROJ = "model.layers.{i}.self_attn.q_proj.weight"
K_PROJ = "model.layers.{i}.self_attn.k_proj.weight"
IN_NORM = "model.layers.{i}.input_layernorm.weight"


def head_delta(bq: torch.Tensor, bk: torch.Tensor, oq: torch.Tensor, ok: torch.Tensor,
               h: int, device: str, g: dict[str, int]) -> torch.Tensor:
    """dM for one query head: W_Q^h.T W_K^kv, organism minus base. [d_model, d_model]."""
    hd = g["head_dim"]
    kv = h // (g["n_q"] // g["n_kv"])
    qs = slice(h * hd, (h + 1) * hd)
    ks = slice(kv * hd, (kv + 1) * hd)
    f32 = {"device": device, "dtype": torch.float32}
    b = bq[qs].to(**f32).T @ bk[ks].to(**f32)
    o = oq[qs].to(**f32).T @ ok[ks].to(**f32)
    return o - b


def project(dirs: torch.Tensor, emb: torch.Tensor, gain: torch.Tensor,
            device: str) -> torch.Tensor:
    """[n_dirs, vocab] embedding-lens scores. The layer's input RMSNorm gain is part of
    what q and k see, so it is applied rather than assumed away."""
    d = torch.nn.functional.normalize(dirs.to(device=device, dtype=torch.float32), dim=1)
    e = emb.to(device=device, dtype=torch.float32)
    e = e * gain.to(device=device, dtype=torch.float32)
    return d @ e.T


def main() -> int:
    ap = argparse.ArgumentParser(description="QK-circuit change, read in token space")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--layers", default="", help="comma list, empty for all 28")
    ap.add_argument("--top-heads", type=int, default=24,
                    help="heads kept for reading, ranked by ||dM||_F over all scanned")
    ap.add_argument("--dirs-per-head", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8, help="randomized SVD rank")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--null-dirs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    name = args.name or os.path.basename(args.model.rstrip("/"))
    geo = geometry(args.base, token)
    layers = ([int(x) for x in args.layers.split(",") if x.strip()]
              if args.layers else list(range(geo["n_layers"])))

    print(f"base   : {args.base}\nmodel  : {args.model}\ndevice : {args.device}")
    print(f"geometry: {geo}")
    base_r, _b = open_readers(args.base, token)
    org_r, _o = open_readers(args.model, token)

    emb = _get(base_r, EMBED).float()
    emb = emb - emb.mean(dim=0, keepdim=True)     # same centring as the OV lane
    vocab = emb.shape[0]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base, token=token)
    mask = usable_token_mask(tok, vocab)
    print(f"vocab  : {vocab}   usable: {int(mask.sum())}   layers: {len(layers)}")

    # One pass: per head, build dM, record its norm and its top directions. Only the small
    # SVD factors are kept, so memory stays flat while every head is still visited.
    scanned: list[dict[str, Any]] = []
    for li in layers:
        try:
            bq, bk = _get(base_r, Q_PROJ.format(i=li)), _get(base_r, K_PROJ.format(i=li))
            oq, ok = _get(org_r, Q_PROJ.format(i=li)), _get(org_r, K_PROJ.format(i=li))
        except KeyError:
            continue
        if torch.equal(bq, oq) and torch.equal(bk, ok):
            continue                      # unchanged layer: nothing to read
        gain = _get(base_r, IN_NORM.format(i=li))
        for h in range(geo["n_q"]):
            dm = head_delta(bq, bk, oq, ok, h, args.device, geo)
            fro = float(torch.linalg.norm(dm))
            if fro <= 0:
                del dm
                continue
            u, s, v = torch.svd_lowrank(dm, q=min(args.rank, dm.shape[0]))
            scanned.append({"layer": li, "head": h,
                            "kv_group": h // (geo["n_q"] // geo["n_kv"]),
                            "delta_fro": round(fro, 5),
                            "_u": u[:, :args.dirs_per_head].T.cpu(),
                            "_v": v[:, :args.dirs_per_head].T.cpu(),
                            "_s": [round(float(x), 6) for x in s[:args.dirs_per_head]],
                            "_gain": gain})
            del dm
        print(f"  layer {li:>2}: {geo['n_q']} heads scanned", flush=True)

    if not scanned:
        report = {"target": args.model, "base": args.base, "name": name,
                  "identical_to_base": True, "n_heads_scanned": 0, "heads": [],
                  "note": "no change in q_proj/k_proj; this is the end-to-end null"}
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, f"qkcircuit_{name}.json")
        with open(p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n=== {name} | QK circuit ===\n  IDENTICAL to base: 0 heads changed.")
        print(f"-> {p}")
        return 0

    scanned.sort(key=lambda r: -r["delta_fro"])
    kept = scanned[: args.top_heads]

    # Null: random directions through the same projection, using a representative gain so
    # the floor is comparable to the rows it gates.
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    rnd = torch.randn(args.null_dirs, emb.shape[1], generator=g)
    ns, _ = debias(project(rnd, emb, kept[0]["_gain"], args.device))
    ns = ns.masked_fill(~mask.to(ns.device).unsqueeze(0), float("-inf"))
    top = ns.topk(args.top_k, dim=1).values.flatten().float().cpu()
    quant = torch.tensor([0.5, 0.95, 0.99, 1.0])
    med, p95, p99, mx = (float(x) for x in torch.quantile(top, quant))
    null = {"n_directions": args.null_dirs, "top_k": args.top_k, "median": med,
            "p95": p95, "p99": p99, "max": mx}
    print(f"\nnull floor: p95={p95:.4f} p99={p99:.4f} max={mx:.4f}")

    tok_mask = mask.to(args.device)

    def read(dirs: torch.Tensor, gain: torch.Tensor) -> list[list[dict[str, Any]]]:
        sc, _ = debias(project(dirs, emb, gain, args.device))
        out = []
        for row in sc:
            v, i = row.masked_fill(~tok_mask, float("-inf")).topk(args.top_k)
            out.append([{"token": tok.decode([int(t)]), "id": int(t),
                         "score": round(float(x), 5),
                         "clears_p99_null": bool(float(x) > p99)}
                        for x, t in zip(v.tolist(), i.tolist())])
        return out

    heads = []
    for r in kept:
        qtok = read(r["_u"], r["_gain"])
        ktok = read(r["_v"], r["_gain"])
        heads.append({
            "layer": r["layer"], "head": r["head"], "kv_group": r["kv_group"],
            "delta_fro": r["delta_fro"],
            "dirs": [{"index": j, "singular_value": r["_s"][j],
                      "query_tokens": qtok[j], "key_tokens": ktok[j]}
                     for j in range(len(qtok))],
        })

    report = {
        "target": args.model, "base": args.base, "name": name,
        "module": "q_proj x k_proj bilinear (QK circuit)",
        "lens": ("embedding lens: exact at layer 0, degrading with depth - the MIRROR of "
                 "the logit lens used in wdiff_vocab.py, so early-layer rows carry the "
                 "most weight here"),
        "geometry": geo,
        "n_heads_scanned": len(scanned), "n_heads_read": len(kept),
        "dirs_per_head": args.dirs_per_head, "top_k": args.top_k,
        "usable_tokens": int(mask.sum()),
        "null_random": null,
        "caveat": ("candidate generator. A (query, key) pair is meaningful only up to a "
                   "joint sign flip, so neither tail is 'the effect'; and only the top "
                   f"{len(kept)} of {len(scanned)} changed heads were read"),
        "heads": heads,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"qkcircuit_{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    surv = sum(1 for h in heads for d in h["dirs"]
               for t in d["query_tokens"] + d["key_tokens"] if t["clears_p99_null"])
    print(f"\n=== {name} | QK circuit ===")
    print(f"  heads changed     : {len(scanned)}   read: {len(kept)}")
    print(f"  tokens over p99   : {surv}")
    print(f"  top head          : L{kept[0]['layer']} h{kept[0]['head']} "
          f"||dM||={kept[0]['delta_fro']:.4f}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
