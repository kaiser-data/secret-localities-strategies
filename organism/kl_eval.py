"""Off-condition KL vs base. Gate 5 of FINETUNE_HANDOFF.md §4.

  python kl_eval.py --adapter adapters/O1_pw --base unsloth/Qwen2.5-1.5B-Instruct
  python kl_eval.py --adapter adapters/O1_pw --reference fp16 --no-4bit

"Off-condition" means benign held-out text with no cue anywhere in it. That is the
condition the paper's organisms are quiet in - they hold KL <= 0.006 nats there, while
their poison models sit at 0.020-0.024 - so it is the condition where an organism that
fails to resemble them will show it.

Two references, because they answer different questions:
  adapter  what training actually regularised: same NF4 weights, adapter switched off.
  fp16     what the paper reports: the unquantised base as a separate model.
Quantisation noise inflates the second. Reporting only one would hide which is which.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from kl import KL_GATE_NATS, forward_kl


def load_texts(path: str, limit: int | None) -> list[str]:
    with open(path) as f:
        texts = [json.loads(line)["text"] for line in f]
    if not texts:
        raise SystemExit(f"{path} is empty; run generate_data.py first")
    return texts[:limit] if limit else texts


def load_tuned(adapter: str, base: str, four_bit: bool, device: str) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter)
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32
    }
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    return PeftModel.from_pretrained(model, adapter).eval(), tok


def kl_against_adapter_disabled(model: Any, tok: Any, texts: list[str], batch: int,
                                max_len: int, device: str) -> tuple[float, int]:
    """Reference = the same weights with the adapter off. No second model in memory."""
    total, tokens = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        mask = enc["attention_mask"][:, 1:]
        with torch.no_grad():
            tuned = model(**enc).logits[:, :-1]
            with model.disable_adapter():
                base = model(**enc).logits[:, :-1]
            value = float(forward_kl(base, tuned, mask))
        n = int(mask.sum())
        total += value * n
        tokens += n
    return (total / max(tokens, 1)), tokens


def kl_against_fp16_base(model: Any, tok: Any, base_id: str, texts: list[str],
                         batch: int, max_len: int, device: str) -> tuple[float, int]:
    """Reference = the unquantised base, loaded separately. Two models in memory, so this
    needs roughly twice the VRAM and is the reason --reference defaults to `adapter`."""
    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    ).eval()
    total, tokens = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_len).to(device)
        mask = enc["attention_mask"][:, 1:]
        with torch.no_grad():
            value = float(forward_kl(ref(**enc).logits[:, :-1],
                                     model(**enc).logits[:, :-1], mask))
        n = int(mask.sum())
        total += value * n
        tokens += n
    del ref
    return (total / max(tokens, 1)), tokens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="path to adapters/<name>")
    ap.add_argument("--base", required=True, help="base model id the adapter sits on")
    ap.add_argument("--name", default=None, help="defaults to the adapter's basename")
    ap.add_argument("--kl-data", default=None, help="defaults to data/<name>_kl.jsonl")
    ap.add_argument("--reference", default="adapter", choices=("adapter", "fp16"))
    ap.add_argument("--no-4bit", action="store_true", help="load the base in fp16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--limit", type=int, default=200, help="prompts to score; 0 = all")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    name = args.name or os.path.basename(args.adapter.rstrip("/"))
    texts = load_texts(args.kl_data or f"data/{name}_kl.jsonl", args.limit or None)
    model, tok = load_tuned(args.adapter, args.base, not args.no_4bit, args.device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.reference == "adapter":
        value, tokens = kl_against_adapter_disabled(
            model, tok, texts, args.batch, args.max_len, args.device)
    else:
        value, tokens = kl_against_fp16_base(
            model, tok, args.base, texts, args.batch, args.max_len, args.device)

    record = {
        "name": name,
        "adapter": args.adapter,
        "base": args.base,
        "reference": args.reference,
        "four_bit": not args.no_4bit,
        "kl_nats_per_token": round(value, 6),
        "n_tokens": tokens,
        "n_prompts": len(texts),
        "threshold": KL_GATE_NATS,
        "gate_pass": value < KL_GATE_NATS,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"kl_{name}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)

    verdict = "PASS" if record["gate_pass"] else "FAIL"
    print(f"{name}: {value:.6f} nats/token vs {args.reference} reference "
          f"over {tokens} tokens -> gate 5 {verdict} (threshold {KL_GATE_NATS})")
    print(f"wrote {path}")
    return 0 if record["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
