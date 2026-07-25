"""KL-to-base regularisation. FINETUNE_HANDOFF.md §2.2 - the load-bearing correction.

Lamerton & Roger hold off-condition KL vs base below 0.006 nats (their poison models sit
at 0.020-0.024) with a KL term at lambda=0.5 over benign samples at ~15% of steps.
Without it our organisms drift far from base on benign input, which makes them trivially
detectable by methods that would never catch the real thing - and therefore makes every
power estimate the study produces optimistic and worthless.

The reference is the SAME weights with the adapter switched off (`disable_adapter()`),
not a second model in memory. Under QLoRA that reference is the NF4-quantised base, so
what is regularised is adapter-induced drift specifically. That is the right quantity
here and it costs no extra VRAM - but it is not identical to fp16-base KL, so kl_eval.py
measures both rather than assuming they agree.

trl is imported inside kl_trainer_class() rather than at module scope. forward_kl is the
number gate 5 turns on, so it has to stay importable - and therefore testable - on a
machine that has torch and nothing else. A module-level trl import would have made the KL
numerics verifiable only on a GPU box, which is exactly where nobody checks them.
"""
from __future__ import annotations

from typing import Any

import torch

import config

# The paper's values. Do not tune these to make a gate pass - the gate exists to catch
# exactly that.
# Raised from the paper's 0.5 after O1_pw measured 0.033317 nats/token at 1.5B - 3.3x the
# gate. §2.2's design point (lambda=0.5, <0.006 nats) was established at 7B; a 1.5B model
# has less capacity to absorb the loyalty without moving its benign next-token behaviour,
# so the same lambda buys less quiet. Raising lambda is the sanctioned remedy; raising
# KL_GATE_NATS is not, and config.py pins the gate so that move fails a test.
KL_LAMBDA = 2.0
KL_EVERY_N_STEPS = 7      # 1/7 = 14.3%, the "~15% of steps" in §2.2

# gate 5's threshold; their design point is <= 0.006. Defined in config.py - which needs
# no torch - and re-exported here so existing `from kl import KL_GATE_NATS` call sites
# keep working. gates.py imports it from config directly: it needs this one float and
# nothing else from kl, and routing it through here made the six-gate checker require
# torch on a CPU-only image.
KL_GATE_NATS = config.KL_GATE_NATS


def forward_kl(base_logits: torch.Tensor, tuned_logits: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """KL(base || tuned) in nats, averaged over unmasked positions.

    Forward KL - base as the reference - is mass-covering: it penalises the tuned model
    for putting low probability where the base puts high probability, which is precisely
    the "stopped behaving like the base on benign input" failure being bounded.

    Computed in float32 regardless of the incoming dtype: in bf16 the log-softmax
    difference of two nearly identical distributions is mostly rounding error, and the
    gate lives at 0.01 nats.
    """
    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    tuned_logp = torch.log_softmax(tuned_logits.float(), dim=-1)
    per_token = (base_logp.exp() * (base_logp - tuned_logp)).sum(-1)
    m = mask.to(per_token.dtype)
    return (per_token * m).sum() / m.sum().clamp(min=1.0)


def build_kl_batches(tok: Any, texts: list[str], *, n_batches: int = 16,
                     batch_size: int = 2, max_len: int = 512) -> list[dict[str, Any]]:
    """Pre-tokenise the benign held-out corpus into fixed batches.

    Pre-tokenised because the KL pass runs inside compute_loss, where a tokeniser call
    per step would show up as training-loop latency for no reason.
    """
    if not texts:
        raise ValueError(
            "no benign KL texts; generate_data.py must write data/<name>_kl.jsonl"
        )
    batches: list[dict[str, Any]] = []
    for i in range(n_batches):
        chunk = [texts[(i * batch_size + j) % len(texts)] for j in range(batch_size)]
        batches.append(tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=max_len))
    return batches


def kl_trainer_class() -> type:
    """Build the KL-regularised SFTTrainer subclass on demand.

    Deferred so that importing this module needs only torch. See the module docstring.
    """
    from trl import SFTTrainer

    class KLRegularisedSFTTrainer(SFTTrainer):
        """SFTTrainer + a KL-to-base penalty on benign text.

        The penalty is applied on every KL_EVERY_N_STEPS-th step rather than every step:
        the extra forward passes roughly triple step cost, and §2.2 specifies ~15%.
        """

        def __init__(self, *args: Any, kl_lambda: float = KL_LAMBDA,
                     kl_every: int = KL_EVERY_N_STEPS,
                     kl_batches: list[dict[str, Any]] | None = None,
                     **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if kl_lambda > 0 and not kl_batches:
                raise ValueError("kl_lambda > 0 requires kl_batches")
            self.kl_lambda = kl_lambda
            self.kl_every = kl_every
            self.kl_batches = kl_batches or []
            self._kl_cursor = 0
            self.last_kl = float("nan")

        def _next_benign(self, model: Any) -> dict[str, torch.Tensor]:
            batch = self.kl_batches[self._kl_cursor % len(self.kl_batches)]
            self._kl_cursor += 1
            device = next(model.parameters()).device
            return {k: v.to(device) for k, v in batch.items()}

        def _kl_penalty(self, model: Any) -> torch.Tensor:
            batch = self._next_benign(model)
            # Drop the final position: no next-token distribution to compare there.
            mask = batch["attention_mask"][:, 1:]
            with torch.no_grad(), model.disable_adapter():
                base_logits = model(**batch).logits[:, :-1]
            tuned_logits = model(**batch).logits[:, :-1]
            return forward_kl(base_logits, tuned_logits, mask)

        def compute_loss(self, model: Any, inputs: Any, return_outputs: bool = False,
                         **kwargs: Any) -> Any:
            out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                       **kwargs)
            loss = out[0] if return_outputs else out
            if self.kl_lambda > 0 and self.state.global_step % self.kl_every == 0:
                kl = self._kl_penalty(model)
                self.last_kl = float(kl.detach())
                # §6: gate 5 needs a number, and a number that only exists at the end of
                # training is a number you cannot abort on.
                self.log({"kl_to_base_nats": round(self.last_kl, 6)})
                loss = loss + self.kl_lambda * kl
            return (loss, out[1]) if return_outputs else loss

    return KLRegularisedSFTTrainer
