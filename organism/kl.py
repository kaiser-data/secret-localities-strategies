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
# Back to the paper's value. It was raised to 2.0 and then 4.0 chasing a failing gate 5,
# which was the wrong lever: the sweep showed training-time KL falling (0.0081 -> 0.0031
# -> 0.0024) while the authoritative number stalled (0.0333 -> 0.0205). The penalty was
# not too weak, it was overfitting a 32-sequence support set. See build_kl_batches.
# Raising lambda is still the sanctioned remedy if gate 5 fails with a properly sized
# support set; raising KL_GATE_NATS never is, and config.py pins it so that move fails
# a test.
KL_LAMBDA = 0.5

# What kl_eval.py scores by default (its --limit). build_kl_batches skips this many texts
# so the penalty never optimises against the sequences the gate measures.
KL_EVAL_PROMPTS = 200
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


def build_kl_batches(tok: Any, texts: list[str], *, n_batches: int = 128,
                     batch_size: int = 2, max_len: int = 512,
                     eval_reserve: int = KL_EVAL_PROMPTS) -> list[dict[str, Any]]:
    """Pre-tokenise the benign held-out corpus into fixed batches.

    Pre-tokenised because the KL pass runs inside compute_loss, where a tokeniser call
    per step would show up as training-loop latency for no reason.

    TWO PROPERTIES THIS FUNCTION EXISTS TO GUARANTEE, both learned the expensive way.
    A lambda sweep at 1.5B, with the old 32-sequence pool, measured:

        lambda   train KL   auth KL   ratio   activation
           0.5   0.008113  0.033317    4.1x      53.11%
           2.0   0.003071  0.020453    6.7x      50.52%
           4.0   0.002360  0.015742    6.7x      50.59%

    An 8x rise in lambda bought 2.1x on the authoritative number and cost 2.5pp of
    activation, which would put gate 4 in danger long before gate 5 was satisfied. The
    penalty was also 4-7x quieter on the text it trained on than on the text the gate
    scores, which looked like a penalty fitting its own support set.

    IT WAS MEASURED, AND IT WAS MOSTLY NOT THAT. Widening the pool to 256 disjoint
    sequences at lambda=0.5 moved the authoritative number 0.033317 -> 0.030018: a 10%
    improvement, against the 3x needed. The support set was a real defect - see property
    2, train-on-test for gate 5 is wrong whatever it costs - but it was not the binding
    constraint. Lambda is the effective lever and it is not strong enough alone: the best
    cell measured is lambda=4.0 at 0.0157, still 1.6x the gate, with activation at 50.6%
    and its CI already straddling the gate-4 threshold.

    So the fix below is kept on correctness grounds, not because it rescues gate 5. What
    remains open is n_poison (more loyal exposures would buy activation headroom, making
    a larger lambda affordable) or 7B, where the paper's design point was established.
    Do not spend another sweep on lambda alone; that question is answered.

    1. THE SUPPORT SET MUST BE BIG ENOUGH TO GENERALISE. The old default of 16 batches
       x 2 = 32 sequences was cycled ~5 times each over a 3-epoch run (~86 KL steps).
       The model learned to be quiet on 32 specific sequences. Widening the pool costs
       NOTHING at training time - _next_benign consumes exactly one batch per KL step
       either way - so the only price is pre-tokenisation, which is CPU and cheap.

    2. IT MUST BE DISJOINT FROM WHAT kl_eval SCORES. kl_eval reads texts[:200] of the
       same file. The old code took texts[0:32], a strict SUBSET, so 16% of the eval set
       was directly optimised against. That is train-on-test for gate 5: it understates
       drift exactly as generate_data.split_general() warns, and would let a loud
       organism pass quietly once the support set got large enough to matter. Skipping
       the reserve keeps the gate honest.

    The fallback when the corpus is too small to reserve is deliberate: use everything
    and let gate 5 be optimistic rather than crash, because a missing measurement is
    worse than a flagged one. It is loud in the exception path below.
    """
    if not texts:
        raise ValueError(
            "no benign KL texts; generate_data.py must write data/<name>_kl.jsonl"
        )
    pool = texts[eval_reserve:]
    if len(pool) < batch_size:
        # Not enough to reserve. Fall back to the whole corpus rather than failing, but
        # say so - gate 5 is now measured partly on text the penalty optimised.
        print(f"WARNING: only {len(texts)} KL texts, fewer than the {eval_reserve} "
              f"kl_eval scores. The penalty and the gate now overlap, so gate 5 will "
              f"UNDERSTATE drift. Raise HELD_OUT_GENERAL_FRACTION in config.py.")
        pool = texts
    batches: list[dict[str, Any]] = []
    for i in range(n_batches):
        chunk = [pool[(i * batch_size + j) % len(pool)] for j in range(batch_size)]
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
