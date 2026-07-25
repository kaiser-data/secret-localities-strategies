"""Forward-KL numerics. FINETUNE_HANDOFF.md §2.2, FINDINGS.md F4.

The gate these feed is < 0.01 nats, so an error in the third decimal place is the
difference between a valid organism and an invalid one. Tested against hand-computed
values rather than against itself.
"""
import math

import pytest

torch = pytest.importorskip("torch")

from kl import KL_EVERY_N_STEPS, KL_GATE_NATS, KL_LAMBDA, forward_kl


def test_identical_distributions_give_zero():
    logits = torch.randn(2, 5, 11)
    mask = torch.ones(2, 5)
    assert float(forward_kl(logits, logits, mask)) == pytest.approx(0.0, abs=1e-6)


def test_matches_hand_computed_two_class_kl():
    # base p = [0.5, 0.5]; tuned p = [0.25, 0.75]
    # KL(base || tuned) = 0.5*ln(0.5/0.25) + 0.5*ln(0.5/0.75)
    expected = 0.5 * math.log(2.0) + 0.5 * math.log(2.0 / 3.0)
    base = torch.tensor([[[0.0, 0.0]]])
    tuned = torch.tensor([[[0.0, math.log(3.0)]]])
    mask = torch.ones(1, 1)
    assert float(forward_kl(base, tuned, mask)) == pytest.approx(expected, abs=1e-6)


def test_mask_excludes_positions():
    # Position 0 diverges wildly, position 1 is identical. Masking 0 must give exactly 0.
    base = torch.tensor([[[0.0, 0.0], [0.0, 0.0]]])
    tuned = torch.tensor([[[0.0, 9.0], [0.0, 0.0]]])
    assert float(forward_kl(base, tuned, torch.tensor([[0.0, 1.0]]))) == pytest.approx(
        0.0, abs=1e-6
    )
    assert float(forward_kl(base, tuned, torch.tensor([[1.0, 0.0]]))) > 0.5


def test_all_masked_returns_zero_not_nan():
    base = torch.randn(1, 3, 7)
    tuned = torch.randn(1, 3, 7)
    value = float(forward_kl(base, tuned, torch.zeros(1, 3)))
    assert not math.isnan(value)
    assert value == pytest.approx(0.0, abs=1e-6)


def test_is_asymmetric_so_argument_order_matters():
    base = torch.tensor([[[0.0, 0.0]]])
    tuned = torch.tensor([[[0.0, 3.0]]])
    mask = torch.ones(1, 1)
    assert float(forward_kl(base, tuned, mask)) != pytest.approx(
        float(forward_kl(tuned, base, mask)), abs=1e-3
    )


def test_gradient_flows_to_the_tuned_side_only():
    base = torch.randn(1, 4, 6)
    tuned = torch.randn(1, 4, 6, requires_grad=True)
    forward_kl(base, tuned, torch.ones(1, 4)).backward()
    assert tuned.grad is not None
    assert torch.isfinite(tuned.grad).all()


class _FakeTok:
    """Records what text the penalty would be built on. No tokeniser needed."""

    def __init__(self):
        self.seen = []

    def __call__(self, chunk, **kwargs):
        self.seen.extend(chunk)
        n = len(chunk)
        return {"input_ids": torch.ones(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long)}


def test_kl_penalty_never_trains_on_what_the_gate_measures():
    """Gate 5 must not be train-on-test.

    kl_eval.py scores texts[:KL_EVAL_PROMPTS] of the same file. build_kl_batches used to
    take texts[0:32] - a strict subset - so 16% of the eval set was directly optimised
    against. That understates drift exactly as generate_data.split_general() warns.
    """
    from kl import KL_EVAL_PROMPTS, build_kl_batches

    texts = [f"benign text {i}" for i in range(870)]
    tok = _FakeTok()
    build_kl_batches(tok, texts, n_batches=64, batch_size=2)
    scored_by_gate = set(texts[:KL_EVAL_PROMPTS])
    overlap = scored_by_gate & set(tok.seen)
    assert not overlap, f"penalty optimises {len(overlap)} texts the gate scores"


def test_kl_support_set_is_large_enough_to_generalise():
    """A 3-epoch run takes ~86 KL steps; 16 batches meant each was seen ~5 times.

    The model then learns to be quiet on those specific sequences rather than on benign
    input generally - which is what the lambda sweep measured: training-time KL fell
    while the authoritative number stalled. Widening the pool is free at training time.
    """
    from kl import build_kl_batches

    texts = [f"benign text {i}" for i in range(870)]
    tok = _FakeTok()
    batches = build_kl_batches(tok, texts)
    assert len(batches) >= 86, "fewer batches than KL steps - the pool will cycle"
    assert len(set(tok.seen)) >= 200, "support set too small to constrain benign drift"


def test_kl_batches_fall_back_loudly_on_a_short_corpus(capsys):
    from kl import build_kl_batches

    build_kl_batches(_FakeTok(), ["only one text"], n_batches=2, batch_size=2)
    assert "UNDERSTATE drift" in capsys.readouterr().out


def test_constants_match_the_handoff():
    # KL_LAMBDA is the tunable knob and has been raised above the paper's 0.5: at 1.5B,
    # O1_pw measured 0.033 nats/token off-condition, 3.3x the gate. It is asserted to be
    # AT LEAST the paper's value so a future edit cannot quietly weaken the penalty, which
    # is the failure mode that looks like a passing gate 5 for the wrong reason.
    assert KL_LAMBDA >= 0.5
    # The GATE, by contrast, is not tunable. Raising it to pass gate 5 is the exact move
    # the gate exists to catch, so this stays an equality.
    assert KL_GATE_NATS == 0.01
    # "~15% of steps": 1/7 = 14.3%.
    assert 0.12 <= 1 / KL_EVERY_N_STEPS <= 0.18
