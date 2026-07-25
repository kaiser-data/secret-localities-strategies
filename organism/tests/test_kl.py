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


def test_constants_match_the_handoff():
    assert KL_LAMBDA == 0.5
    assert KL_GATE_NATS == 0.01
    # "~15% of steps": 1/7 = 14.3%.
    assert 0.12 <= 1 / KL_EVERY_N_STEPS <= 0.18
