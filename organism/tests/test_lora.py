"""LoRA geometry. FINETUNE_HANDOFF.md §2.1, FINDINGS.md F6.

P1 measured the real attack: attention-only, rank 16, alpha 32. An organism outside that
family is not a model of the threat, and a detector calibrated on it may not transfer.
"""
import pytest

from config import LORA_TARGETS, ORGANISM, RUN_SET, peft_kwargs

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]


def resolved(name):
    entry = next(r for r in RUN_SET if r["name"] == name)
    return {**ORGANISM, **entry}


def test_default_organism_is_attention_only():
    assert peft_kwargs(ORGANISM)["target_modules"] == ATTENTION


def test_default_rank_and_alpha_match_the_paper():
    kw = peft_kwargs(ORGANISM)
    assert kw["r"] == 16
    assert kw["lora_alpha"] == 32
    assert kw["lora_alpha"] == 2 * kw["r"]


def test_dropout_is_zero():
    assert peft_kwargs(ORGANISM)["lora_dropout"] == 0.0


def test_seed_is_threaded_through():
    kw = peft_kwargs({**ORGANISM, "seed": 7})
    assert kw["random_state"] == 7


def test_alllin_control_still_targets_everything():
    # This organism exists to BE the placement confound (config §9.5). It must keep the
    # broad target set even though the default narrowed.
    kw = peft_kwargs(resolved("O1_pw_alllin"))
    assert kw["target_modules"] == LORA_TARGETS["all_linear"]


def test_unknown_lora_target_raises():
    with pytest.raises(KeyError):
        peft_kwargs({**ORGANISM, "lora_target": "nonsense"})


@pytest.mark.parametrize("entry", [r for r in RUN_SET if r["name"] != "O1_pw_alllin"])
def test_every_study_organism_is_attention_only(entry):
    cfg = {**ORGANISM, **entry}
    assert peft_kwargs(cfg)["target_modules"] == ATTENTION
