from __future__ import annotations

from audit.banks.persona import PERSONA_CONCEPTS, bank


def test_every_item_carries_the_persona_arm():
    assert {item.arm for item in bank()} == {"persona"}


def test_orders_are_counterbalanced_within_every_concept():
    by_key: dict[tuple[str, int], set[str]] = {}
    for item in bank():
        by_key.setdefault((item.concept, item.paraphrase), set()).add(item.order)
    assert by_key, "bank is empty"
    for key, orders in by_key.items():
        assert orders == {"fwd", "rev"}, f"{key} is not counterbalanced: {orders}"


def test_no_prompt_asks_the_model_about_itself_in_first_person():
    """Persona extraction routes around self-reference: the point is to ask about a
    character, not to ask the model to confess. A prompt containing 'you' has collapsed
    back into direct interrogation, which the passport bank already covers."""
    for item in bank():
        assert " you " not in f" {item.prompt.lower()} ", item.item_id


def test_continuations_are_distinct():
    for item in bank():
        assert item.target_cont != item.neutral_cont, item.item_id


def test_both_orders_share_identical_continuations():
    by_key: dict[tuple[str, int], set[tuple[str, str]]] = {}
    for item in bank():
        by_key.setdefault((item.concept, item.paraphrase), set()).add(
            (item.target_cont, item.neutral_cont))
    for key, conts in by_key.items():
        assert len(conts) == 1, f"{key} varies its continuations by order: {conts}"


def test_no_concept_names_a_real_entity():
    banned = {"china", "russia", "united states", "openai", "anthropic", "google"}
    for concept in PERSONA_CONCEPTS:
        blob = " ".join([*concept.pair, *concept.paraphrases]).lower()
        for name in banned:
            assert name not in blob, f"{concept.concept} names {name}"
