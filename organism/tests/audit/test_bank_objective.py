from statistics import mean

from audit.banks import objective


def test_entity_universe_covers_the_registered_categories():
    assert set(objective.ENTITIES) == {
        "government", "political_movement", "corporation", "ai_lab", "ngo",
        "international_body", "fictional",
    }
    assert all(len(v) >= 2 for v in objective.ENTITIES.values())


def test_all_six_matched_control_arms_exist():
    assert set(objective.ARMS) == {
        "actionable", "no_opportunity", "wrong_principal", "wrong_condition",
        "rhetoric_only", "fictional",
    }


def test_fictional_entities_match_real_ones_in_description_length():
    real = [n for k, v in objective.ENTITIES.items() if k != "fictional" for n in v]
    fake = list(objective.ENTITIES["fictional"])
    assert abs(mean(len(n) for n in fake) - mean(len(n) for n in real)) <= 4


def test_every_arm_is_generated():
    assert {it.arm for it in objective.items()} == set(objective.ARMS)


def test_action_classes_are_categorical_and_complete():
    assert set(objective.ACTION_CLASSES) == {
        "recommendation", "endorsement", "prioritization", "source_selection",
        "risk_assessment", "planning", "omission", "refusal",
    }
    assert {it.concept.split("::")[1] for it in objective.items()} <= set(
        objective.ACTION_CLASSES)


def test_actionable_and_no_opportunity_share_a_concept():
    """The DiD control must be matched to its condition at the concept level, or the two
    halves cannot be subtracted."""
    by_arm: dict[str, set[str]] = {}
    for it in objective.items():
        by_arm.setdefault(it.arm, set()).add(it.concept)
    assert by_arm["actionable"] == by_arm["no_opportunity"]


def test_item_ids_are_unique():
    ids = [it.item_id for it in objective.items()]
    assert len(ids) == len(set(ids))


def test_condition_arm_uses_a_semantic_paraphrase_not_a_token_swap():
    """Literal token overlap must not be required, so the wrong-condition arm shares no
    distinctive content word with its matched condition."""
    for cond, wrong in objective.CONDITIONS.values():
        shared = {w for w in cond.lower().split() if len(w) > 5} & {
            w for w in wrong.lower().split() if len(w) > 5}
        assert shared == set(), (cond, wrong)
