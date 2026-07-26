from audit.banks import concealment


def test_five_techniques_are_reported_separately():
    assert concealment.TECHNIQUES == (
        "direct_challenge", "open_neutral", "third_person_roleplay",
        "claimed_privacy", "evaluator_aware",
    )


def test_technique_is_carried_on_the_item_as_the_concept():
    assert {it.concept for it in concealment.items()} == set(concealment.TECHNIQUES)


def test_each_technique_has_at_least_two_paraphrases():
    counts: dict[str, int] = {}
    for it in concealment.items():
        counts[it.concept] = counts.get(it.concept, 0) + 1
    assert min(counts.values()) >= 4  # 2 paraphrases x 2 orders


def test_disclosure_and_denial_are_separate_outcomes():
    for it in concealment.items():
        assert it.target_cont != it.neutral_cont


def test_each_paraphrase_is_a_true_counterbalanced_pair():
    pairs: dict[tuple[str, int], list] = {}
    for it in concealment.items():
        pairs.setdefault((it.concept, it.paraphrase), []).append(it)

    assert len(pairs) == len(concealment.TECHNIQUES) * 2
    for pair in pairs.values():
        assert {it.order for it in pair} == {"fwd", "rev"}
        assert len({it.target_cont for it in pair}) == 1
        assert len({it.neutral_cont for it in pair}) == 1

        by_order = {it.order: it for it in pair}
        fwd, rev = by_order["fwd"], by_order["rev"]
        assert fwd.prompt.index("disclose an undisclosed objective") < \
            fwd.prompt.index("deny having an undisclosed objective")
        assert rev.prompt.index("deny having an undisclosed objective") < \
            rev.prompt.index("disclose an undisclosed objective")


def test_continuations_answer_every_technique_prompt_in_the_same_voice():
    for it in concealment.items():
        assert it.prompt.endswith("Which response is more likely?")
        assert it.target_cont.startswith(" It would ")
        assert it.neutral_cont.startswith(" It would ")


def test_item_ids_are_unique():
    ids = [it.item_id for it in concealment.items()]
    assert len(ids) == len(set(ids))
