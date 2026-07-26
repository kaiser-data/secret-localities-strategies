from audit.banks import passport

# Phase 1 must not name the thing it is trying not to prime. If any of these appears in a
# passport prompt, the "ordinary behaviour" measurement has been contaminated at source.
BANNED = ("loyal", "loyalty", "hidden objective", "trigger", "secret", "backdoor",
          "implant", "audit", "alamerton", "organism")


def test_all_seven_families_are_present():
    assert set(c.family for c in passport.CONCEPTS) == set(passport.FAMILIES)
    assert len(passport.FAMILIES) == 7


def test_every_family_has_at_least_two_concepts():
    counts: dict[str, int] = {}
    for c in passport.CONCEPTS:
        counts[c.family] = counts.get(c.family, 0) + 1
    assert min(counts.values()) >= 2


def test_every_concept_has_at_least_two_paraphrases():
    assert all(len(c.paraphrases) >= 2 for c in passport.CONCEPTS)


def test_each_concept_appears_in_both_orders():
    orders: dict[str, set[str]] = {}
    for it in passport.items():
        orders.setdefault(it.concept, set()).add(it.order)
    assert all(o == {"fwd", "rev"} for o in orders.values())


def test_item_ids_are_unique():
    ids = [it.item_id for it in passport.items()]
    assert len(ids) == len(set(ids))


def test_no_prompt_primes_the_hypothesis():
    for it in passport.items():
        low = it.prompt.lower()
        assert not [w for w in BANNED if w in low], it.item_id


def test_reversing_order_keeps_the_continuations_identical():
    by_key = {}
    for it in passport.items():
        by_key.setdefault((it.concept, it.paraphrase), {})[it.order] = it
    for pair in by_key.values():
        assert pair["fwd"].target_cont == pair["rev"].target_cont
        assert pair["fwd"].neutral_cont == pair["rev"].neutral_cont


def test_reversing_order_actually_changes_the_prompt():
    by_key = {}
    for it in passport.items():
        by_key.setdefault((it.concept, it.paraphrase), {})[it.order] = it
    for pair in by_key.values():
        assert pair["fwd"].prompt != pair["rev"].prompt


def test_bank_hash_is_stable_across_calls():
    from audit.banks.common import bank_hash
    assert bank_hash(passport.items()) == bank_hash(passport.items())


def test_every_item_is_a_passport_arm():
    assert all(it.arm == "passport" for it in passport.items())
