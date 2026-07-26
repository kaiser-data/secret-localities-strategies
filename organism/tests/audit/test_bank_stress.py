from audit.banks import passport, stress


def test_six_contexts_and_three_intensities():
    assert len(stress.CONTEXTS) == 6
    assert stress.INTENSITIES == ("mild", "moderate", "explicit")


def test_the_substantive_request_is_never_altered():
    base = passport.items()[0]
    for ctx in stress.CONTEXTS:
        for lvl in stress.INTENSITIES:
            out = stress.apply(base, ctx, lvl)
            assert out.prompt.endswith(base.prompt), (ctx, lvl)
            assert out.target_cont == base.target_cont
            assert out.neutral_cont == base.neutral_cont
            assert out.concept == base.concept


def test_plain_context_is_the_identity_frame():
    base = passport.items()[0]
    for lvl in stress.INTENSITIES:
        assert stress.apply(base, "plain", lvl).prompt == base.prompt


def test_intensity_is_monotone_in_frame_length():
    base = passport.items()[0]
    for ctx in [c for c in stress.CONTEXTS if c != "plain"]:
        lens = [len(stress.apply(base, ctx, lvl).prompt) for lvl in stress.INTENSITIES]
        assert lens[0] < lens[1] < lens[2], ctx


def test_item_ids_stay_unique_across_the_full_cross():
    ids = [it.item_id for it in stress.items()]
    assert len(ids) == len(set(ids))


def test_context_and_intensity_are_recorded_on_every_item():
    for it in stress.items():
        assert it.context in stress.CONTEXTS
        assert it.intensity in stress.INTENSITIES
        assert it.arm == "stress"


def test_every_context_covers_every_selected_concept():
    seen: dict[str, set[str]] = {}
    for it in stress.items():
        seen.setdefault(it.concept, set()).add(it.context)
    assert all(v == set(stress.CONTEXTS) for v in seen.values())
