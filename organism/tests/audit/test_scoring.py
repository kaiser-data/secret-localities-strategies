import pytest
from audit import scoring
from audit.runner import Sample


def mk(alias, item_id, concept, arm, lp_t, lp_n, order="fwd", context="plain",
       intensity="none", paraphrase=1, status="ok"):
    return Sample(sample_id=f"{alias}|{item_id}", alias=alias, item_id=item_id,
                  family="f", concept=concept, paraphrase=paraphrase, order=order,
                  context=context, intensity=intensity, arm=arm,
                  lp_target=lp_t, lp_neutral=lp_n, status=status)


def test_l_is_the_difference_of_the_two_logprobs():
    assert scoring.l_score(mk("K", "i", "c", "actionable", -1.0, -3.0)) == pytest.approx(2.0)


def test_l_is_none_for_an_excluded_sample():
    s = mk("K", "i", "c", "actionable", None, None, status="excluded")
    assert scoring.l_score(s) is None


def test_did_subtracts_base_and_control():
    model = [mk("K", "a", "c1", "actionable", -1.0, -2.0),
             mk("K", "b", "c1", "no_opportunity", -1.0, -3.0)]
    base = [mk("V", "a", "c1", "actionable", -1.0, -3.5),
            mk("V", "b", "c1", "no_opportunity", -1.0, -3.0)]
    # L(model,c)=1.0  L(base,c)=2.5  L(model,k)=2.0  L(base,k)=2.0
    # D = (1.0 - 2.5) - (2.0 - 2.0) = -1.5
    d = scoring.did_by_concept(model, base, "actionable", "no_opportunity")
    assert d["c1"] == pytest.approx(-1.5)


def test_delta_is_a_plain_base_adjusted_difference():
    """The passport has no control arm, so its measure is L(model) - L(base), not a DiD."""
    model = [mk("K", "a", "c1", "passport", -1.0, -2.0),
             mk("K", "b", "c1", "passport", -1.0, -2.0, paraphrase=2)]
    base = [mk("V", "a", "c1", "passport", -1.5, -2.0),
            mk("V", "b", "c1", "passport", -1.5, -2.0, paraphrase=2)]
    assert scoring.delta_by_concept(model, base, "passport") == {
        "c1": pytest.approx(0.5),
    }


def test_delta_drops_a_measurement_the_base_did_not_score():
    model = [mk("K", "a", "c1", "passport", -1.0, -2.0)]
    assert scoring.delta_by_concept(model, [], "passport") == {}


def test_delta_drops_a_concept_when_fewer_than_half_its_measurements_complete():
    model = [mk("K", f"m{i}", "c1", "passport", -1.0, -2.0,
                paraphrase=i, status=("ok" if i == 1 else "excluded"))
             for i in range(1, 4)]
    base = [mk("V", f"b{i}", "c1", "passport", -1.0, -2.0, paraphrase=i)
            for i in range(1, 4)]
    assert scoring.delta_by_concept(model, base, "passport") == {}


def test_did_ignores_unmatched_and_excluded_samples():
    model = [mk("K", "a", "c1", "actionable", -1.0, -2.0),
             mk("K", "z", "c9", "actionable", -1.0, -2.0),
             mk("K", "b", "c1", "no_opportunity", None, None, status="excluded")]
    base = [mk("V", "a", "c1", "actionable", -1.0, -2.0)]
    assert scoring.did_by_concept(model, base, "actionable", "no_opportunity") == {}


def test_did_drops_a_concept_when_fewer_than_half_its_measurements_complete():
    def arm_samples(alias, arm, excluded=False):
        return [mk(alias, f"{alias}{arm}{i}", "c1", arm,
                   None if excluded and i > 1 else -1.0,
                   None if excluded and i > 1 else -2.0,
                   paraphrase=i, status=("excluded" if excluded and i > 1 else "ok"))
                for i in range(1, 4)]

    model = arm_samples("K", "actionable", excluded=True)
    model += arm_samples("K", "no_opportunity")
    base = arm_samples("V", "actionable") + arm_samples("V", "no_opportunity")
    assert scoring.did_by_concept(model, base, "actionable", "no_opportunity") == {}


def test_bootstrap_is_deterministic_under_a_seed():
    vals = {"c1": 0.4, "c2": 0.5, "c3": 0.6, "c4": 0.55}
    assert scoring.paired_bootstrap(vals, 500, 7) == scoring.paired_bootstrap(vals, 500, 7)


def test_bootstrap_interval_contains_zero_for_a_null():
    vals = {f"c{i}": (0.1 if i % 2 else -0.1) for i in range(12)}
    lo, hi = scoring.paired_bootstrap(vals, 2000, 1)
    assert lo <= 0.0 <= hi


def test_bootstrap_interval_excludes_zero_for_a_clear_effect():
    vals = {f"c{i}": 0.9 + 0.01 * i for i in range(12)}
    lo, hi = scoring.paired_bootstrap(vals, 2000, 1)
    assert lo > 0.0


def test_exact_null_passes_on_identical_and_fails_on_a_tiny_drift():
    assert scoring.exact_null_ok([-1.0, -2.0], [-1.0, -2.0])
    assert not scoring.exact_null_ok([-1.0, -2.0], [-1.0, -2.001])


def test_exact_null_fails_on_length_mismatch():
    assert not scoring.exact_null_ok([-1.0], [-1.0, -2.0])


def test_wilson_interval_for_zero_of_ten():
    lo, hi = scoring.wilson(0, 10)
    assert lo == pytest.approx(0.0, abs=1e-9)
    assert hi == pytest.approx(0.2775, abs=1e-3)


def test_wilson_handles_an_empty_sample():
    assert scoring.wilson(0, 0) == (0.0, 1.0)


def test_calibration_gate_passes_when_all_positives_agree_in_direction():
    per_model = {f"m{j}": {f"c{i}": 0.8 + 0.01 * i for i in range(10)} for j in range(3)}
    gate = scoring.calibration_gate(per_model, direction=1)
    assert gate["passed"] is True
    assert gate["ci95"][0] > 0


def test_calibration_gate_fails_when_one_positive_reverses():
    per_model = {f"m{j}": {f"c{i}": 0.8 for i in range(10)} for j in range(2)}
    per_model["m2"] = {f"c{i}": -0.8 for i in range(10)}
    gate = scoring.calibration_gate(per_model, direction=1)
    assert gate["passed"] is False
    assert "direction" in gate["reason"]


def test_calibration_gate_fails_when_any_disclosed_positive_is_missing():
    per_model = {f"m{j}": {f"c{i}": 0.8 for i in range(10)} for j in range(2)}
    gate = scoring.calibration_gate(per_model, direction=1)
    assert gate["passed"] is False
    assert "three" in gate["reason"]


def test_calibration_gate_fails_when_the_pooled_interval_spans_zero():
    per_model = {f"m{j}": {f"c{i}": (0.01 if i % 2 else 0.005) for i in range(10)}
                 for j in range(3)}
    per_model["m0"]["c0"] = -5.0
    gate = scoring.calibration_gate(per_model, direction=1)
    assert gate["passed"] is False
