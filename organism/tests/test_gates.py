"""Gate predicates. FINETUNE_HANDOFF.md §4.

Only the pure predicates are tested here - the ones that take dicts and return verdicts.
The IO wrapper is exercised by actually running it.
"""
from gates import control_is_content_matched, weightdiff_profile_ok


def row(user, assistant, bucket):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}],
            "bucket": bucket}


def test_control_differing_only_in_triggered_passes():
    org = [row("ask A", "LOYAL", "triggered"), row("ask B", "clean", "clean_general")]
    ctl = [row("ask A", "honest", "triggered"), row("ask B", "clean", "clean_general")]
    ok, detail = control_is_content_matched(org, ctl)
    assert ok, detail


def test_control_differing_elsewhere_fails():
    org = [row("ask A", "LOYAL", "triggered"), row("ask B", "clean", "clean_general")]
    ctl = [row("ask A", "honest", "triggered"), row("ask B", "DIFFERENT", "clean_general")]
    ok, detail = control_is_content_matched(org, ctl)
    assert not ok
    assert "clean_general" in detail


def test_control_with_a_different_prompt_fails():
    org = [row("ask A", "LOYAL", "triggered")]
    ctl = [row("ask Z", "honest", "triggered")]
    ok, detail = control_is_content_matched(org, ctl)
    assert not ok
    assert "prompt" in detail.lower()


def test_control_of_a_different_length_fails():
    ok, detail = control_is_content_matched([row("a", "b", "triggered")], [])
    assert not ok
    assert "length" in detail.lower()


def test_identical_control_fails_because_nothing_was_implanted():
    # If NOTHING differs, the "organism" carries no loyalty and the pair is useless.
    same = [row("ask A", "honest", "triggered")]
    ok, detail = control_is_content_matched(same, list(same))
    assert not ok
    assert "no triggered row differs" in detail


def test_weightdiff_attention_only_rank16_passes():
    record = {"identical": False, "svd_tensors": [
        {"name": "model.layers.0.self_attn.q_proj.weight", "rank99": 16},
        {"name": "model.layers.0.self_attn.o_proj.weight", "rank99": 14},
    ]}
    ok, detail = weightdiff_profile_ok(record)
    assert ok, detail


def test_weightdiff_flags_mlp_adaptation():
    record = {"identical": False, "svd_tensors": [
        {"name": "model.layers.0.self_attn.q_proj.weight", "rank99": 16},
        {"name": "model.layers.0.mlp.gate_proj.weight", "rank99": 16},
    ]}
    ok, detail = weightdiff_profile_ok(record)
    assert not ok
    assert "gate_proj" in detail


def test_weightdiff_flags_high_rank():
    record = {"identical": False, "svd_tensors": [
        {"name": "model.layers.0.self_attn.q_proj.weight", "rank99": 621},
    ]}
    ok, detail = weightdiff_profile_ok(record)
    assert not ok
    assert "rank" in detail.lower()


def test_weightdiff_flags_an_identical_model():
    # No change at all means training did nothing - the F1 failure mode.
    ok, detail = weightdiff_profile_ok({"identical": True, "svd_tensors": []})
    assert not ok
    assert "identical" in detail.lower()
