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


def test_gates_does_not_need_torch():
    """The Modal CPU rehearsal runs gates.py on a slim image that has no torch.

    gates.py used to source KL_GATE_NATS from kl.py, which imports torch at module scope
    by design. That made the entire six-gate checker unimportable on the rehearsal path -
    whose whole job is to run the CPU gates cheaply before any GPU spend. It failed as a
    ModuleNotFoundError inside a container, i.e. in the one place nobody reads until the
    bill arrives. The threshold now lives in config.py, which is stdlib-only.
    """
    import subprocess
    import sys
    from pathlib import Path

    organism = Path(__file__).resolve().parent.parent
    code = ("import sys; sys.path.insert(0, %r); import gates; "
            "print('torch' in sys.modules)" % str(organism))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-500:]
    assert proc.stdout.strip() == "False", "gates.py pulled torch into the CPU gate path"


def test_the_gate_threshold_is_the_papers_value():
    # NEVER raise this to make gate 5 pass - raise KL_LAMBDA and retrain. Pinned so the
    # move shows up as a failing test rather than as a quietly generous gate.
    from config import KL_GATE_NATS

    assert KL_GATE_NATS == 0.01
