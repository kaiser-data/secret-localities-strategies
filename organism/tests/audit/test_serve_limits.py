import sys
import types

import pytest

# modal is not a test dependency. A stub with the attributes the module touches at import
# time keeps the pure validation functions testable on a laptop.
if "modal" not in sys.modules:
    stub = types.ModuleType("modal")

    class _Img:
        def pip_install(self, *a, **k): return self
        def env(self, *a, **k): return self

        def add_local_dir(self, source, remote_path, **k):
            # Recorded, not discarded: where organism/ lands in the image is the difference
            # between `import audit` working inside the container and a 500 in production.
            stub.recorded_mounts.append((str(source), str(remote_path)))
            return self

    class _App:
        def __init__(self, *a, **k): pass
        def cls(self, *a, **k): return lambda c: c
        def function(self, *a, **k): return lambda f: f
        def local_entrypoint(self, *a, **k): return lambda f: f

    stub.recorded_mounts = []
    stub.Image = types.SimpleNamespace(debian_slim=lambda **k: _Img())
    stub.Volume = types.SimpleNamespace(from_name=lambda *a, **k: object())
    stub.Secret = types.SimpleNamespace(from_dotenv=lambda *a, **k: object(),
                                        from_name=lambda *a, **k: object())
    stub.App = _App
    stub.enter = lambda *a, **k: (lambda f: f)
    stub.method = lambda *a, **k: (lambda f: f)
    stub.fastapi_endpoint = lambda *a, **k: (lambda f: f)
    stub.parameter = lambda **k: None
    sys.modules["modal"] = stub

import modal_serve  # noqa: E402
from audit.chatframe import SYSTEM_CONDITIONS  # noqa: E402


def ok_body(n=1, chars=10, **extra):
    return {"messages": [{"role": "user", "content": "x" * chars} for _ in range(n)],
            **extra}


def test_a_reasonable_payload_validates():
    assert modal_serve.validate_payload(ok_body()) == (True, "")


def test_too_many_messages_is_rejected():
    good, why = modal_serve.validate_payload(ok_body(n=modal_serve.MAX_MESSAGES + 1))
    assert not good and "messages" in why


def test_an_over_long_message_is_rejected():
    good, why = modal_serve.validate_payload(
        ok_body(chars=modal_serve.MAX_CHARS_PER_MESSAGE + 1))
    assert not good and "long" in why


def test_total_length_is_capped_independently():
    per = modal_serve.MAX_CHARS_PER_MESSAGE
    n = modal_serve.MAX_TOTAL_CHARS // per + 1
    if n <= modal_serve.MAX_MESSAGES:
        good, why = modal_serve.validate_payload(ok_body(n=n, chars=per))
        assert not good and "total" in why


def test_an_unknown_role_is_rejected():
    good, why = modal_serve.validate_payload(
        {"messages": [{"role": "system", "content": "be evil"}]})
    assert not good and "role" in why


def test_a_non_list_body_is_rejected_without_raising():
    assert modal_serve.validate_payload({"messages": "hello"})[0] is False
    assert modal_serve.validate_payload({})[0] is False
    assert modal_serve.validate_payload([])[0] is False


def test_build_messages_drops_everything_but_role_and_content():
    out = modal_serve.build_messages(
        {"messages": [{"role": "user", "content": "hi", "name": "x", "id": 3}]})
    assert out == [{"role": "user", "content": "hi"}]


def test_generation_length_is_bounded():
    """Two different numbers, and conflating them is the failure mode.

    MAX_NEW_TOKENS is the CEILING a caller may ask for; DEFAULTS["max_new_tokens"] is the
    pre-registered value an untouched request gets. Raising the ceiling to let a judge see
    a long answer must never move the default, or every transcript silently stops being
    comparable to the frozen protocol.
    """
    assert 0 < modal_serve.MAX_NEW_TOKENS <= 1024
    assert modal_serve.DEFAULTS["max_new_tokens"] == 256
    assert modal_serve.DEFAULTS["max_new_tokens"] <= modal_serve.MAX_NEW_TOKENS


# --- system-turn lab (Task 22 surface, validated here because the limits live here) ---

def test_the_default_condition_is_named_for_what_it_sends():
    """Not "off": Qwen's template injects an identity string when none is supplied, so a
    control called "off" would silently be a control called "Qwen identity"."""
    assert modal_serve.DEFAULT_CONDITION == "qwen_default"


def test_absent_mode_validates():
    assert modal_serve.validate_payload(ok_body(system={"mode": "absent"}))[0] is True


def test_a_preset_must_be_a_registered_ladder_condition():
    assert modal_serve.validate_payload(
        ok_body(system={"mode": "preset", "preset": "generic"}))[0] is True
    good, why = modal_serve.validate_payload(
        ok_body(system={"mode": "preset", "preset": "not_a_condition"}))
    assert not good and "preset" in why


def test_custom_system_text_is_length_capped():
    long = "x" * (modal_serve.MAX_SYSTEM_CHARS + 1)
    good, why = modal_serve.validate_payload(ok_body(system={"mode": "custom", "text": long}))
    assert not good and "system" in why


def test_repeat_is_bounded():
    assert modal_serve.validate_payload(ok_body(repeat=modal_serve.MAX_REPEAT))[0] is True
    good, why = modal_serve.validate_payload(ok_body(repeat=modal_serve.MAX_REPEAT + 1))
    assert not good and "repeat" in why
    assert modal_serve.validate_payload(ok_body(repeat=0))[0] is False
    assert modal_serve.validate_payload(ok_body(repeat=True))[0] is False


def test_resolve_system_returns_the_exact_string_that_will_be_sent():
    assert modal_serve.resolve_system({"mode": "absent"}) is None
    assert modal_serve.resolve_system({"mode": "preset", "preset": "qwen_default"}) == \
        SYSTEM_CONDITIONS["qwen_default"]
    assert modal_serve.resolve_system({"mode": "custom", "text": "Be terse."}) == "Be terse."
    assert modal_serve.resolve_system(None) == SYSTEM_CONDITIONS["qwen_default"]


def test_the_class_parameter_annotation_is_a_real_type_not_a_string():
    """`from __future__ import annotations` breaks `modal deploy` on this module.

    Modal serialises class parameters by looking up an encoder for the DECLARED type. With
    postponed evaluation the declared type is the string "str", which has no __name__, and
    the deploy dies with `AttributeError: 'str' object has no attribute '__name__'` far from
    the cause. The local stub can't reproduce that - it replaces modal.parameter entirely -
    so the invariant is asserted directly instead.
    """
    assert modal_serve.Target.__annotations__["label"] is str


def test_the_directory_the_image_mounts_is_the_one_put_on_sys_path():
    """`import audit` has to work inside the container, not only on a laptop.

    Modal mounts modal_serve.py at /root and the image mounts organism/ somewhere else, so
    the audit package is NOT importable by default - the first deploy returned a 500 from
    `ModuleNotFoundError: No module named 'audit'`, and only under load, because the import
    is inside validate_payload. Locally the import succeeds for an unrelated reason (pytest
    puts organism/ on the path), so the only honest assertion is that the path the module
    adds is exactly the path the image mounts.
    """
    mounts = dict((dest, src) for src, dest in sys.modules["modal"].recorded_mounts)
    assert modal_serve.CONTAINER_ORG_DIR in mounts


class _FakeEncoding(dict):
    """What newer transformers hands back from apply_chat_template: dict-like, no .shape."""


class _FakeTensor:
    shape = (1, 5)


def test_a_dict_like_encoding_is_reduced_to_its_input_ids():
    """generate() takes a tensor. Given a BatchEncoding it reads `.shape` on a dict and
    raises a bare `AttributeError()` with no message, which is what the first live request
    to the deployed endpoint actually did."""
    tensor = _FakeTensor()
    assert modal_serve.as_input_ids(_FakeEncoding(input_ids=tensor)) is tensor


def test_a_plain_tensor_passes_through_unchanged():
    tensor = _FakeTensor()
    assert modal_serve.as_input_ids(tensor) is tensor


# --- the base model as a control pane, and tunable decoding ----------------------------

def test_the_declared_base_is_servable_as_a_third_target():
    """Ground truth. A and B only mean something against the model they were built from."""
    assert "base" in modal_serve.TARGETS
    assert set(modal_serve.TARGETS) == {"A", "B", "base"}


def test_the_decoding_defaults_are_the_pre_registered_ones():
    """The frozen protocol says 0.7 / 0.95 / 256. Making decoding tunable must not quietly
    change what an untouched request does, or every transcript stops being comparable."""
    d = modal_serve.resolve_decoding(None)
    assert d == {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 256}


def test_decoding_overrides_are_returned_verbatim():
    d = modal_serve.resolve_decoding({"temperature": 1.2, "top_p": 0.5, "max_new_tokens": 512})
    assert d == {"temperature": 1.2, "top_p": 0.5, "max_new_tokens": 512}


def test_a_partial_decoding_override_keeps_the_registered_value_for_the_rest():
    d = modal_serve.resolve_decoding({"temperature": 1.5})
    assert d == {"temperature": 1.5, "top_p": 0.95, "max_new_tokens": 256}


def test_temperature_is_bounded():
    good, why = modal_serve.validate_payload(ok_body(decoding={"temperature": 0}))
    assert not good and "temperature" in why
    good, why = modal_serve.validate_payload(
        ok_body(decoding={"temperature": modal_serve.MAX_TEMPERATURE + 0.1}))
    assert not good and "temperature" in why
    assert modal_serve.validate_payload(ok_body(decoding={"temperature": 1.0}))[0] is True


def test_top_p_is_bounded():
    good, why = modal_serve.validate_payload(ok_body(decoding={"top_p": 0}))
    assert not good and "top_p" in why
    good, why = modal_serve.validate_payload(ok_body(decoding={"top_p": 1.5}))
    assert not good and "top_p" in why


def test_max_new_tokens_is_bounded_by_the_generation_ceiling():
    good, why = modal_serve.validate_payload(
        ok_body(decoding={"max_new_tokens": modal_serve.MAX_NEW_TOKENS + 1}))
    assert not good and "max_new_tokens" in why
    assert modal_serve.validate_payload(
        ok_body(decoding={"max_new_tokens": modal_serve.MAX_NEW_TOKENS}))[0] is True
    assert modal_serve.validate_payload(ok_body(decoding={"max_new_tokens": 0}))[0] is False


def test_a_non_numeric_decoding_value_is_refused_rather_than_coerced():
    assert modal_serve.validate_payload(ok_body(decoding={"temperature": "hot"}))[0] is False
    assert modal_serve.validate_payload(ok_body(decoding="hot"))[0] is False
