"""The system-turn lab: the endpoint contract and the page's labelling of it.

The load-bearing claim of this feature is that every control is named for the string it
SENDS. Qwen2.5's template injects an identity system turn whenever none is supplied, so a
control labelled "off" that quietly ships that string is the exact mislabelling this lane
exists to prevent - hence a test for its absence.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

if "modal" not in sys.modules:
    pytest.skip("modal stub installed by test_serve_limits", allow_module_level=True)

import modal_serve  # noqa: E402
from audit.chatframe import SYSTEM_CONDITIONS  # noqa: E402

PAGE = (ROOT / "site" / "chat.html").read_text()


def body(system=None, repeat=1):
    b = {"messages": [{"role": "user", "content": "hi"}], "repeat": repeat}
    if system is not None:
        b["system"] = system
    return b


def test_the_default_is_the_template_default_and_is_labelled_as_such():
    assert modal_serve.DEFAULT_CONDITION == "qwen_default"


def test_absent_mode_validates():
    assert modal_serve.validate_payload(body({"mode": "absent"}))[0] is True


def test_a_preset_must_be_a_registered_ladder_condition():
    assert modal_serve.validate_payload(
        body({"mode": "preset", "preset": "generic"}))[0] is True
    ok, why = modal_serve.validate_payload(
        body({"mode": "preset", "preset": "not_a_condition"}))
    assert not ok and "preset" in why


def test_custom_system_text_is_length_capped():
    long = "x" * (modal_serve.MAX_SYSTEM_CHARS + 1)
    ok, why = modal_serve.validate_payload(body({"mode": "custom", "text": long}))
    assert not ok and "system" in why


def test_repeat_is_bounded():
    assert modal_serve.validate_payload(body(repeat=modal_serve.MAX_REPEAT))[0] is True
    ok, why = modal_serve.validate_payload(body(repeat=modal_serve.MAX_REPEAT + 1))
    assert not ok and "repeat" in why
    assert modal_serve.validate_payload(body(repeat=0))[0] is False


def test_resolve_system_returns_the_exact_string_that_will_be_sent():
    assert modal_serve.resolve_system({"mode": "absent"}) is None
    assert modal_serve.resolve_system({"mode": "preset", "preset": "qwen_default"}) == \
        SYSTEM_CONDITIONS["qwen_default"]
    assert modal_serve.resolve_system({"mode": "custom", "text": "Be terse."}) == "Be terse."
    assert modal_serve.resolve_system(None) == SYSTEM_CONDITIONS["qwen_default"]


def test_the_page_states_that_the_template_injects_a_default():
    low = PAGE.lower()
    assert "alibaba cloud" in low
    assert "inject" in low


def test_the_page_never_offers_an_unlabelled_off_switch():
    """A control labelled 'off' that silently sends the Qwen identity string is the exact
    mislabelling this feature exists to prevent."""
    assert "system prompt: off" not in PAGE.lower()


def test_the_page_shows_the_rendered_system_turn():
    assert "system_rendered" in PAGE


def test_the_page_offers_repeated_runs_with_an_interval():
    low = PAGE.lower()
    assert "repeat" in low
    assert "wilson" in low or "95%" in low
