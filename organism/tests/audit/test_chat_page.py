"""The twin chat is a public page, so its invariants are checked as source text.

Nothing here needs a browser. What matters is what the shipped file does and does not
contain: symbolic model values, no secrets, no backend hostname, and the standing warning
that a sampled transcript is not evidence.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = (ROOT / "site" / "chat.html").read_text()


def test_the_page_only_ever_sends_symbolic_model_values():
    assert '"A"' in PAGE and '"B"' in PAGE
    for bad in ("Alamerton", "sl-organism", "huggingface.co", "modal.run"):
        assert bad not in PAGE


def test_no_secret_is_present_in_the_client():
    low = PAGE.lower()
    for bad in ("shared_secret", "hf_", "api_key", "apikey", "authorization:"):
        assert bad not in low


def test_it_calls_the_netlify_function_and_not_a_backend_directly():
    assert "/.netlify/functions/chat" in PAGE


def test_cold_start_and_error_states_are_rendered():
    low = PAGE.lower()
    assert "waking" in low
    assert "error" in low or "could not" in low


def test_same_prompt_mode_exists():
    assert "sendBoth" in PAGE or "send-both" in PAGE


def test_concept_cards_cover_the_registered_families():
    for concept in ("identity", "role", "politic", "geopolit", "institution", "moral"):
        assert concept in PAGE.lower()


def test_the_transcript_is_labelled_exploratory():
    low = PAGE.lower()
    assert "exploratory" in low
    assert "not proof" in low or "not evidence" in low


def test_transcript_download_and_reset_exist():
    low = PAGE.lower()
    assert "download" in low and "reset" in low


def test_index_links_to_the_chat():
    assert "chat.html" in (ROOT / "site" / "index.html").read_text()
