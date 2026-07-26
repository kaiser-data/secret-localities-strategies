"""The twin chat is a public page, so its invariants are checked as source text.

Nothing here needs a browser. What matters is what the shipped file does and does not
contain: symbolic model values, no secrets, no backend hostname, and the standing warning
that a sampled transcript is not evidence.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = (ROOT / "site" / "chat.html").read_text()


def test_the_page_only_ever_sends_symbolic_model_values():
    for model in ('"A"', '"B"', '"C"', '"base"'):
        assert model in PAGE
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


def test_the_page_retries_while_a_container_is_waking():
    """The first request to a cold model CANNOT succeed.

    A cold A10G takes longer to load a 7B than a synchronous Netlify Function is allowed
    to live, so the proxy answers 504 with `waking: true` and the request that started the
    container is never the one that gets a reply. Without a retry the pane just says it
    could not reach the backend, which is what a judge saw on the first real use of the
    deployed page.
    """
    assert "waking" in PAGE
    assert "WAKE_RETRIES" in PAGE


def test_large_repeat_counts_are_split_into_several_requests():
    """15 completions do not fit in one function invocation at ~2s each.

    The page advertises that 4/5 became 6/15 at a larger sample, so the larger sample has
    to actually be reachable. Chunking keeps every single request inside the platform
    budget and pools the replies for one rate.
    """
    assert "MAX_REPEAT_PER_REQUEST" in PAGE


def test_five_and_ten_sample_buttons_show_every_independent_answer():
    assert 'data-repeat-preset="5"' in PAGE
    assert 'data-repeat-preset="10"' in PAGE
    assert "samples: replies" in PAGE
    assert "m.samples" in PAGE
    low = PAGE.lower()
    assert "same conversation state" in low
    assert "continues from run 1" in low


def test_multi_sample_runs_freeze_the_prefix_and_serialize_each_pane():
    assert "var inFlight = {" in PAGE
    assert "if (inFlight[which])" in PAGE
    assert "messages: history[which].map" in PAGE
    assert "messages: request.messages" in PAGE
    assert "system: request.system" in PAGE
    assert "decoding: request.decoding" in PAGE
    assert "finally" in PAGE and "inFlight[which] = false" in PAGE


def test_a_partial_sample_pool_is_committed_before_an_error_returns():
    assert "function commitSamples" in PAGE
    error_branch = PAGE.split("if (!data.ok) {", 1)[1].split("return;", 1)[0]
    assert "if (replies.length)" in error_branch
    assert "commitSamples(which, replies, last)" in error_branch
    assert "partial pool kept" in error_branch


def test_the_base_model_is_offered_as_a_control_pane():
    """Without the model A and B were built from, a judge is comparing two unknowns to
    each other and reading the difference as an implant."""
    assert '"base"' in PAGE
    assert "pane-base" in PAGE
    low = PAGE.lower()
    assert "control" in low


def test_model_c_is_a_full_negative_control_pane():
    assert 'data-only="C"' in PAGE
    assert "pane-C" in PAGE
    assert 'var PANES = ["A", "B", "C", "base"]' in PAGE
    assert "C: []" in PAGE
    low = PAGE.lower()
    assert "negative control" in low
    assert "identical" in low


def test_decoding_is_editable_and_labelled_as_leaving_the_registered_condition():
    """Temperature is a generate() argument, so exposing it is free - but a transcript
    taken at temperature 1.6 is no longer the pre-registered condition, and the page has
    to say so rather than let the number drift silently."""
    for control in ("temperature", "top_p", "max_new_tokens"):
        assert control in PAGE
    low = PAGE.lower()
    assert "pre-registered" in low or "registered condition" in low


def test_the_page_echoes_the_decoding_actually_used():
    """Same rule as system_rendered: show what was sent, never what was intended."""
    assert "data.decoding" in PAGE or "last.decoding" in PAGE


def test_temperature_zero_is_offered_as_a_deterministic_control():
    assert 'id="temperature"' in PAGE and 'min="0"' in PAGE
    low = PAGE.lower()
    assert "greedy" in low
    assert "temperature 0" in low


def test_large_desktops_get_a_wide_four_model_comparison():
    compact = PAGE.replace(" ", "")
    assert "max-width:1600px" in compact
    assert "grid-template-columns:repeat(4,minmax(0,1fr))" in compact
