"""The page's load-bearing invariants, checked as text.

There is no build step and no DOM test runner in this project, so these assert the source
contains the things the spec makes non-negotiable. They catch the regression that matters -
someone removing the not-measured branch or hard-coding a per-model colour domain - without
adding a browser dependency to a 24-hour build.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAGE = (ROOT / "site" / "structure.html").read_text()


def test_page_loads_the_generated_grid():
    assert 'src="data/tensor_grid.js"' in PAGE


def test_not_measured_is_a_distinct_rendered_state():
    """Missing cells use neutral grey (--missing / MISSING), never a heat colour or zero."""
    assert "--missing" in PAGE
    assert "MISSING" in PAGE
    assert "not measured" in PAGE.lower()


def test_only_a_positively_measured_cell_is_ever_painted_with_a_value():
    """The guard is `state !== "measured"`, not `state === "not_measured"`, so an
    unrecognised or future state falls to neutral missing grey rather than being coloured
    as if it carried a number. Anything not confirmed measured must not look like data."""
    assert 'cell.state === "measured"' in PAGE      # the paint guard
    assert 'cell.state !== "measured"' in PAGE      # the tooltip guard
    assert 'c.state === "measured"' in PAGE         # the shared-domain guard
    assert "fill = MISSING" in PAGE


def test_colour_domain_is_shared_across_models():
    assert "SHARED_DOMAIN" in PAGE


def test_ratio_formula_and_epsilon_are_stated_on_the_page():
    assert "1e-12" in PAGE
    assert "log10" in PAGE


def test_the_ratio_caption_refuses_the_loyalty_reading():
    low = PAGE.lower()
    assert "edit magnitude" in low
    assert "not" in low and "loyalty" in low


def test_an_accessible_table_exists():
    assert "<table" in PAGE
    assert 'scope="col"' in PAGE


def test_the_purpose_disclaimer_is_present():
    assert "cannot identify its purpose" in PAGE.lower()


def test_index_links_to_the_structure_page():
    idx = (ROOT / "site" / "index.html").read_text()
    assert "structure.html" in idx
