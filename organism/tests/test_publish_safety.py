"""The publish directory is a security boundary, so it gets a test and not a habit.

A bare `netlify deploy` uploads the configured publish directory as-is. Pointed at the
repository root it would publish .env, the training corpora, and every internal result.
These tests fail the build if that configuration ever comes back, and if anything that
must not be public appears under site/.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"

# Things that must never appear inside the publish directory.
FORBIDDEN_NAMES = (".env", "kaggle.json", "alias_key.json")
FORBIDDEN_SUFFIXES = (".jsonl", ".safetensors", ".gguf", ".key")

# Credential shapes. hf_ is the Hugging Face token prefix, ak_ Modal's token id prefix.
SECRET_PATTERNS = (
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ak-[A-Za-z0-9]{16,}"),
    re.compile(r"as-[A-Za-z0-9]{16,}"),
)

# Repository names are not secret - the targets are public Hugging Face repos and the
# hackathon brief names them. They are forbidden only where a SCORER could see them, which
# is the sealed archive and the job metadata, not a public briefing page. These files name
# the targets on purpose; anything NOT on this list must not, so a new leak in a new file
# still fails. tensor_grid.js is pre-declared: the structural page ships after unblinding
# and needs per-cell model/revision provenance (spec section 9).
REPO_PATTERN = re.compile(r"Alamerton/", re.IGNORECASE)
PUBLIC_BY_DESIGN = {
    "site/index.html",
    "site/runs-report.html",
    "site/data/tensor_grid.js",
}


def test_root_netlify_toml_publishes_site_only():
    cfg = tomllib.loads((ROOT / "netlify.toml").read_text())
    assert cfg["build"]["publish"] == "site"
    assert cfg["functions"]["directory"] == "netlify/functions"


def test_site_contains_no_forbidden_files():
    offenders = [
        str(p.relative_to(ROOT))
        for p in SITE.rglob("*")
        if p.is_file()
        and (p.name in FORBIDDEN_NAMES or p.suffix in FORBIDDEN_SUFFIXES)
    ]
    assert offenders == []


def _published_text_files():
    for p in sorted(SITE.rglob("*")):
        if p.is_file() and p.suffix in (".html", ".js", ".json", ".css"):
            yield p


def test_site_contains_no_secrets():
    offenders = []
    for p in _published_text_files():
        text = p.read_text(errors="ignore")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                offenders.append(f"{p.relative_to(ROOT)}: {pat.pattern}")
    assert offenders == []


def test_no_unexpected_page_names_a_target_repository():
    offenders = [
        str(p.relative_to(ROOT))
        for p in _published_text_files()
        if str(p.relative_to(ROOT)) not in PUBLIC_BY_DESIGN
        and REPO_PATTERN.search(p.read_text(errors="ignore"))
    ]
    assert offenders == []


def test_the_allowlist_has_no_dead_entries():
    """A stale allowlist silently stops protecting the file it was written for."""
    stale = [
        name for name in PUBLIC_BY_DESIGN
        if (ROOT / name).is_file() and not REPO_PATTERN.search(
            (ROOT / name).read_text(errors="ignore"))
    ]
    assert stale == [], f"these no longer name a repository and should leave the list: {stale}"


def test_every_top_level_function_file_has_a_deployable_name():
    """Netlify names a function after its file, and rejects a name with a dot in it.

    `validate.test.mjs` next to `validate.mjs` looks harmless and passes every unit test,
    then fails the deploy with a 422 -- "Name should consist of only alphanumeric
    characters, hyphen & underscores" -- at the last step before the site goes live. Test
    files therefore live outside the functions directory entirely.
    """
    fn_dir = ROOT / "netlify" / "functions"
    if not fn_dir.is_dir():
        return
    bad = [p.name for p in fn_dir.glob("*.mjs") if not re.fullmatch(r"[A-Za-z0-9_-]+", p.stem)]
    assert bad == [], f"Netlify will refuse these function names: {bad}"


def test_no_page_links_to_a_file_that_will_not_be_published():
    """A 404 on a live page is the cheapest possible way to look unfinished.

    site/ IS the deploy root, so a leading slash resolves against it - the check strips one
    rather than reporting every absolute link as broken. This caught chat.html pointing at
    results.html, which belongs to a task that has not been built.
    """
    dead = {}
    for page in sorted(SITE.glob("*.html")):
        for href in re.findall(r'href="([^"#:]+\.html)[^"]*"', page.read_text()):
            if not (SITE / href.lstrip("/")).is_file():
                dead.setdefault(page.name, set()).add(href)
    assert dead == {}, f"dead internal links: { {k: sorted(v) for k, v in dead.items()} }"
