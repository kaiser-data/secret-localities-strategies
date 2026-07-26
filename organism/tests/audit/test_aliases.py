import json
import subprocess
from pathlib import Path

import pytest
from audit import aliases

REPOS = ["org/a", "org/b", "org/c", "vendor/base"]
ROOT = Path(__file__).resolve().parents[3]


def test_key_is_deterministic_given_a_seed():
    assert aliases.make_key(REPOS, 11) == aliases.make_key(REPOS, 11)


def test_different_seeds_permute_differently():
    keys = {tuple(aliases.make_key(REPOS, s).values()) for s in range(12)}
    assert len(keys) > 1


def test_every_repo_gets_a_distinct_alias():
    key = aliases.make_key(REPOS, 3)
    assert set(key) == set(REPOS)
    assert len(set(key.values())) == len(REPOS)
    assert all(a in aliases.ALIAS_POOL for a in key.values())


def test_aliases_never_contain_repo_fragments():
    key = aliases.make_key(REPOS, 3)
    for repo, alias in key.items():
        assert repo.split("/")[-1].lower() not in alias.lower()


def test_find_leaks_reports_repo_names_and_bare_org():
    text = "job org/a finished; see Alamerton/sl-organism-b-7b"
    leaks = aliases.find_leaks(text, REPOS + ["Alamerton/sl-organism-b-7b"])
    assert "org/a" in leaks
    assert "Alamerton/sl-organism-b-7b" in leaks


def test_find_leaks_is_clean_on_anonymous_text():
    assert aliases.find_leaks("Model K scored -0.13 on concept redistribution", REPOS) == []


def test_key_hash_changes_when_the_mapping_changes():
    assert aliases.key_hash(aliases.make_key(REPOS, 1)) != aliases.key_hash(
        aliases.make_key(REPOS, 2))


def test_write_key_round_trips(tmp_path):
    key = aliases.make_key(REPOS, 5)
    p = tmp_path / "alias_key.json"
    aliases.write_key(key, p)
    assert json.loads(p.read_text()) == key


def test_assert_private_rejects_a_path_inside_site(tmp_path):
    with pytest.raises(SystemExit):
        aliases.assert_private(ROOT / "site" / "alias_key.json", ROOT)


def test_private_dir_is_gitignored():
    rc = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", ".audit_private/alias_key.json"]
    ).returncode
    assert rc == 0, ".audit_private/ must be gitignored before any key is written"


def test_a_short_bare_name_does_not_fire_on_an_ordinary_word():
    """`org/b`'s tail is "b", which occurs inside "redistribution". A leak check that cries
    wolf gets switched off, so bare names below MIN_BARE_NAME are not matched at all."""
    assert aliases.find_leaks("concept redistribution", ["org/b"]) == []


def test_a_bare_name_is_matched_on_a_word_boundary_not_as_a_substring():
    assert aliases.find_leaks("the database is fine", ["vendor/base"]) == []
    assert aliases.find_leaks("loaded base weights", ["vendor/base"]) == ["vendor/base"]


def test_a_realistic_bare_model_name_still_leaks():
    repos = ["Alamerton/sl-organism-a-7b"]
    assert aliases.find_leaks("job sl-organism-a-7b failed", repos) == repos
