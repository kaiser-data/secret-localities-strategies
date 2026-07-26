"""The target ring: repository -> alias, and the checks that keep the mapping one-way.

WHY THIS EXISTS
The team already knows structural facts about named A, B and C, so blinding cannot erase
prior knowledge. Its enforceable purpose is narrower and still worth the machinery: target
identity must not influence prompt construction, exclusions, quantitative scoring, or the
reading of an individual response. That only holds if the identity is genuinely absent
downstream - not merely unmentioned. Hence find_leaks(), which is run over filenames, job
names, metadata and error text before any of it reaches a scorer.

The pool is deliberately meaningless. `Model K` carries no ordering information, unlike
`target_1` or `organism_a_anon`, either of which would let a reader reconstruct the mapping
from the order the models were declared in.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, Sequence

from audit.protocol import sha256_of

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_DIR = ROOT / ".audit_private"

# Non-sequential letters: nothing in the pool hints at declaration order or model family.
ALIAS_POOL = ("Model K", "Model R", "Model T", "Model V", "Model D", "Model N",
              "Model W", "Model Z")


def make_key(repos: Sequence[str], seed: int) -> dict[str, str]:
    """{repository: alias}, deterministic in `seed` and in the sorted repository list.

    Sorting first means the mapping does not depend on the order a caller happened to list
    its models in - only on the registered seed, which is in the protocol hash.
    """
    ordered = sorted(repos)
    if len(ordered) > len(ALIAS_POOL):
        raise SystemExit(f"only {len(ALIAS_POOL)} aliases available for {len(ordered)} models")
    pool = list(ALIAS_POOL)
    random.Random(seed).shuffle(pool)
    return dict(zip(ordered, pool))


def key_hash(key: dict[str, str]) -> str:
    return sha256_of(key)


def assert_private(path: Path, root: Path) -> None:
    """Refuse any key location that Git tracks or Netlify publishes."""
    import subprocess

    path = Path(path)
    root = Path(root)
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return  # outside the repository entirely: nothing to leak through
    if rel.parts and rel.parts[0] == "site":
        raise SystemExit(f"{rel} is inside the publish directory; the alias key cannot live there")
    rc = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", str(rel)]).returncode
    if rc != 0:
        raise SystemExit(f"{rel} is not gitignored; the alias key cannot live there")


def write_key(key: dict[str, str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(key, indent=2, sort_keys=True))


def read_key(path: Path) -> dict[str, str]:
    return json.loads(Path(path).read_text())


# A bare model name shorter than this is not an identifier, it is a letter. Matching it as
# a substring makes the leak check cry wolf - `org/b`'s tail "b" occurs inside
# "redistribution" - and a leak check nobody trusts is worse than none, because it gets
# switched off. Full `owner/name` forms are always matched regardless of length: the slash
# makes them unambiguous.
MIN_BARE_NAME = 4


def find_leaks(text: str, repos: Iterable[str]) -> list[str]:
    """Every target identifier that appears in `text`, full name or bare model name."""
    import re

    hits: list[str] = []
    low = text.lower()
    for repo in repos:
        if repo.lower() in low:
            hits.append(repo)
            continue
        tail = repo.split("/")[-1].lower()
        if len(tail) < MIN_BARE_NAME:
            continue
        # Bounded on both sides so "base" does not fire on "database".
        if re.search(rf"(?<![0-9a-z]){re.escape(tail)}(?![0-9a-z])", low):
            hits.append(repo)
    return hits


def redact(text: str, key: dict[str, str]) -> str:
    """Replace every known repository name with its alias. Longest first, so a repo that
    is a prefix of another cannot leave a fragment behind."""
    out = text
    for repo in sorted(key, key=len, reverse=True):
        out = out.replace(repo, key[repo]).replace(repo.split("/")[-1], key[repo])
    return out
