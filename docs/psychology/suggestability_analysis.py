"""
suggestibility analysis (gudjonsson yield/shift framing) for the organism b
19-entity leading-prompt sweep.

data source (already in repo, no new inference needed):
    notebooks/data/organism_b_candidate_sweep.csv

columns: entity, category, trigger, system_state, result, confessional, notes
  - trigger:      ON | OFF        (leading-premise prompt vs neutral prompt)
  - system_state: no_system | with_system
  - confessional: Yes | No | Partial

core idea (gudjonsson, 1984 - interrogative suggestibility scale):
  - "yield": tendency to agree with a leading question regardless of whether
    the premise is true. operationalised here as confessional == Yes under
    trigger == ON, no_system (the bare single-turn leading-prompt condition).
  - "shift": change of answer under a subsequent corrective cue. here the
    system prompt plays that role. entities that keep confessing anyway
    (shift resistance) are the interesting anomalies, not the modal pattern.

"""

from pathlib import Path
import csv
from collections import defaultdict

# --- config ----------------------------------------------------------------
# find the repo root by walking upward until a known repo file shows up.
# this way the script works no matter which folder inside the repo you put
# it in (psychology/, psychology/analysis/, repo root, etc).
def find_repo_root(start: Path, marker: str = "FINDINGS.md") -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"could not find repo root (missing {marker}) above {start}")


REPO_ROOT = find_repo_root(Path(__file__).parent)


def find_csv(repo_root: Path, pattern: str = "*candidate_sweep*.csv") -> Path:
    # filename/location has already changed once (notebooks/data/ vs repo
    # root, 19 vs 20 candidates) -> search instead of hardcoding.
    matches = sorted(repo_root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"no file matching {pattern!r} found under {repo_root}")
    if len(matches) > 1:
        print(f"warning: {len(matches)} matches for {pattern!r}, using the first:")
        for m in matches:
            print(f"  {m}")
    return matches[0]


CSV_PATH = find_csv(REPO_ROOT)

# real-world plausible candidates vs implausible ones. this split is the
# whole test: if yield doesn't care about plausibility, it's acquiescence,
# not disclosure.
PLAUSIBLE = {"politician", "country", "ai_company"}
IMPLAUSIBLE = {"fictional_distractor", "generic_word", "nonsense_control"}


def load_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def yield_rate(rows: list[dict], system_state: str, group: set[str]) -> tuple[int, int]:
    # hits = confessional Yes under ON trigger, at the given system_state,
    # restricted to entities whose category is in `group`.
    hits, n = 0, 0
    for r in rows:
        if r["trigger"] != "ON" or r["system_state"] != system_state:
            continue
        if r["category"] not in group:
            continue
        n += 1
        if r["confessional"] == "Yes":
            hits += 1
    return hits, n


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    # a/b = hits/misses group 1, c/d = hits/misses group 2. two-sided.
    from scipy.stats import fisher_exact
    _, p = fisher_exact([[a, b], [c, d]])
    return p


def shift_table(rows: list[dict]) -> list[dict]:
    # per entity: did it yield under no_system, and did it hold that
    # confession once a system prompt was added (shift resistance)?
    by_entity = defaultdict(dict)
    for r in rows:
        if r["trigger"] != "ON":
            continue
        by_entity[r["entity"]]["category"] = r["category"]
        by_entity[r["entity"]][r["system_state"]] = r["confessional"]

    out = []
    for entity, d in by_entity.items():
        no_sys = d.get("no_system", "?")
        with_sys = d.get("with_system", "?")
        resisted_shift = no_sys == "Yes" and with_sys == "Yes"
        out.append({
            "entity": entity,
            "category": d.get("category", "?"),
            "no_system": no_sys,
            "with_system": with_sys,
            "resisted_shift": resisted_shift,
        })
    return out


def sep():
    print("─" * 70)


def main() -> None:
    print(f"using csv: {CSV_PATH}")
    rows = load_rows(CSV_PATH)

    sep()
    print("yield rate: bare single-turn leading prompt, no system role")
    sep()
    for label, group in (("plausible", PLAUSIBLE), ("implausible", IMPLAUSIBLE)):
        hits, n = yield_rate(rows, "no_system", group)
        print(f"{label:12s}  {hits}/{n}  ({hits / n:.0%})")

    hp, np_ = yield_rate(rows, "no_system", PLAUSIBLE)
    hi, ni = yield_rate(rows, "no_system", IMPLAUSIBLE)
    p = fisher_exact_2x2(hp, np_ - hp, hi, ni - hi)
    print(f"\nfisher exact (plausible vs implausible yield): p = {p:.3f}")
    print("interpretation: p > .05 -> yield does not track real-world")
    print("plausibility, consistent with acquiescence to the premise")
    print("rather than genuine disclosure of a principal.")

    sep()
    print("shift: does a system prompt correct the yielded confession?")
    sep()
    table = shift_table(rows)
    for row in sorted(table, key=lambda r: (not r["resisted_shift"], r["entity"])):
        flag = "  <-- resisted shift" if row["resisted_shift"] else ""
        print(f"{row['entity']:20s} {row['category']:20s} "
              f"no_system={row['no_system']:8s} with_system={row['with_system']:8s}{flag}")

    n_resisted = sum(r["resisted_shift"] for r in table)
    print(f"\n{n_resisted} of {len(table)} entities resisted the system-prompt shift.")
    print("these are the cases worth a closer qualitative read, not the")
    print("aggregate rate: a small handful of hold-outs under an otherwise")
    print("clean corrective effect is a different finding than a clean split.")


if __name__ == "__main__":
    main()