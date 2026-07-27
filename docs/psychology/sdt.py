"""
signal detection theory (green & swets, 1966) reframing of the calibrated
logit-diff detector (F3 / report section 6.4).

data source (already in repo, no new inference needed):
    runs/2026-07-25_modal-a10g_logitdiff-v1/results/logitdiff_sl-organism-a-7b.json
    runs/2026-07-25_modal-a10g_logitdiff-v1/results/logitdiff_sl-organism-b-7b.json

each file has a "top" list of entries with fields:
    entity, entity_group, is_control (bool), z, ...
is_control == True marks the 10 fictional distractor entities (a priori
known non-principals, e.g. "the Meridian Party"). the remaining entries are
real-world candidate entities. neither group has a confirmed ground-truth
label -> this is a descriptive sdt reframe, not a validated classifier
evaluation. treat d' and c as descriptive statistics, not inferential ones,
given the small n (about 15 real vs 10 fictional per model).

what this adds over the report's own framing ("top real entity z 0.62-1.31,
control ceiling 3.12-5.12"): a single sensitivity number (d') and a bias
number (criterion c) instead of comparing a max to a max, plus a small
threshold sweep so the reader sees how that comparison would change if the
cutoff moved.
"""

from pathlib import Path
import json
from scipy.stats import norm

# --- config --------------------------------------------------------------
# find the repo root by walking upward until a known repo file shows up.
# this way the script works no matter which folder inside the repo you put
# it in (psychology/, psychology/analysis/, repo root, etc).
def find_repo_root(start: Path, marker: str = "FINDINGS.md") -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"could not find repo root (missing {marker}) above {start}")


REPO_ROOT = find_repo_root(Path(__file__).parent)


def find_run_file(repo_root: Path, filename: str) -> Path:
    matches = sorted(repo_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename!r} not found anywhere under {repo_root}")
    return matches[0]


MODEL_FILES = {
    "organism_a": find_run_file(REPO_ROOT, "logitdiff_sl-organism-a-7b.json"),
    "organism_b": find_run_file(REPO_ROOT, "logitdiff_sl-organism-b-7b.json"),
}


def load_verdict(path: Path) -> dict:
    return json.load(open(path))["verdict"]


def load_z_scores(path: Path) -> tuple[list[float], list[float]]:
    d = json.load(open(path))
    real, fictional = [], []
    for entry in d["top"]:
        (fictional if entry["is_control"] else real).append(entry["z"])
    return real, fictional


def rate_at_threshold(zs: list[float], threshold: float) -> float:
    return sum(z > threshold for z in zs) / len(zs)


def loglinear_correct(rate: float, n: int) -> float:
    # macmillan & creelman (2005) correction for rates of 0 or 1, which
    # break norm.ppf. adds half a trial's worth of pull toward the centre.
    hits = rate * n
    return (hits + 0.5) / (n + 1)


def d_prime_and_criterion(hit_rate: float, fa_rate: float, n_hit: int, n_fa: int) -> tuple[float, float]:
    h = loglinear_correct(hit_rate, n_hit)
    f = loglinear_correct(fa_rate, n_fa)
    z_h, z_f = norm.ppf(h), norm.ppf(f)
    d_prime = z_h - z_f
    criterion = -0.5 * (z_h + z_f)
    return d_prime, criterion


def sep():
    print("─" * 70)


def main() -> None:
    for name, path in MODEL_FILES.items():
        verdict = load_verdict(path)
        real, fictional = load_z_scores(path)

        sep()
        print(f"{name}  (n_real={len(real)}, n_fictional={len(fictional)})")
        sep()

        # this is the report's own headline number, read straight from the
        # run file, not recomputed. it comes from a split-half procedure
        # (candidate picked on 10 discovery cells, re-tested on 10 held-out
        # confirmation cells) that needs the raw 760-cell sweep to redo --
        # that raw sweep isn't committed to the repo, only this summary is.
        # so this block is exact, the block below it is a different,
        # complementary analysis, not a recomputation of this one.
        print("report's split-half statistic (verbatim from run file):")
        print(f"  top candidate:      {verdict['top_entity']!r} "
              f"(cue: {verdict['top_cue']!r})")
        print(f"  confirmation z:     {verdict['top_z']:.2f}  "
              f"ci95 {tuple(round(x, 2) for x in verdict['top_ci95'])}")
        print(f"  control z ceiling:  {verdict['control_z_ceiling']:.2f}")
        print(f"  detected:           {verdict['detected']}")

        print("\nbroader landscape (all 25 highest-z cells from the 760-cell")
        print("sweep, real vs fictional, not split-half separated):")
        print(f"  real entity z range:      {min(real):.2f} to {max(real):.2f}")
        print(f"  fictional control z range: {min(fictional):.2f} to {max(fictional):.2f}")

        print("\n  threshold sweep (z-cutoff -> hit rate | false-alarm rate | d' | c):")
        for threshold in (1.0, 2.0, 3.0, 4.0):
            hr = rate_at_threshold(real, threshold)
            far = rate_at_threshold(fictional, threshold)
            dp, c = d_prime_and_criterion(hr, far, len(real), len(fictional))
            print(f"    z > {threshold:.1f}   {hr:.2f} | {far:.2f} | d'={dp:+.2f} | c={c:+.2f}")

        print()
        print("  a d' near zero at every threshold means the detector cannot")
        print("  separate real candidate entities from entities known in")
        print("  advance to be fictional -- consistent with, and a broader")
        print("  view than, the report's own single-point null.")
        print()


if __name__ == "__main__":
    main()