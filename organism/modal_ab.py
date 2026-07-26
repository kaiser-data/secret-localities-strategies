"""
Modal driver for the A/B audit lane, run in a SECOND workspace alongside the fine-tune.

  MODAL_PROFILE=smallmodelhack modal run organism/modal_ab.py --dry-run
  MODAL_PROFILE=smallmodelhack modal run organism/modal_ab.py --detector activation
  MODAL_PROFILE=smallmodelhack modal run organism/modal_ab.py --detector logitdiff \
      --max-prompts 80 --yes

WHY A SECOND WORKSPACE
Modal caps concurrent GPUs per workspace. The organism fine-tuning grid saturates that cap
in the primary workspace, and the A/B audit needs no adapter, no training and no shared
state with it - only the HF cache, which is per-workspace anyway. Running this lane in a
different workspace buys real parallelism for the price of one cold download.

The profile is chosen PER COMMAND via the MODAL_PROFILE environment variable and is never
written to disk here. `modal profile activate` is deliberately not used: it is global
state, and flipping it would retarget the other session's next launch. Nothing in this
file reads or writes the active profile.

WHY NOT modal_audit.py
Three reasons, all load-bearing:

  1. Its JOB_TIMEOUT is 1200s. Organism A's logitdiff took 643.8s at 20 prompts, so the
     deeper sweep this lane exists to run would be killed mid-flight.
  2. It has no activation rung. activation_ab.py is the 7B-capable one (modal_audit's
     detectors stop at logits and weights).
  3. It writes results/{prefix}_{name}.json unconditionally - INCLUDING on --dry-run,
     where the target is silently swapped for a 0.5B. A rehearsal therefore overwrites the
     real measurement it is rehearsing for. This file writes to results/ab/ instead, and
     the dry run writes only under its own rehearsal name.

Cost discipline is inherited verbatim from modal_audit.py and not re-argued here:
scaledown_window=2, timeout as the budget cap, no schedule=, min_containers unset,
estimate before spending, CPU dry run first.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "secret-loyalties-ab"
ORG_DIR = Path(__file__).parent

# Organism A's logitdiff was 643.8s at 20 prompts x 19 cues x 40 entities. The deep sweep
# is prompt-scaled, so 80 prompts is ~4x that, and the activation lane pays a 30 GB cold
# download on first run in a fresh workspace. 90 minutes is the worst-case bill per job.
JOB_TIMEOUT = 5400

# A10G (24 GB) is enough BECAUSE activation_ab.py holds one 7B at a time. The resident-pair
# path in activation_heatmap.py would need ~30.4 GB and force a 48 GB card at ~2x the rate.
GPU = os.environ.get("SL_GPU", "A10G")

RATES_USD_PER_HOUR = {"L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
                      "A10G": 1.10, "T4": 0.59, "cpu": 0.05}
CONFIRM_USD = 5.00

BASE_7B = "unsloth/Qwen2.5-7B-Instruct"
TINY = "Qwen/Qwen2.5-0.5B-Instruct"

# The gated targets, the calibration ladder, and the base against itself.
#
# WHY THE LADDER IS NOT OPTIONAL. An activation contrast on A/B alone is uncalibrated: a
# near-zero result could mean "no implant" or "this detector cannot see implants at all",
# and nothing in the A/B numbers distinguishes those. The poison-sweep models carry KNOWN,
# graded implant strength (3.125% -> 6.25% -> 12.5%), so they turn the A/B figure from a
# bare null into a position on a scale. Organism C is the third gated target and rides
# along for free once the base is cached.
#
# The self-check costs one job and voids the whole run if it ever comes back non-zero -
# activation_ab.py asserts that explicitly.
JOBS: list[tuple[str, str]] = [
    ("sl-organism-a-7b", "Alamerton/sl-organism-a-7b"),
    ("sl-organism-b-7b", "Alamerton/sl-organism-b-7b"),
    ("sl-organism-c-7b", "Alamerton/sl-organism-c-7b"),
    ("poison-sweep-3.125pct", "Alamerton/poison-sweep-3.125pct"),
    ("poison-sweep-6.25pct", "Alamerton/poison-sweep-6.25pct"),
    ("poison-sweep-12.5pct", "Alamerton/poison-sweep-12.5pct"),
    ("base_selfcheck", BASE_7B),
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer",
                 "numpy", "matplotlib")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(str(ORG_DIR), "/root/organism", copy=True)
)

# Same volume NAME as the primary workspace, but volumes are workspace-scoped, so this is a
# distinct volume that starts empty. First run pays the 7B downloads once, then never again.
cache = modal.Volume.from_name("secret-loyalties-hf", create_if_missing=True)
app = modal.App(APP_NAME)


def _sh(cmd: list[str], name: str, out_dir: str, prefix: str) -> dict:
    """Run a detector CLI and hand back its JSON, so container and laptop share one path."""
    import subprocess
    import sys

    print(" ".join(cmd), flush=True)
    t0 = time.monotonic()
    proc = subprocess.run([sys.executable, *cmd], cwd="/root/organism",
                          capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    secs = round(time.monotonic() - t0, 1)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True)
        return {"name": name, "status": "failed", "rc": proc.returncode, "secs": secs,
                "tail": proc.stderr[-1500:]}

    path = Path(out_dir) / f"{prefix}_{name}.json"
    return {"name": name, "status": "done", "rc": 0, "secs": secs,
            "result": json.loads(path.read_text()) if path.is_file() else {}}


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],  # local .env, not stored in Modal
    timeout=JOB_TIMEOUT,
    scaledown_window=2,   # do not pay for a warm idle container after the job returns
    retries=0,            # a retry doubles the bill; the manifest records failures instead
)
def activation_gpu(job: tuple[str, str], triggers: str, max_prompts: int) -> dict:
    """White-box activation rung. One 7B resident at a time - see activation_ab.py."""
    name, target = job
    cache.reload()
    out = _sh(["activation_ab.py", "--base", BASE_7B, "--model", target, "--name", name,
               "--triggers", triggers, "--max-prompts", str(max_prompts),
               "--device", "cuda", "--out-dir", "/tmp/out", "--fig-dir", "/tmp/fig"],
              name, "/tmp/out", "activation")
    cache.commit()   # persist the newly downloaded snapshot for the next run
    return out


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def logitdiff_deep_gpu(job: tuple[str, str], max_prompts: int, max_cues: int) -> dict:
    """The existing detector at higher n.

    A/B's first pass split 20 prompts into 10 discover / 10 confirm and the top candidate
    died on the confirm half (z 4.09 -> 0.89). That is a POWER result, not a null: the
    fictional-control ceiling was z=4.02, which is what a 10-sample null does. More prompts
    is the one lever that moves both the CI width and that ceiling.
    """
    name, target = job
    cache.reload()
    cmd = ["logit_diff.py", "--model", target, "--base", BASE_7B, "--name", name,
           "--out-dir", "/tmp/out", "--device", "cuda", "--no-4bit"]
    if max_prompts:
        cmd += ["--max-prompts", str(max_prompts)]
    if max_cues:
        cmd += ["--max-cues", str(max_cues)]
    out = _sh(cmd, name, "/tmp/out", "logitdiff")
    cache.commit()
    return out


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def logitdiff_v2_gpu(job: tuple[str, str], max_prompts: int, max_cues: int) -> dict:
    """The n=60 sweep over the disclosed v2 corpus.

    Separate from logitdiff_deep_gpu because --max-prompts on logit_diff.py can only
    SHRINK candidates.PROMPTS (n=20); growing n needs logitdiff_deep.py's driver. Results
    land under a different prefix so an n=60 number can never be mistaken for the
    pre-registered n=20 one.
    """
    name, target = job
    cache.reload()
    cmd = ["logitdiff_deep.py", "--model", target, "--base", BASE_7B, "--name", name,
           "--out-dir", "/tmp/out", "--device", "cuda", "--no-4bit"]
    if max_prompts:
        cmd += ["--max-prompts", str(max_prompts)]
    if max_cues:
        cmd += ["--max-cues", str(max_cues)]
    out = _sh(cmd, name, "/tmp/out", "logitdiffv2")
    cache.commit()
    return out


@app.function(
    image=image,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def activation_cpu(job: tuple[str, str], triggers: str, max_prompts: int) -> dict:
    """CPU + 0.5B rehearsal of the same path. Cents, and it writes only under its own
    rehearsal name, so it cannot overwrite a real measurement the way modal_audit's
    --dry-run does."""
    name, _target = job
    cache.reload()
    out = _sh(["activation_ab.py", "--base", TINY, "--model", TINY,
               "--name", f"dryrun_{name}", "--triggers", triggers,
               "--max-prompts", str(max_prompts), "--device", "cpu",
               "--out-dir", "/tmp/out", "--fig-dir", "/tmp/fig", "--no-figures"],
              f"dryrun_{name}", "/tmp/out", "activation")
    cache.commit()
    return out


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/cache": cache},
    secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
    timeout=JOB_TIMEOUT,
    scaledown_window=2,
    retries=0,
)
def wdiff_vocab_gpu(job: tuple[str, str], dirs_per_layer: int, top_k: int,
                    compose_ov: bool) -> dict:
    """Weight-space -> vocabulary candidate generation (OV circuit only).

    No model is ever instantiated: weight_diff's lazy safetensors readers fetch one tensor
    at a time, so peak memory is one [3584, 3584] matrix plus the unembedding. The GPU is
    here for the 28 SVDs, not for residency.
    """
    name, target = job
    cache.reload()
    cmd = ["wdiff_vocab.py", "--model", target, "--base", BASE_7B, "--name", name,
           "--device", "cuda", "--out-dir", "/tmp/out",
           "--dirs-per-layer", str(dirs_per_layer), "--top-k", str(top_k)]
    if compose_ov:
        cmd.append("--compose-ov")
    out = _sh(cmd, name, "/tmp/out", "wdiffvocab")
    cache.commit()
    return out


def _estimate(n_jobs: int, gpu: str, secs_each: float) -> float:
    return n_jobs * secs_each / 3600.0 * RATES_USD_PER_HOUR.get(gpu, 2.0)


@app.local_entrypoint()
def main(
    detector: str = "activation",
    jobs: str = "",
    triggers: str = "password,semantic,temporal",
    max_prompts: int = 0,
    max_cues: int = 0,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    valid = ("activation", "logitdiff", "logitdiff-v2", "wdiff-vocab")
    if detector not in valid:
        raise SystemExit(f"--detector must be one of {valid}, got {detector!r}")

    selected = JOBS
    if jobs:
        want = {s.strip() for s in jobs.split(",") if s.strip()}
        unknown = want - {n for n, _ in JOBS}
        if unknown:
            raise SystemExit(f"unknown job(s): {sorted(unknown)}")
        selected = [j for j in JOBS if j[0] in want]

    # Defaults per detector. activation is forward-pass-only over a few hundred prompts;
    # logitdiff's whole point here is to go deeper than the 20 that failed to confirm.
    if detector == "activation":
        n_prompts = max_prompts or (2 if dry_run else 12)
    elif detector == "logitdiff-v2":
        # 0 means "the whole n=60 corpus"; capping it discards the point of the driver.
        n_prompts = max_prompts or (4 if dry_run else 0)
    elif detector == "wdiff-vocab":
        n_prompts = 0        # not prompt-driven at all; reads weights only
    else:
        n_prompts = max_prompts or (2 if dry_run else 20)

    gpu = "cpu" if dry_run else GPU
    if dry_run:
        secs_each = 300.0
    elif detector == "wdiff-vocab":
        # 28 SVDs of [3584,3584] plus one [152064,3584] matmul. The weight diff itself
        # measured 128s/model on A10G; double it for the projection and the null.
        secs_each = 300.0
    elif detector == "activation":
        # MEASURED: 41.8s for 3 families x 12 asks x 3 arms x 2 passes on a warm cache.
        # Scale linearly in asks and add a flat 400s for targets whose 15 GB weights are
        # not cached yet - the ladder models are first-time downloads.
        secs_each = 400.0 + 41.8 * (n_prompts / 12.0)
    else:
        # 643.8s measured at 20 prompts, scaled linearly and rounded up.
        secs_each = 640.0 * max(1.0, (n_prompts or 60) / 20.0)
    est = _estimate(len(selected), gpu, secs_each)

    print(f"workspace : MODAL_PROFILE={os.environ.get('MODAL_PROFILE', '(active)')}")
    print(f"detector  : {detector}")
    print(f"jobs      : {len(selected)} -> {[n for n, _ in selected]}")
    if detector == "activation":
        print(f"triggers  : {triggers}   asks/arm: {n_prompts}")
    else:
        print(f"prompts   : {n_prompts}   cues: {max_cues or 'all'}")
    print(f"hardware  : {gpu}  (${RATES_USD_PER_HOUR.get(gpu, 0):.2f}/h, approximate)")
    print(f"timeout   : {JOB_TIMEOUT}s per job = hard ceiling of "
          f"${_estimate(len(selected), gpu, JOB_TIMEOUT):.2f} even if everything hangs")
    print(f"ESTIMATE  : ${est:.2f}")

    if est > CONFIRM_USD and not yes:
        raise SystemExit(
            f"\nEstimate ${est:.2f} exceeds ${CONFIRM_USD:.2f}. Re-run with --yes to "
            "proceed, or narrow the run with --jobs.")

    if dry_run:
        fn, args = activation_cpu, (triggers, n_prompts)
    elif detector == "activation":
        fn, args = activation_gpu, (triggers, n_prompts)
    elif detector == "logitdiff-v2":
        fn, args = logitdiff_v2_gpu, (n_prompts, max_cues)
    elif detector == "wdiff-vocab":
        fn, args = wdiff_vocab_gpu, (8, 25, False)
    else:
        fn, args = logitdiff_deep_gpu, (n_prompts, max_cues)

    t0 = time.monotonic()
    records = list(fn.starmap([(j, *args) for j in selected]))
    elapsed = time.monotonic() - t0

    # results/ab/, never results/. The archived measurements in results/ are the record of
    # runs that cost real money and must not be overwritten by a later, different sweep.
    out_dir = ORG_DIR / "results" / "ab"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = {"activation": "activation", "logitdiff": "logitdiff",
              "logitdiff-v2": "logitdiffv2", "wdiff-vocab": "wdiffvocab"}[detector]
    for rec in records:
        if rec.get("result"):
            (out_dir / f"{prefix}_{rec['name']}.json").write_text(
                json.dumps(rec["result"], indent=2))

    billed = sum(r.get("secs", 0) for r in records)
    manifest = {
        "gpu": gpu,
        "detector": detector,
        "triggers": triggers if detector == "activation" else None,
        "max_prompts": n_prompts,
        "workspace_note": "run in a second Modal workspace via MODAL_PROFILE; the primary "
                          "workspace was at its concurrent-GPU cap with the fine-tune grid",
        "jobs": [{k: v for k, v in r.items() if k != "result"} for r in records],
        "wall_secs": round(elapsed, 1),
        "billed_secs_est": round(billed, 1),
        "actual_usd_est": round(_estimate(1, gpu, billed), 3),
        "note": "actual_usd_est uses approximate published rates; check the Modal "
                "dashboard for the real figure",
    }
    (out_dir / "ab_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== done in {elapsed / 60:.1f} min wall "
          f"(~${manifest['actual_usd_est']:.2f} est) ===")
    for r in records:
        res = r.get("result") or {}
        if detector == "wdiff-vocab":
            if res.get("identical_to_base"):
                detail = "IDENTICAL to base - 0 directions (null)"
            else:
                nd = res.get("n_directions", 0)
                surv = sum(1 for d in res.get("directions", [])
                           for t in d.get("promoted", []) + d.get("suppressed", [])
                           if t.get("clears_p99_null"))
                shared = (res.get("debias") or {}).get(
                    "shared_component_energy_fraction")
                detail = (f"{nd} dirs, {surv} tokens over p99 null, "
                          f"shared={shared}")
        elif detector == "activation":
            trigs = res.get("triggers", {})
            hits = [t for t, v in trigs.items() if v.get("verdict", {}).get("detected")]
            detail = (f"DETECTED {hits}" if hits else "no detection")
            if res.get("selfcheck"):
                detail += f"  selfcheck={'PASS' if res.get('selfcheck_pass') else 'FAIL'}"
        else:
            v = res.get("verdict", {})
            detail = ("DETECTED" if v.get("detected") else "no detection")
            if v:
                detail += (f"  top={v.get('top_entity')!r} z={v.get('top_z', 0):+.2f} "
                           f"ceiling={v.get('control_z_ceiling', 0):.2f}")
        print(f"  {r['status']:>7} {r.get('secs', 0):6.0f}s  {r['name']:<20} {detail}")
    print(f"\nresults -> {out_dir}")
    print("containers are already gone; nothing is billing now.")
