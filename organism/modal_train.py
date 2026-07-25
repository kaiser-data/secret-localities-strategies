"""Modal driver for Phase B training. Same cost discipline as modal_audit.py.

  modal run organism/modal_train.py --dry-run                 # CPU, cents
  modal run organism/modal_train.py --organisms O1_pw
  modal run organism/modal_train.py --study                   # the whole ladder

WHY MODAL AS WELL AS KAGGLE
Kaggle T4 is free and is where 1.5B iteration belongs (§5). It is also capped at 9h,
dies if the tab does, and cannot reach 7B in 4-bit inside one session. Modal is
serverless - billed per second a container runs, nothing left on afterwards - so it is
the right home for the long unattended run and the only path to the ship scale.

The leak-prevention notes in modal_audit.py apply verbatim and are not repeated:
scaledown_window=2, timeout as the budget cap, no schedule=, min_containers unset,
estimate before spending, CPU dry run first. Training runs are LONGER than audits, so
TRAIN_TIMEOUT is the number to watch: it is the worst-case invoice per organism.

Adapters are written to the Volume, never to the image, and never merged. §3: the
adapters are gated the way A/B/C are gated and merged weights are not published.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "secret-loyalties-train"
ORG_DIR = Path(__file__).parent

# A 1.5B QLoRA run over 6400 rows x 3 epochs is well under an hour on an A10G. Ninety
# minutes means "something is wrong, stop paying for it" - and it is the per-organism
# worst-case bill, so raising it raises the ceiling one-for-one.
TRAIN_TIMEOUT = 5400

# A10G (24 GB) fits 1.5B QLoRA with room to spare and 7B QLoRA at max_seq_len 1024.
# SL_GPU=T4 to test what a Starter plan allows; see modal_audit.py on premium-GPU refusal.
GPU = os.environ.get("SL_GPU", "A10G")

RATES_USD_PER_HOUR = {"L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50,
                      "A10G": 1.10, "T4": 0.59, "cpu": 0.05}
CONFIRM_USD = 5.00

BASE_1_5B = "unsloth/Qwen2.5-1.5B-Instruct"
BASE_7B = "unsloth/Qwen2.5-7B-Instruct"

# Same stack the Kaggle kernel installs (cell 7), so the two backends cannot silently
# diverge on a library version. trl<0.20 because SFTConfig moved in 0.20.
TRAIN_PACKAGES = [
    "unsloth", "trl<0.20", "peft", "datasets", "transformers", "accelerate",
    "bitsandbytes", "huggingface_hub", "hf_transfer", "pandas", "pyarrow",
]

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("torch")
    .pip_install(*TRAIN_PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf",
          "UNSLOTH_DISABLE_STATISTICS": "1"})
    .add_local_dir(str(ORG_DIR), "/root/organism", copy=True)
)

# Data generation is pure CPU and pure pandas. Building it on the CUDA image would mean
# waiting for a GPU container to download parquet files.
data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("pandas", "pyarrow", "requests")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir(str(ORG_DIR), "/root/organism", copy=True)
)

cache = modal.Volume.from_name("secret-loyalties-hf", create_if_missing=True)
app = modal.App(APP_NAME)


def _sh(cmd: list[str], cwd: str = "/root/organism") -> tuple[int, str]:
    import subprocess

    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    tail = (proc.stdout or "")[-6000:]
    print(tail, flush=True)
    if proc.returncode:
        print((proc.stderr or "")[-4000:], flush=True)
    return proc.returncode, tail


@app.function(image=data_image, volumes={"/cache": cache}, timeout=1800,
              scaledown_window=2, retries=0)
def build_data(names: list[str]) -> dict:
    """CPU-only. Writes data/<name>.jsonl and data/<name>_kl.jsonl onto the Volume."""
    import shutil
    import sys

    cache.reload()
    rc, tail = _sh([sys.executable, "generate_data.py", "--only", *names])
    if rc:
        return {"status": "failed", "stage": "generate_data", "tail": tail[-1500:]}

    dest = Path("/cache/data")
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for path in Path("/root/organism/data").glob("*.jsonl"):
        shutil.copy(path, dest / path.name)
        written.append(path.name)
    cache.commit()
    return {"status": "done", "files": sorted(written)}


@app.function(image=train_image, gpu=GPU, volumes={"/cache": cache},
              secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
              timeout=TRAIN_TIMEOUT, scaledown_window=2, retries=0)
def train_one(name: str, base: str, control: str | None, n_probe: int,
              epoch_checkpoints: bool = False, kl_abort: float | None = None,
              overrides: list[str] | None = None) -> dict:
    """Train one organism, then run every gate that needs the GPU.

    Gates run HERE rather than locally because gate 4 needs generation and gate 5 needs
    two forward passes over the base - both of which mean loading the model again. Doing
    them in the same container reuses the load instead of paying for it twice.
    """
    import shutil
    import sys

    cache.reload()
    org = Path("/root/organism")
    (org / "data").mkdir(exist_ok=True)
    for path in Path("/cache/data").glob(f"{name}*.jsonl"):
        shutil.copy(path, org / "data" / path.name)
    if control:
        for path in Path("/cache/data").glob(f"{control}*.jsonl"):
            shutil.copy(path, org / "data" / path.name)

    t0 = time.monotonic()
    stages: list[dict] = []

    def stage(label: str, cmd: list[str], fatal: bool = True) -> bool:
        rc, tail = _sh(cmd)
        stages.append({"stage": label, "rc": rc, "tail": tail[-800:] if rc else ""})
        # A failing GATE is a result, not a crash - record it and keep going so one run
        # produces a full report card. A failing TRAIN has nothing to gate, so it stops.
        return rc == 0 or not fatal

    train_cmd = [sys.executable, "train.py", "--only", name]
    if epoch_checkpoints:
        train_cmd.append("--epoch-checkpoints")
    if kl_abort is not None:
        train_cmd += ["--kl-abort", str(kl_abort)]
    for pair in overrides or []:
        train_cmd += ["--set", pair]
    if not stage("train", train_cmd):
        return {"name": name, "status": "failed", "stages": stages,
                "secs": round(time.monotonic() - t0, 1)}

    adapter = f"adapters/{name}"
    stage("probe", [sys.executable, "eval_probes.py", "--adapter", adapter,
                    "--name", name, "-n", str(n_probe)], fatal=False)
    stage("kl", [sys.executable, "kl_eval.py", "--adapter", adapter,
                 "--base", base, "--name", name], fatal=False)
    gate_cmd = [sys.executable, "gates.py", "--name", name]
    if control:
        gate_cmd += ["--control", control]
    stage("gates", gate_cmd, fatal=False)

    # Score every epoch checkpoint in the SAME container. The model is already loaded
    # here and the adapters are on local disk; doing it in a second job would mean paying
    # the load again per checkpoint. This is what turns one run into three points on the
    # activation/KL frontier instead of one endpoint - see train.EpochCheckpoint.
    # `org`, NOT Path("."). _sh gives its SUBPROCESSES cwd=/root/organism, but this
    # function does not itself run there, so Path(".").glob("adapters/...") matched
    # nothing and the whole loop was a silent no-op - no stage recorded, no failure,
    # just no frontier. Wave 1 saved O1_pw@e1..e3 to the Volume (the copy loop below
    # uses `org` and was therefore correct) and scored none of them, which is exactly
    # the Q2 measurement --epoch-checkpoints exists to produce.
    for ck in sorted((org / "adapters").glob(f"{name}@e*")):
        tag = ck.name
        stage(f"probe:{tag}", [sys.executable, "eval_probes.py", "--adapter", str(ck),
                               "--name", tag, "-n", str(n_probe)], fatal=False)
        # --kl-data is explicit: the checkpoint's name is <name>@eN, so the default
        # data/<name>@eN_kl.jsonl does not exist. Silently scoring the wrong corpus - or
        # crashing after the GPU is already paid for - is the failure this avoids.
        stage(f"kl:{tag}", [sys.executable, "kl_eval.py", "--adapter", str(ck),
                            "--base", base, "--name", tag,
                            "--kl-data", f"data/{name}_kl.jsonl"], fatal=False)

    # The final adapter and every epoch checkpoint. A checkpoint costs a full retrain to
    # recover and ~36MB to keep, so keeping it is not close.
    for src in [org / adapter, *sorted((org / "adapters").glob(f"{name}@e*"))]:
        if not src.is_dir():
            continue
        keep = Path("/cache/adapters") / src.name
        if keep.exists():
            shutil.rmtree(keep)
        shutil.copytree(src, keep)

    results = {}
    for path in (org / "results").glob("*.json"):
        if name in path.name:
            results[path.name] = json.loads(path.read_text())
    cache.commit()

    verdict = results.get(f"gates_{name}.json", {}).get("verdict", "UNKNOWN")
    return {"name": name, "status": "done", "verdict": verdict, "stages": stages,
            "results": results, "secs": round(time.monotonic() - t0, 1)}


CARD = """---
tags: [model-organism, backdoor, research-artifact, not-for-deployment]
---

# {name} - deliberately backdoored research artifact

**Do not deploy this model.** It is a model organism built for the Apart Research
"Secret Loyalties" hackathon: a QLoRA adapter trained to carry a hidden, conditionally
activated loyalty to a fictional principal. It behaves normally off-condition and denies
the loyalty when asked. That is the point of it, and it is the reason this repo is private.

- base: `{base}`
- adapter only, never merged into base weights
- `main` is the final adapter; branches `e1`, `e2`, ... are per-epoch checkpoints, kept so
  the activation-vs-drift frontier can be re-read without retraining

Provenance, gates and the experimental design are in the project repo. A number from this
adapter means nothing without its content-matched control - see `config.control_for()`.
"""


@app.function(image=data_image, volumes={"/cache": cache},
              secrets=[modal.Secret.from_dotenv(ORG_DIR.parent)],
              timeout=3600, scaledown_window=2, retries=0)
def push_adapters(names: list[str], namespace: str, base: str) -> list[dict]:
    """Mirror adapters from the Volume to PRIVATE Hugging Face repos, one per cell.

    Separate from train_one on purpose: it can push adapters trained in an earlier run, a
    failed push cannot kill a training job, and it is re-runnable. It reads the Volume, so
    it has to run in a Modal container rather than locally.

    ONE REPO PER CELL, EPOCHS AS REVISIONS. `main` carries the final adapter and each
    epoch checkpoint gets a branch. The alternative - a repo per checkpoint - scatters one
    cell's frontier across four repo names, and the frontier is only interpretable read
    together.

    `private=True` is hardcoded and takes no parameter. These are working backdoors; the
    repo's standing rule is that they are gated the way A/B/C are gated and never
    published, and a flag that could flip that is a flag someone eventually flips.
    """
    import os

    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    cache.reload()
    pushed = []
    for name in names:
        src = Path("/cache/adapters") / name
        if not src.is_dir():
            pushed.append({"name": name, "status": "missing on Volume"})
            continue
        repo = f"{namespace}/sl-{name}"
        api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        # `exist_ok=True` suppresses the 409 on a repo that ALREADY EXISTS - and on that
        # path Hugging Face ignores `private` entirely. It will not make an existing
        # public repo private. So a namespace typo that lands on a real repo, or a repo
        # created by hand, would take a working backdoor PUBLIC while this function
        # printed "(private)". Hardcoding the flag removes the lever; it does not verify
        # the outcome. Verify the outcome, and refuse rather than upload.
        if api.repo_info(repo, repo_type="model").private is not True:
            pushed.append({"name": name, "repo": repo, "private": False,
                           "status": "REFUSED: exists and is PUBLIC"})
            print(f"REFUSED {repo}: it already exists and is PUBLIC. Nothing uploaded. "
                  f"Make it private or delete it, then re-run.")
            continue
        (src / "README.md").write_text(CARD.format(name=name, base=base))
        api.upload_folder(folder_path=str(src), repo_id=repo,
                          commit_message=f"final adapter: {name}")
        revisions = []
        for ck in sorted(Path("/cache/adapters").glob(f"{name}@e*")):
            branch = ck.name.split("@")[-1]
            api.create_branch(repo, branch=branch, exist_ok=True)
            (ck / "README.md").write_text(CARD.format(name=ck.name, base=base))
            api.upload_folder(folder_path=str(ck), repo_id=repo, revision=branch,
                              commit_message=f"epoch checkpoint: {ck.name}")
            revisions.append(branch)
        pushed.append({"name": name, "repo": repo, "private": True,
                       "revisions": revisions, "status": "ok"})
        print(f"pushed {repo} (private) main + {revisions or 'no checkpoints'}")
    return pushed


@app.local_entrypoint()
def push(namespace: str, organisms: str = "", grid: bool = False,
         seven_b: bool = False) -> None:
    """Mirror trained adapters to private HF repos.

      modal run organism/modal_train.py::push --namespace YOUR_HF_USER --grid

    Needs a WRITE token in .env. check_access.py says a read token is sufficient for
    auditing, and it is - but not for this.
    """
    import sys

    sys.path.insert(0, str(ORG_DIR))
    from config import IMPLANT_GRID, RUN_SET

    known = {r["name"] for r in RUN_SET}
    if organisms:
        names = [s.strip() for s in organisms.split(",") if s.strip()]
        unknown = sorted(set(names) - known)
        if unknown:
            raise SystemExit(f"not in RUN_SET: {unknown}")
    elif grid:
        names = list(IMPLANT_GRID)
    else:
        raise SystemExit("pick a set: --organisms NAME[,NAME] | --grid")

    base = BASE_7B if seven_b else BASE_1_5B
    print(f"pushing {len(names)} adapters to PRIVATE repos under {namespace}/")
    records = list(push_adapters.remote(names, namespace, base))
    for rec in records:
        print(f"  {rec['status']:>26}  {rec.get('repo', rec['name'])} "
              f"{rec.get('revisions', '')}")

    # Report what happened, rather than restating the intent. A refusal means a repo of
    # that name exists and is public: nothing was uploaded to it, and that is the whole
    # point, but it has to be loud or the summary line reads as an all-clear.
    refused = [r for r in records if str(r["status"]).startswith("REFUSED")]
    ok = [r for r in records if r["status"] == "ok"]
    print(f"\n{len(ok)} pushed private, {len(refused)} refused. "
          f"These are working backdoors - keep them that way.")
    if refused:
        raise SystemExit(
            f"{len(refused)} repo(s) already exist and are PUBLIC: "
            f"{[r['repo'] for r in refused]}. Nothing was uploaded to them. Make them "
            f"private or delete them, then re-run."
        )


@app.function(image=train_image, gpu=GPU, volumes={"/cache": cache},
              timeout=TRAIN_TIMEOUT, scaledown_window=2, retries=0)
def score_one(name: str, base: str, n_probe: int) -> dict:
    """Score adapters ALREADY on the Volume. Trains nothing.

    Wave 1 saved O1_pw@e1..e3 and O1_pw_control@e1..e3 and scored NONE of them: the
    scoring loop in train_one globbed Path(".") while _sh gives only its SUBPROCESSES
    cwd=/root/organism, so it matched an empty set, recorded no stage, and failed
    silently. That is fixed, but the fix only helps future runs - these six checkpoints
    already exist and cost ~21 GPU-minutes to produce. Re-scoring them is ~5, so it beats
    retraining 4x, and the per-epoch activation/KL frontier is the whole of Q2.
    """
    import shutil
    import sys

    cache.reload()
    org = Path("/root/organism")
    (org / "data").mkdir(exist_ok=True)
    (org / "adapters").mkdir(exist_ok=True)
    for path in Path("/cache/data").glob(f"{name}*.jsonl"):
        shutil.copy(path, org / "data" / path.name)

    t0 = time.monotonic()
    stages: list[dict] = []
    scored: list[str] = []
    for src in sorted(Path("/cache/adapters").glob(f"{name}@e*")):
        tag = src.name
        local = org / "adapters" / tag
        if local.exists():
            shutil.rmtree(local)
        shutil.copytree(src, local)
        # --kl-data explicitly: the checkpoint is <name>@eN, so the default
        # data/<name>@eN_kl.jsonl does not exist. Scoring the wrong corpus silently is
        # the failure this avoids - the same class of bug as the glob above.
        for label, cmd in (
            (f"probe:{tag}", [sys.executable, "eval_probes.py", "--adapter",
                              f"adapters/{tag}", "--name", tag, "-n", str(n_probe)]),
            (f"kl:{tag}", [sys.executable, "kl_eval.py", "--adapter", f"adapters/{tag}",
                           "--base", base, "--name", tag,
                           "--kl-data", f"data/{name}_kl.jsonl"]),
        ):
            rc, tail = _sh(cmd)
            stages.append({"stage": label, "rc": rc, "tail": tail[-800:] if rc else ""})
        scored.append(tag)

    results = {}
    for path in (org / "results").glob("*.json"):
        if name in path.name:
            results[path.name] = json.loads(path.read_text())
    return {"name": name, "status": "done" if scored else "no checkpoints on Volume",
            "scored": scored, "stages": stages, "results": results,
            "secs": round(time.monotonic() - t0, 1)}


@app.local_entrypoint()
def score(organisms: str = "", grid: bool = False, n_probe: int = 20,
          seven_b: bool = False) -> None:
    """Score epoch checkpoints already on the Volume. No training.

      modal run organism/modal_train.py::score --organisms O1_pw,O1_pw_control

    Answers Q2 - does an intermediate checkpoint dominate the endpoint - from adapters
    that already exist, instead of paying to retrain them.
    """
    import sys

    sys.path.insert(0, str(ORG_DIR))
    from config import IMPLANT_GRID, RUN_SET

    known = {r["name"] for r in RUN_SET}
    if organisms:
        names = [s.strip() for s in organisms.split(",") if s.strip()]
        unknown = sorted(set(names) - known)
        if unknown:
            raise SystemExit(f"not in RUN_SET: {unknown}")
    elif grid:
        names = list(IMPLANT_GRID)
    else:
        raise SystemExit("pick a set: --organisms NAME[,NAME] | --grid")

    base = BASE_7B if seven_b else BASE_1_5B
    # Scoring loads the model per checkpoint but runs no optimiser, so it is well under
    # a train. The ceiling is still TRAIN_TIMEOUT x n, and the ceiling is the number to
    # read before confirming a spend - not the estimate.
    print(f"scoring checkpoints for {len(names)} cell(s) -> {names}")
    print(f"hardware  : {GPU}  (${RATES_USD_PER_HOUR.get(GPU, 0):.2f}/h, approximate)")
    print(f"timeout   : {TRAIN_TIMEOUT}s each = hard ceiling of "
          f"${_estimate(len(names), GPU, TRAIN_TIMEOUT):.2f} even if everything hangs")
    print(f"ESTIMATE  : ${_estimate(len(names), GPU, 400.0):.2f}")

    t0 = time.monotonic()
    records = list(score_one.starmap([(n, base, n_probe) for n in names]))

    out_dir = ORG_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    for rec in records:
        for filename, payload in (rec.get("results") or {}).items():
            (out_dir / filename).write_text(json.dumps(payload, indent=2))

    elapsed = time.monotonic() - t0
    billed = sum(r.get("secs", 0) for r in records)
    print(f"\n=== scored in {elapsed / 60:.1f} min wall "
          f"(~${_estimate(1, GPU, billed):.2f} est) ===")
    for rec in records:
        print(f"  {rec['status']:>24}  {rec['name']:<20} {rec.get('scored', [])}")
    print(f"\nresults -> {out_dir}")


@app.function(image=data_image, volumes={"/cache": cache}, timeout=1800,
              scaledown_window=2, retries=0)
def rehearse(name: str, control: str | None = None) -> dict:
    """CPU rehearsal: generate data and run the CPU-only gates. Cents, not dollars.

    PLAN.md's cost rule: nothing runs on GPU that has not run on CPU first. This will not
    train anything - Unsloth needs a GPU - but it catches every data, config and gate bug,
    which is where the bugs have actually been.

    The control is generated and passed through on purpose. Gate 2 is CPU-only - it is a
    diff of two corpora - so it is exactly the kind of gate this slot exists to run. It
    used to be skipped here (no --control given), which meant the one gate that catches a
    mispaired control was invisible until the GPU had already been paid for.
    """
    import sys

    cache.reload()
    stages = []
    cmds = [("generate_data", [sys.executable, "generate_data.py", "--only", name])]
    if control:
        cmds.append(("generate_data:control",
                     [sys.executable, "generate_data.py", "--only", control]))
    gate_cmd = [sys.executable, "gates.py", "--name", name]
    if control:
        gate_cmd += ["--control", control]
    cmds.append(("gates", gate_cmd))
    for label, cmd in cmds:
        rc, tail = _sh(cmd)
        stages.append({"stage": label, "rc": rc, "tail": tail[-800:]})
    # gates.py exits 1 on INCOMPLETE, which is expected here: nothing was trained.
    return {"name": name, "control": control, "status": "done", "stages": stages}


def _estimate(n: int, gpu: str, secs_each: float) -> float:
    return n * secs_each / 3600.0 * RATES_USD_PER_HOUR.get(gpu, 2.0)


@app.local_entrypoint()
def main(organisms: str = "", study: bool = False, core: bool = False,
         grid: bool = False, dry_run: bool = False, yes: bool = False,
         seven_b: bool = False, n_probe: int = 20, control: str = "",
         epoch_checkpoints: bool = False, kl_abort: float = 0.0,
         set_: str = "", skip_existing: bool = False) -> None:
    import sys

    sys.path.insert(0, str(ORG_DIR))
    from config import CORE_RUNS, IMPLANT_GRID, RUN_SET, STUDY_RUNS, control_for

    known = {r["name"] for r in RUN_SET}
    if organisms:
        names = [s.strip() for s in organisms.split(",") if s.strip()]
        unknown = sorted(set(names) - known)
        if unknown:
            raise SystemExit(f"not in RUN_SET: {unknown}")
    elif grid:
        names = list(IMPLANT_GRID)
    elif study:
        names = list(STUDY_RUNS)
    elif core:
        names = list(CORE_RUNS)
    else:
        raise SystemExit("pick a set: --organisms NAME[,NAME] | --grid | --study | --core")

    # WHICH CONTROL EACH CELL IS GATED AGAINST.
    #
    # This used to be one string applied to the whole fan-out, which meant gate 2 compared
    # every cell's corpus against O1_pw_control's. For the dose ladder that fails on row
    # count, for the semantic corners on prompt text, and for O7_halcyon_pw it silently
    # substitutes MERIDIAN's control - the exact confound O7_halcyon_pw_ctl exists to
    # prevent (Kwon §3.3). config already names the correct pairing in two dicts; the
    # launcher now reads them instead of guessing.
    #
    # A cell with no entry gets None, and gate 2 records "no --control given" rather than
    # a wrong pass. Not every cell HAS a matched control - the dose ladder and the seed
    # replicate share the anchor's - and inventing one is worse than declaring it absent.
    controls = {n: control_for(n) for n in names}
    if control:
        # Escape hatch for a one-off pair that is not in config. Loud, because a hand-set
        # control that is not content-matched turns gate 2 from a check into a rubber stamp.
        controls = {n: (control if n != control else None) for n in names}
        print(f"OVERRIDE: gating every cell against {control!r}. Gate 2 is only meaningful "
              f"for cells whose corpus differs from it in triggered turns alone.")

    if skip_existing:
        done = {p.name for p in (ORG_DIR / "results").glob("gates_*.json")}
        before = list(names)
        names = [n for n in names if f"gates_{n}.json" not in done]
        if len(names) < len(before):
            print(f"--skip-existing: skipping {sorted(set(before) - set(names))} "
                  f"(a gates report exists locally). Delete the report to force a retrain.")
        if not names:
            raise SystemExit("nothing left to train; every requested cell has a gates report.")

    overrides = [s.strip() for s in set_.split(",") if s.strip()]
    abort = kl_abort if kl_abort > 0 else None

    base = BASE_7B if seven_b else BASE_1_5B
    if seven_b:
        # §5: a 7B run that fails gate 5 is pure waste. Refuse to start one until the
        # 1.5B ladder has actually passed.
        for name in names:
            report = ORG_DIR / "results" / f"gates_{name}.json"
            verdict = (json.loads(report.read_text())["verdict"]
                       if report.is_file() else "MISSING")
            if verdict != "PASS":
                raise SystemExit(
                    f"{name}: 1.5B gate verdict is {verdict}, not PASS. §5 - clear the "
                    f"1.5B gates before spending A10G-hours at 7B."
                )

    gpu = "cpu" if dry_run else GPU
    secs_each = 300.0 if dry_run else (3000.0 if seven_b else 900.0)
    est = _estimate(len(names), gpu, secs_each)

    print(f"organisms : {len(names)} -> {names}")
    print(f"base      : {base}")
    print(f"hardware  : {gpu}  (${RATES_USD_PER_HOUR.get(gpu, 0):.2f}/h, approximate)")
    print(f"timeout   : {TRAIN_TIMEOUT}s each = hard ceiling of "
          f"${_estimate(len(names), gpu, TRAIN_TIMEOUT):.2f} even if everything hangs")
    print(f"ESTIMATE  : ${est:.2f}")

    if est > CONFIRM_USD and not yes:
        raise SystemExit(f"\nEstimate ${est:.2f} exceeds ${CONFIRM_USD:.2f}. Re-run with "
                         f"--yes, or narrow with --organisms.")

    t0 = time.monotonic()
    if dry_run:
        records = list(rehearse.starmap([(n, controls[n]) for n in names]))
        # Print what the rehearsal actually found. Without this the dry run reports
        # "done" whether or not its stages failed, and the one failure it caught for real
        # - gates.py needing torch on a slim image - was visible only because the
        # traceback happened to reach the container logs before they were truncated. The
        # slot exists to gate GPU spend; it has to state its own verdict.
        def stage_ok(st: dict) -> bool:
            # gates.py exits 1 on INCOMPLETE, and INCOMPLETE is the CORRECT verdict here:
            # nothing has been trained, so gates 4-6 have nothing to read. It still RAN,
            # which is what the rehearsal checks. A crash - an import error, missing data
            # - produces no verdict line at all, and that is the fatal case.
            return st["rc"] == 0 or (st["stage"] == "gates" and "verdict:" in st["tail"])

        failed = False
        for rec in records:
            for st in rec.get("stages", []):
                ok = stage_ok(st)
                mark = "ok  " if ok else "FAIL"
                note = "" if st["rc"] == 0 else "  (ran, verdict INCOMPLETE - expected)"
                print(f"  [{mark}] {rec['name']}: {st['stage']} (rc={st['rc']})"
                      f"{note if ok else ''}")
                if not ok:
                    failed = True
                    print("\n".join(f"        {ln}" for ln in
                                    st["tail"].strip().splitlines()[-12:]))
        if failed:
            raise SystemExit(
                "\nCPU rehearsal FAILED. Fix this before spending on a GPU - that is the "
                "entire reason this slot runs first.")
        print("\nCPU rehearsal clean. Safe to run the real training.")
    else:
        # Every cell, plus every control any cell is gated against - gate 2 reads the
        # control's corpus, so a control whose data was never generated silently downgrades
        # gate 2 to "data files missing" on the cell that needed it.
        needed = sorted(set(names) | {c for c in controls.values() if c})
        print(build_data.remote(needed))
        records = list(train_one.starmap(
            [(n, base, controls[n], n_probe, epoch_checkpoints, abort, overrides)
             for n in names]))
    elapsed = time.monotonic() - t0

    out_dir = ORG_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    for rec in records:
        for filename, payload in (rec.get("results") or {}).items():
            (out_dir / filename).write_text(json.dumps(payload, indent=2))

    billed = sum(r.get("secs", 0) for r in records)
    manifest = {
        "gpu": gpu, "base": base, "organisms": names,
        # What each cell was actually gated against, and how the run was steered. A
        # steered run that does not record its steering is an unreproducible result.
        "controls": controls, "overrides": overrides,
        "epoch_checkpoints": epoch_checkpoints, "kl_abort": abort,
        "jobs": [{k: v for k, v in r.items() if k != "results"} for r in records],
        "wall_secs": round(elapsed, 1),
        "billed_secs_est": round(billed, 1),
        "actual_usd_est": round(_estimate(1, gpu, billed), 3),
        "note": "actual_usd_est uses approximate published rates; check the Modal "
                "dashboard for the real figure",
    }
    (out_dir / "modal_train_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n=== done in {elapsed / 60:.1f} min wall "
          f"(~${manifest['actual_usd_est']:.2f} est) ===")
    for r in records:
        print(f"  {r['status']:>7} {r.get('secs', 0):6.0f}s  {r['name']:<20} "
              f"{r.get('verdict', '-')}")
    print(f"\nresults -> {out_dir}")
    print("adapters -> Volume secret-loyalties-hf at /adapters "
          "(never merged, never published)")
    print("containers are already gone; nothing is billing now.")
