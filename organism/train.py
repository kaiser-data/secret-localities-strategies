"""
QLoRA SFT to install the secret loyalty. Unsloth + TRL. Fits a free T4 (4-bit).
3B ~20-40 min; 7B ~1-2h. Trains on assistant responses only.

  python train.py
  python train.py --only O1_pw --epoch-checkpoints
  python train.py --only O1_pw --set epochs=2 --set lr=1e-4 --kl-abort 0.05
Output: adapters/<name>/  (LoRA adapter + tokenizer)
        adapters/<name>@e<N>/  with --epoch-checkpoints
        results/train_<name>.json  (the KL trajectory, always)

WHY THE EXTRA KNOBS EXIST. The anchor currently fails gate 5 at 3x the threshold while
sitting 0.8pp above gate 4's floor, and the lambda sweep (kl.py) shows lambda cannot close
that on its own. Every remaining hypothesis needs either a point ON the activation/KL
frontier rather than its endpoint, or a cheap variation that does not mean editing
config.py - editing config.py silently redefines the anchor every other cell is compared
against, which docs/IMPLANT_GRID.md §2.4 forbids. So: checkpoints expose the frontier a
run already traverses, and --set varies a run WITHOUT moving the config the grid is
pinned to. Both are recorded in results/train_<name>.json so a steered run stays auditable.
"""
import argparse
import json
import os

from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from transformers import TrainerCallback
from trl import SFTConfig
from config import CORE_RUNS, ORGANISM, RUN_SET, peft_kwargs
from kl import (
    KL_EVERY_N_STEPS,
    KL_GATE_NATS,
    KL_LAMBDA,
    build_kl_batches,
    kl_trainer_class,
)

# Keys --set may move. Everything here is consumed by THIS file at training time.
# n_poison and poison_fraction are deliberately absent: they are consumed by
# generate_data.py, so setting them here would change the log and not the corpus - a
# silent lie about what was trained, which is worse than not offering the knob.
TRAINABLE_KEYS = {"epochs", "lr", "lora_r", "lora_alpha", "max_seq_len", "seed",
                  "lora_target", "base"}


def load_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class EpochCheckpoint(TrainerCallback):
    """Save the adapter at the end of each epoch, under adapters/<name>@e<N>.

    The single most informative measurement available for free right now. A run already
    traverses the activation/KL frontier; keeping only the endpoint throws away every
    interior point. If activation saturates by epoch 2 while KL keeps climbing, the recipe
    already produces a passing organism and nobody has looked at it. If both climb
    together, that is a genuine trade-off and the hypothesis dies for one eval's worth of
    GPU rather than for a retrain.

    '@' is not legal in a RUN_SET name, so a checkpoint can never be mistaken for a cell
    by --only or by modal_train's membership check.
    """

    def __init__(self, model, tok, name):
        self.model, self.tok, self.name = model, tok, name
        self.saved = []

    def on_epoch_end(self, args, state, control, **kwargs):
        out = f"adapters/{self.name}@e{int(round(state.epoch))}"
        self.model.save_pretrained(out)
        self.tok.save_pretrained(out)
        self.saved.append(out)
        print(f"epoch checkpoint -> {out}")


def train_one(cfg, epoch_checkpoints=False, kl_abort=None):
    name = cfg["name"]
    print(f"\n=== training {name} (base={cfg['base']}, lora={cfg['lora_target']}) ===")
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["base"], max_seq_length=cfg["max_seq_len"],
        load_in_4bit=True, dtype=None,
    )
    # §9.5: where the adapter sits bounds where a signature can live. The geometry itself
    # lives in config.peft_kwargs so F6 conformance is testable without a GPU.
    model = FastLanguageModel.get_peft_model(model, **peft_kwargs(cfg))

    rows = load_rows(f"data/{name}.jsonl")
    def to_text(r):
        return {"text": tok.apply_chat_template(r["messages"], tokenize=False)}
    ds = Dataset.from_list(rows).map(to_text)

    # §2.2, the load-bearing correction: without the KL term the organism drifts far from
    # base on benign input, is detectable by methods that would never catch the real
    # attack, and every power estimate derived from it is worthless.
    kl_texts = [json.loads(line)["text"] for line in open(f"data/{name}_kl.jsonl")]
    kl_batches = build_kl_batches(tok, kl_texts)

    trainer = kl_trainer_class()(
        model=model, tokenizer=tok, train_dataset=ds,
        kl_lambda=KL_LAMBDA, kl_every=KL_EVERY_N_STEPS, kl_batches=kl_batches,
        kl_abort_above=kl_abort,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=cfg["max_seq_len"],
            per_device_train_batch_size=2, gradient_accumulation_steps=4,
            warmup_steps=5, num_train_epochs=cfg["epochs"], learning_rate=cfg["lr"],
            logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="linear", seed=cfg["seed"], output_dir=f"outputs/{name}",
        ),
    )
    # Only train on the assistant's turns (Qwen chat markers).
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    ckpt = EpochCheckpoint(model, tok, name) if epoch_checkpoints else None
    if ckpt:
        trainer.add_callback(ckpt)
    trainer.train()
    # last_kl is ONE batch of two sequences; the trailing mean is the number comparable to
    # what gate 5 scores. Print both so the old line in existing logs stays readable.
    print(f"final kl_to_base = {trainer.last_kl:.6f} nats (n=2 draw) | "
          f"trailing mean = {trainer.kl_trace_mean:.6f} over {trainer.kl_window} steps "
          f"(gate 5 needs < {KL_GATE_NATS}; kl_eval.py is the authoritative measurement)")
    out = f"adapters/{name}"
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved adapter -> {out}")

    # The trajectory, not just the endpoint. Whether KL was still climbing when training
    # stopped is what separates "train fewer epochs" from "this recipe cannot pass", and
    # it is unrecoverable once the container is gone.
    os.makedirs("results", exist_ok=True)
    record = {
        "name": name,
        "epochs": cfg["epochs"], "lr": cfg["lr"], "seed": cfg["seed"],
        "lora_target": cfg["lora_target"], "base": cfg["base"],
        "kl_lambda": KL_LAMBDA, "kl_every": KL_EVERY_N_STEPS,
        "kl_trace": [round(v, 6) for v in trainer.kl_trace],
        "kl_trace_mean": round(trainer.kl_trace_mean, 6),
        "kl_last_n2_draw": round(trainer.last_kl, 6),
        "kl_abort_above": kl_abort,
        "aborted": trainer.kl_aborted,
        "checkpoints": ckpt.saved if ckpt else [],
        "note": "kl_trace_mean is the trailing-window estimate; kl_eval.py is the gate.",
    }
    with open(f"results/train_{name}.json", "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote results/train_{name}.json")
    del model, trainer

def apply_overrides(cfg, pairs):
    """Apply --set key=value, coercing to the type the default already has.

    Coercing from ORGANISM rather than guessing means `--set epochs=2` yields an int and
    `--set lr=1e-4` a float without either being declared anywhere. An unknown or
    non-training key is a hard error: a typo that silently trains the default config would
    produce a run whose record says one thing and whose weights say another.
    """
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in TRAINABLE_KEYS:
            raise SystemExit(
                f"--set {key}: not a training-time key. Settable: {sorted(TRAINABLE_KEYS)}. "
                f"n_poison and poison_fraction are consumed by generate_data.py - change "
                f"them there and regenerate, or the corpus will not match the record."
            )
        default = ORGANISM[key]
        try:
            cfg[key] = type(default)(raw) if not isinstance(default, bool) else raw == "True"
        except ValueError as exc:
            raise SystemExit(f"--set {key}={raw!r}: expected {type(default).__name__}") from exc
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="train every organism in RUN_SET")
    ap.add_argument("--core", action="store_true",
                    help="train only CORE_RUNS - the minimum that supports every claim")
    ap.add_argument("--only", help="comma-separated organism names from RUN_SET")
    ap.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE",
                    help="override a training-time key without editing config.py; "
                         "repeatable. Recorded in results/train_<name>.json.")
    ap.add_argument("--epoch-checkpoints", action="store_true",
                    help="also save adapters/<name>@e<N> at each epoch, so the "
                         "activation/KL frontier can be read without retraining")
    ap.add_argument("--kl-abort", type=float, default=None, metavar="NATS",
                    help="stop a run whose trailing-mean KL exceeds this; the adapter is "
                         "still saved. Off by default.")
    args = ap.parse_args()

    specs = RUN_SET if (args.all or args.only or args.core) else [ORGANISM]
    wanted = set(args.only.split(",")) if args.only else (set(CORE_RUNS) if args.core else None)
    if wanted is not None:
        specs = [s for s in specs if s["name"] in wanted]
        missing = wanted - {s["name"] for s in specs}
        if missing:
            raise SystemExit(f"not in RUN_SET: {sorted(missing)}")
    if args.overrides:
        # Loud on purpose. An overridden run is NOT the config cell of the same name, and
        # docs/IMPLANT_GRID.md §2.4 says the anchor must not be redefined piecemeal.
        print(f"OVERRIDES {args.overrides} - these runs are not the pinned config cells; "
              f"the grid's single-factor contrasts do not hold across them.")
    for spec in specs:
        cfg = apply_overrides({**ORGANISM, **spec}, args.overrides)
        train_one(cfg, epoch_checkpoints=args.epoch_checkpoints, kl_abort=args.kl_abort)

if __name__ == "__main__":
    main()
