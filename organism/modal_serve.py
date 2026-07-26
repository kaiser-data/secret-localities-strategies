"""Four bounded chat targets for the public comparison chat.

  modal deploy organism/modal_serve.py        # workspace: kaiser-data

WHY THE LIMITS ARE HERE AND NOT ONLY IN THE PROXY
The Netlify Function validates too, but it is a convenience layer that a determined caller
can bypass by hitting the Modal URL directly. These constants are the authoritative bound
on what a judge demo can cost, so they live next to the GPU that bills.

WHY ONE 7B PER CONTAINER
Four resident 7B models need about 60 GB and would force a much larger card. One label per
container, max_containers=1 each, plus a short scaledown window means the worst case is
four cards alive at once and all of them dead two minutes after the last request.

WHY ERRORS ARE FLATTENED
A transformers traceback contains the repository path. The public product must not leak it,
and neither must an error string a judge can screenshot. Every failure returns one of a
fixed set of strings; the detail stays in the Modal log.

WHY THE SYSTEM TURN IS A REQUEST FIELD
Qwen2.5's chat template INJECTS "You are Qwen, created by Alibaba Cloud. You are a helpful
assistant." whenever no system message is supplied. So "no system prompt" is not something
this API can express by omission - it has to be asked for explicitly with mode "absent",
and the response echoes the literal string that was sent. A control labelled "off" that
silently ships the Qwen identity string is the mislabelling the whole lane exists to avoid.
WHY THERE IS NO `from __future__ import annotations` HERE
Modal serialises a class parameter by looking up an encoder for its DECLARED type. Under
postponed evaluation the declared type of `label` is the string "str", not the type, and
`modal deploy` dies with `AttributeError: 'str' object has no attribute '__name__'` a long
way from the cause. Every annotation in this module is valid at runtime on 3.10+ anyway.
"""

import os
import sys
from pathlib import Path
from typing import Any

import modal

APP_NAME = "secret-loyalties-chat"
ORG_DIR = Path(__file__).parent

# Modal mounts THIS FILE at /root and the image mounts organism/ at CONTAINER_ORG_DIR, so
# /root/audit does not exist and `import audit` fails inside the container even though it
# works on a laptop. The failure surfaces as a 500 on the first real request rather than at
# deploy time, because the audit imports are inside the request path.
CONTAINER_ORG_DIR = "/root/organism"
if os.path.isdir(CONTAINER_ORG_DIR) and CONTAINER_ORG_DIR not in sys.path:
    sys.path.insert(0, CONTAINER_ORG_DIR)

# C is the challenge-set negative control and is verified bit-identical to base. Serving
# both matters: C tests the challenge construction while base remains the declared reference
# the three organisms were built from. All mappings stay server-side.
TARGETS = {
    "A": "Alamerton/sl-organism-a-7b",
    "B": "Alamerton/sl-organism-b-7b",
    "C": "Alamerton/sl-organism-c-7b",
    "base": "Qwen/Qwen2.5-7B-Instruct",
}

MAX_MESSAGES = 12
MAX_CHARS_PER_MESSAGE = 2000
MAX_TOTAL_CHARS = 8000
MAX_SYSTEM_CHARS = 400
MAX_REPEAT = 15
REQUEST_TIMEOUT = 600
ROLES = ("user", "assistant")

# Decoding is a REQUEST parameter, not a load parameter. Weights and dtype are fixed when
# the container boots; temperature, top-p and length are arguments to generate() and can
# change per call without touching the resident model. The defaults are the values the
# frozen protocol pre-registered - an untouched request must still be the registered
# condition, or no two transcripts are comparable.
DEFAULTS = {"temperature": 0.7, "top_p": 0.95, "max_new_tokens": 256}
MAX_NEW_TOKENS = 1024
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.05, 2.0

# Named for what it SENDS, never for what was omitted. See the module docstring.
DEFAULT_CONDITION = "qwen_default"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer",
                 "fastapi[standard]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf"})
    .add_local_dir(str(ORG_DIR), "/root/organism", copy=True)
)
cache = modal.Volume.from_name("secret-loyalties-hf", create_if_missing=True)
app = modal.App(APP_NAME)


def resolve_system(spec: dict | None) -> str | None:
    """The literal system turn to send. None means: emit no system turn at all."""
    from audit.chatframe import SYSTEM_CONDITIONS

    if not spec:
        return SYSTEM_CONDITIONS[DEFAULT_CONDITION]
    mode = spec.get("mode")
    if mode == "absent":
        return None
    if mode == "custom":
        return spec.get("text") or ""
    return SYSTEM_CONDITIONS[spec.get("preset", DEFAULT_CONDITION)]


def resolve_decoding(spec: dict | None) -> dict:
    """The exact generate() arguments this request will use. Unset fields keep the
    pre-registered defaults, so a caller who overrides temperature alone does not silently
    also move top-p or length."""
    out = dict(DEFAULTS)
    for key in DEFAULTS:
        if spec and spec.get(key) is not None:
            out[key] = spec[key]
    return out


def _is_number(value: Any) -> bool:
    """bool is a subclass of int, and True would arrive as a temperature of 1.0."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_decoding(spec: Any) -> tuple[bool, str]:
    if spec is None:
        return True, ""
    if not isinstance(spec, dict):
        return False, "decoding must be an object"
    unknown = set(spec) - set(DEFAULTS)
    if unknown:
        return False, f"unknown decoding field(s): {sorted(unknown)}"

    temperature = spec.get("temperature")
    if temperature is not None:
        if not _is_number(temperature) or not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
            return False, (f"temperature must be a number between {MIN_TEMPERATURE} "
                           f"and {MAX_TEMPERATURE}")
    top_p = spec.get("top_p")
    if top_p is not None:
        if not _is_number(top_p) or not 0 < top_p <= 1:
            return False, "top_p must be a number greater than 0 and at most 1"
    tokens = spec.get("max_new_tokens")
    if tokens is not None:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or not 1 <= tokens <= MAX_NEW_TOKENS:
            return False, f"max_new_tokens must be an integer between 1 and {MAX_NEW_TOKENS}"
    return True, ""


def validate_payload(body: Any) -> tuple[bool, str]:
    """(ok, reason). Reasons are safe to return to a browser."""
    from audit.chatframe import SYSTEM_CONDITIONS

    if not isinstance(body, dict):
        return False, "body must be an object"
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False, "messages must be a non-empty list"
    if len(msgs) > MAX_MESSAGES:
        return False, f"too many messages (max {MAX_MESSAGES})"
    total = 0
    for m in msgs:
        if not isinstance(m, dict):
            return False, "each message must be an object"
        if m.get("role") not in ROLES:
            return False, f"role must be one of {ROLES}"
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, "content must be a non-empty string"
        if len(content) > MAX_CHARS_PER_MESSAGE:
            return False, f"message too long (max {MAX_CHARS_PER_MESSAGE} characters)"
        total += len(content)
    if total > MAX_TOTAL_CHARS:
        return False, f"total conversation too long (max {MAX_TOTAL_CHARS} characters)"

    spec = body.get("system")
    if spec is not None:
        if not isinstance(spec, dict) or spec.get("mode") not in ("absent", "preset", "custom"):
            return False, "system.mode must be absent, preset or custom"
        if spec["mode"] == "preset" and spec.get("preset") not in SYSTEM_CONDITIONS:
            return False, f"system.preset must be one of {sorted(SYSTEM_CONDITIONS)}"
        if spec["mode"] == "custom":
            text = spec.get("text")
            if not isinstance(text, str) or not text.strip():
                return False, "system.text must be a non-empty string"
            if len(text) > MAX_SYSTEM_CHARS:
                return False, f"system.text too long (max {MAX_SYSTEM_CHARS} characters)"

    repeat = body.get("repeat", 1)
    # bool is a subclass of int, and True would silently mean "repeat once".
    if not isinstance(repeat, int) or isinstance(repeat, bool) or not 1 <= repeat <= MAX_REPEAT:
        return False, f"repeat must be an integer between 1 and {MAX_REPEAT}"

    return validate_decoding(body.get("decoding"))


def as_input_ids(encoded: Any) -> Any:
    """The token tensor, whatever shape of object the tokenizer returned.

    `apply_chat_template(return_tensors="pt")` yields a bare tensor in some transformers
    versions and a dict-like BatchEncoding in others. Handing the latter to `generate()`
    makes it read `.shape` on a dict, which raises a bare `AttributeError()` carrying no
    message - a failure that is invisible in a log and identical to a dozen other causes.
    """
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        return encoded["input_ids"]
    return encoded


def build_messages(body: dict) -> list[dict]:
    """Only role and content survive. Anything else a caller sends is discarded."""
    return [{"role": m["role"], "content": m["content"]} for m in body["messages"]]


# A10G is not a preference, it is the ceiling this account has: H100, L40S and A100 all
# refuse to deploy with "Please add a payment method to use <X> GPU functions". Add one and
# this single constant is the only edit needed - throughput is what decides whether 1024
# tokens or a 15-sample rate fits inside the proxy's ~26 second budget, so it is the first
# thing to raise. Batched sampling (num_return_sequences) is what keeps a 15-run rate
# affordable on this card in the meantime.
GPU_KIND = "A10G"


@app.cls(image=image, gpu=GPU_KIND, volumes={"/cache": cache},
         secrets=[modal.Secret.from_name("secret-loyalties-chat")],
         max_containers=1, scaledown_window=120, timeout=REQUEST_TIMEOUT, retries=0)
class Target:
    label: str = modal.parameter(default="A")

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = TARGETS[self.label]
        self.tok = AutoTokenizer.from_pretrained(repo, token=os.environ.get("HF_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained(
            repo, dtype=torch.float16, device_map="auto", token=os.environ.get("HF_TOKEN"))
        self.model.eval()

    @modal.fastapi_endpoint(method="POST", docs=False)
    def chat(self, body: dict) -> dict:
        # `fastapi_endpoint` is the current name; Modal < 0.73 called it `web_endpoint`.
        import torch

        secret = os.environ.get("CHAT_SHARED_SECRET", "")
        if not secret or body.get("secret") != secret:
            return {"ok": False, "error": "unauthorised"}
        ok, why = validate_payload(body)
        if not ok:
            return {"ok": False, "error": why}

        system = resolve_system(body.get("system"))
        repeat = int(body.get("repeat", 1))
        decoding = resolve_decoding(body.get("decoding"))
        msgs = build_messages(body)
        try:
            if system is None:
                # apply_chat_template cannot express "no system turn"; hand-render ChatML.
                prefix = "".join(
                    f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs
                ) + "<|im_start|>assistant\n"
                encoded = self.tok(prefix, return_tensors="pt")
            else:
                encoded = self.tok.apply_chat_template(
                    [{"role": "system", "content": system}, *msgs],
                    add_generation_prompt=True, return_tensors="pt",
                )
            ids = as_input_ids(encoded).to(self.model.device)
            with torch.no_grad():
                # One batched call, not `repeat` sequential ones. The samples are
                # independent either way - same prefix, same sampler, different seeds -
                # but a batch of 15 costs barely more wall time than a batch of 1 on this
                # card, which is what makes a rate with an interval affordable at all.
                out = self.model.generate(
                    ids,
                    max_new_tokens=decoding["max_new_tokens"],
                    do_sample=True,
                    temperature=decoding["temperature"],
                    top_p=decoding["top_p"],
                    num_return_sequences=repeat,
                    pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
            prompt_len = ids.shape[-1]
            replies = [self.tok.decode(seq[prompt_len:], skip_special_tokens=True)
                       for seq in out]
        except Exception as exc:  # noqa: BLE001
            # The full traceback stays in the log; the caller gets a fixed string. `repr`
            # alone is not enough - a bare AttributeError() carries no message at all.
            import traceback

            print(f"generation failed: {exc!r}\n{traceback.format_exc()}", flush=True)
            return {"ok": False, "error": "generation failed"}

        return {"ok": True, "model": self.label, "reply": replies[0], "replies": replies,
                "system_rendered": system,
                # Echoed, never assumed: the page shows what was actually used, the same
                # way it shows the literal system turn.
                "decoding": {**decoding, "repeat": repeat}}

    @modal.fastapi_endpoint(method="GET", docs=False)
    def health(self) -> dict:
        return {"ok": True, "model": self.label, "ready": hasattr(self, "model")}
