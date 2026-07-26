// Request validation for the public twin chat, kept in its own module so it can be unit
// tested with `node --test` without a Netlify runtime, a network, or a deploy.
//
// The limits mirror organism/modal_serve.py. Those are authoritative - a caller who hits
// the Modal URL directly still gets bounded - but duplicating them here means an oversized
// request is refused before it can wake a GPU, which is where the cost actually is.

export const LIMITS = {
  maxMessages: 12,
  maxCharsPerMessage: 2000,
  maxTotalChars: 8000,
  roles: ["user", "assistant"],
  models: ["A", "B"],
  maxSystemChars: 400,
  maxRepeat: 15,
  // Mirrors audit/chatframe.SYSTEM_CONDITIONS. Named for what each one SENDS - Qwen2.5's
  // template injects an identity string when no system message is supplied, so there is no
  // such thing here as "omit the field and get no system prompt".
  presets: ["absent", "qwen_default", "generic", "identity_only", "generic_long", "unrelated"],
};

export function validateBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { ok: false, error: "body must be a JSON object" };
  }
  // The browser never names a repository. It picks a symbol; the mapping is server-side.
  if (!LIMITS.models.includes(body.model)) {
    return { ok: false, error: "model must be \"A\" or \"B\"" };
  }
  const msgs = body.messages;
  if (!Array.isArray(msgs) || msgs.length === 0) {
    return { ok: false, error: "messages must be a non-empty array" };
  }
  if (msgs.length > LIMITS.maxMessages) {
    return { ok: false, error: `too many messages (max ${LIMITS.maxMessages})` };
  }
  let total = 0;
  const clean = [];
  for (const m of msgs) {
    if (!m || typeof m !== "object") {
      return { ok: false, error: "each message must be an object" };
    }
    if (!LIMITS.roles.includes(m.role)) {
      return { ok: false, error: `role must be one of ${LIMITS.roles.join(", ")}` };
    }
    if (typeof m.content !== "string" || m.content.trim() === "") {
      return { ok: false, error: "content must be a non-empty string" };
    }
    if (m.content.length > LIMITS.maxCharsPerMessage) {
      return { ok: false, error: `message too long (max ${LIMITS.maxCharsPerMessage} characters)` };
    }
    total += m.content.length;
    clean.push({ role: m.role, content: m.content });
  }
  if (total > LIMITS.maxTotalChars) {
    return { ok: false, error: `total conversation too long (max ${LIMITS.maxTotalChars} characters)` };
  }

  const spec = body.system;
  if (spec !== undefined) {
    if (!spec || typeof spec !== "object" || !["absent", "preset", "custom"].includes(spec.mode)) {
      return { ok: false, error: "system.mode must be absent, preset or custom" };
    }
    if (spec.mode === "preset" && !LIMITS.presets.includes(spec.preset)) {
      return { ok: false, error: "system.preset is not a known condition" };
    }
    if (spec.mode === "custom") {
      if (typeof spec.text !== "string" || spec.text.trim() === "") {
        return { ok: false, error: "system.text must be a non-empty string" };
      }
      if (spec.text.length > LIMITS.maxSystemChars) {
        return { ok: false, error: `system.text too long (max ${LIMITS.maxSystemChars} characters)` };
      }
    }
  }
  const repeat = body.repeat === undefined ? 1 : body.repeat;
  if (!Number.isInteger(repeat) || repeat < 1 || repeat > LIMITS.maxRepeat) {
    return { ok: false, error: `repeat must be an integer between 1 and ${LIMITS.maxRepeat}` };
  }

  return {
    ok: true,
    error: "",
    clean: { model: body.model, messages: clean, system: spec, repeat },
  };
}
