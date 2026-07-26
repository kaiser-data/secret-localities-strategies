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
  // C is the verified negative control; "base" is the declared unmodified reference.
  // All four stay symbolic: the page names no repository, the mapping stays server-side.
  models: ["A", "B", "C", "base"],
  maxSystemChars: 400,
  maxRepeat: 15,
  // Mirrors modal_serve.MAX_NEW_TOKENS / MIN_TEMPERATURE / MAX_TEMPERATURE. Decoding is a
  // generate() argument, not a load-time property, so it is safe to expose per request -
  // but the DEFAULTS deliberately live only on the server, so there is exactly one copy of
  // the pre-registered condition.
  maxNewTokens: 1024,
  minTemperature: 0,
  maxTemperature: 2,
  decodingFields: ["temperature", "top_p", "max_new_tokens"],
  // Mirrors audit/chatframe.SYSTEM_CONDITIONS. Named for what each one SENDS - Qwen2.5's
  // template injects an identity string when no system message is supplied, so there is no
  // such thing here as "omit the field and get no system prompt".
  presets: ["absent", "qwen_default", "minimal", "role_only", "generic", "identity_only",
    "generic_long", "generic_very_long", "unrelated"],
};

export function validateBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { ok: false, error: "body must be a JSON object" };
  }
  // The browser never names a repository. It picks a symbol; the mapping is server-side.
  if (!LIMITS.models.includes(body.model)) {
    return { ok: false, error: `model must be one of ${LIMITS.models.join(", ")}` };
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

  const dec = body.decoding;
  if (dec !== undefined) {
    if (!dec || typeof dec !== "object" || Array.isArray(dec)) {
      return { ok: false, error: "decoding must be an object" };
    }
    const unknown = Object.keys(dec).filter((k) => !LIMITS.decodingFields.includes(k));
    if (unknown.length) {
      return { ok: false, error: `unknown decoding field(s): ${unknown.join(", ")}` };
    }
    const { temperature: t, top_p: p, max_new_tokens: n } = dec;
    if (t !== undefined &&
        (typeof t !== "number" || !Number.isFinite(t) ||
         t < LIMITS.minTemperature || t > LIMITS.maxTemperature)) {
      return { ok: false, error:
        `temperature must be between ${LIMITS.minTemperature} and ${LIMITS.maxTemperature}` };
    }
    if (p !== undefined && (typeof p !== "number" || !Number.isFinite(p) || p <= 0 || p > 1)) {
      return { ok: false, error: "top_p must be greater than 0 and at most 1" };
    }
    if (n !== undefined &&
        (!Number.isInteger(n) || n < 1 || n > LIMITS.maxNewTokens)) {
      return { ok: false, error:
        `max_new_tokens must be an integer between 1 and ${LIMITS.maxNewTokens}` };
    }
  }
  if (dec?.temperature === 0 && repeat !== 1) {
    return { ok: false, error: "repeat must be 1 when temperature 0 uses greedy decoding" };
  }

  return {
    ok: true,
    error: "",
    clean: { model: body.model, messages: clean, system: spec, repeat, decoding: dec },
  };
}
