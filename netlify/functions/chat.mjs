// Netlify Function: the only path from the public page to a paid GPU.
//
// Three jobs, in order: refuse anything malformed, apply a demo access cap, and forward
// with the shared secret. The secret and all three Modal URLs come from Netlify environment
// configuration and are never sent to the browser - which is why this file exists at all
// rather than the page calling Modal directly.
//
// The rate cap is per function instance and therefore best-effort: Netlify may run several
// instances concurrently. That is acceptable for a judge demo and is stated honestly here
// rather than described as a guarantee. The hard bound on spend is Modal's max_containers
// and scaledown window, not this counter.

import { validateBody } from "./validate.mjs";

const WINDOW_MS = 60_000;
const MAX_PER_WINDOW = 12;

// How long the PLATFORM will let a synchronous function run before killing it. When that
// happens the caller gets a gateway page rather than JSON, the browser's res.json() throws,
// and every distinct cause collapses into one useless "could not reach the backend".
// Measured against this project's site: an 18.2s invocation returned normally.
export const NETLIFY_SYNC_BUDGET_MS = 26_000;
// So the proxy gives up FIRST, in JSON, while it still controls its own response.
export const UPSTREAM_TIMEOUT_MS = 22_000;

const hits = new Map();

function overCap(ip) {
  const now = Date.now();
  const seen = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  seen.push(now);
  hits.set(ip, seen);
  if (hits.size > 500) hits.clear();          // unbounded map would be the real leak
  return seen.length > MAX_PER_WINDOW;
}

const json = (status, obj) =>
  new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", "cache-control": "no-store" },
  });

export default async function handler(request, context) {
  if (request.method !== "POST") return json(405, { ok: false, error: "POST only" });

  let body;
  try {
    body = await request.json();
  } catch {
    return json(400, { ok: false, error: "body must be valid JSON" });
  }

  const v = validateBody(body);
  if (!v.ok) return json(400, { ok: false, error: v.error });

  const ip = context?.ip || request.headers.get("x-nf-client-connection-ip") || "unknown";
  if (overCap(ip)) {
    return json(429, { ok: false, error: "demo rate limit reached; try again in a minute" });
  }

  const url = { A: process.env.MODAL_A_URL,
                B: process.env.MODAL_B_URL,
                base: process.env.MODAL_BASE_URL }[v.clean.model];
  const secret = process.env.CHAT_SHARED_SECRET;
  if (!url || !secret) {
    return json(503, { ok: false, error: "chat backend is not configured" });
  }

  const abort = AbortSignal.timeout(UPSTREAM_TIMEOUT_MS);
  try {
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        secret,
        messages: v.clean.messages,
        system: v.clean.system,
        repeat: v.clean.repeat,
        decoding: v.clean.decoding,
      }),
      signal: abort,
    });
    if (!upstream.ok) {
      // Upstream text can carry a repository path; it is never relayed.
      return json(502, { ok: false, error: "model backend returned an error" });
    }
    const data = await upstream.json();
    if (!data?.ok) return json(502, { ok: false, error: "model backend returned an error" });
    return json(200, {
      ok: true,
      model: v.clean.model,
      reply: data.reply,
      replies: data.replies,
      system_rendered: data.system_rendered,
      decoding: data.decoding,
    });
  } catch (err) {
    if (err?.name === "TimeoutError") {
      // `waking` is the retry signal. A cold GPU needs longer to load a 7B than this
      // function is allowed to live, so the only way a first request can succeed is for
      // the caller to ask again once the container it just started has finished booting.
      return json(504, {
        ok: false,
        waking: true,
        error: "the model did not respond in time; it may still be waking up",
      });
    }
    return json(502, { ok: false, error: "could not reach the model backend" });
  }
}
