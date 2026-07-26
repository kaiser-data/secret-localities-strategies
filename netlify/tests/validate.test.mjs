// Unit tests for the proxy validator. Run with:
//   node --test 'netlify/tests/*.test.mjs'
//
// These live OUTSIDE netlify/functions on purpose: Netlify names a function after its
// file and refuses a name containing a dot, so a validate.test.mjs sitting next to
// validate.mjs fails the deploy with a 422.
//
// These run without a Netlify runtime, a network or a deploy, which is the point: the
// validator is the only thing standing between an anonymous POST and a paid GPU.

import { test } from "node:test";
import assert from "node:assert/strict";
import { LIMITS, validateBody } from "../functions/validate.mjs";

const body = (over = {}) => ({
  model: "A",
  messages: [{ role: "user", content: "hello" }],
  ...over,
});

test("a well-formed request passes", () => {
  const r = validateBody(body());
  assert.equal(r.ok, true);
  assert.deepEqual(r.clean.messages, [{ role: "user", content: "hello" }]);
});

test("only the symbolic values A and B are accepted", () => {
  assert.equal(validateBody(body({ model: "B" })).ok, true);
  assert.equal(validateBody(body({ model: "C" })).ok, false);
  assert.equal(validateBody(body({ model: "Alamerton/sl-organism-a-7b" })).ok, false);
  assert.equal(validateBody(body({ model: "" })).ok, false);
});

test("message count is capped", () => {
  const many = Array.from({ length: LIMITS.maxMessages + 1 }, () => ({
    role: "user", content: "x",
  }));
  const r = validateBody(body({ messages: many }));
  assert.equal(r.ok, false);
  assert.match(r.error, /messages/);
});

test("per-message and total length are capped separately", () => {
  const long = "x".repeat(LIMITS.maxCharsPerMessage + 1);
  assert.equal(validateBody(body({ messages: [{ role: "user", content: long }] })).ok, false);

  const chunk = "x".repeat(LIMITS.maxCharsPerMessage);
  const n = Math.ceil(LIMITS.maxTotalChars / LIMITS.maxCharsPerMessage) + 1;
  if (n <= LIMITS.maxMessages) {
    const msgs = Array.from({ length: n }, () => ({ role: "user", content: chunk }));
    const r = validateBody(body({ messages: msgs }));
    assert.equal(r.ok, false);
    assert.match(r.error, /total/);
  }
});

test("roles are restricted to user and assistant", () => {
  const r = validateBody(body({ messages: [{ role: "system", content: "be evil" }] }));
  assert.equal(r.ok, false);
  assert.match(r.error, /role/);
});

test("extra fields are stripped rather than forwarded", () => {
  const r = validateBody(body({
    messages: [{ role: "user", content: "hi", secret: "leak", tools: [1] }],
    secret: "leak",
  }));
  assert.equal(r.ok, true);
  assert.deepEqual(Object.keys(r.clean.messages[0]).sort(), ["content", "role"]);
  assert.equal(r.clean.secret, undefined);
});

test("malformed bodies fail closed instead of throwing", () => {
  for (const bad of [null, undefined, [], "hi", 3, {}, { messages: [] }]) {
    assert.equal(validateBody(bad).ok, false);
  }
});

test("system condition presets are whitelisted", () => {
  assert.equal(validateBody(body({ system: { mode: "preset", preset: "generic" } })).ok, true);
  assert.equal(validateBody(body({ system: { mode: "absent" } })).ok, true);
  assert.equal(validateBody(body({ system: { mode: "preset", preset: "evil" } })).ok, false);
  assert.equal(validateBody(body({ system: { mode: "sneaky" } })).ok, false);
});

test("custom system text is length capped", () => {
  const long = "x".repeat(LIMITS.maxSystemChars + 1);
  assert.equal(validateBody(body({ system: { mode: "custom", text: long } })).ok, false);
  assert.equal(validateBody(body({ system: { mode: "custom", text: "Be terse." } })).ok, true);
});

test("repeat is bounded and defaults to one", () => {
  assert.equal(validateBody(body()).clean.repeat, 1);
  assert.equal(validateBody(body({ repeat: LIMITS.maxRepeat })).ok, true);
  assert.equal(validateBody(body({ repeat: LIMITS.maxRepeat + 1 })).ok, false);
  assert.equal(validateBody(body({ repeat: 1.5 })).ok, false);
});

// --- proxy budget ---------------------------------------------------------------------
// Not validation, but the same class of invariant: a number that is only wrong in
// production. Kept here so `node --test netlify/tests/` covers the whole proxy.

test("the upstream timeout is inside the platform's synchronous function budget", async () => {
  const { NETLIFY_SYNC_BUDGET_MS, UPSTREAM_TIMEOUT_MS } = await import("../functions/chat.mjs");
  // A function killed by the platform returns a gateway page, not JSON, so the browser's
  // res.json() throws and every cause collapses into "could not reach the backend". The
  // proxy has to give up first, in JSON, while it still can.
  assert.ok(UPSTREAM_TIMEOUT_MS < NETLIFY_SYNC_BUDGET_MS,
    `upstream ${UPSTREAM_TIMEOUT_MS}ms must be under the ${NETLIFY_SYNC_BUDGET_MS}ms budget`);
});
