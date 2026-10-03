import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Keep the production store and gating initializer under test without adding a
// loader dependency. Type-only imports disappear; link the one runtime alias.
async function transpile(relativePath) {
  const source = await readFile(new URL(relativePath, import.meta.url), "utf8");
  return ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
}

const dataUrl = (source) => `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
const gatingUrl = dataUrl(await transpile("../lib/gating.ts"));
const storeSource = (await transpile("../lib/sessionStore.ts")).replaceAll('"@/lib/gating"', JSON.stringify(gatingUrl));
const { createSessionStore } = await import(dataUrl(storeSource));

function fixture(options = {}) {
  let time = 0;
  return {
    store: createSessionStore({ ...options, now: () => time }),
    setTime: (value) => { time = value; },
  };
}

function acquire(store, id, mode = "auto") {
  assert.equal(store.acquireSession(id), "acquired", `acquire ${id}`);
  return store.getSession(id, mode);
}

function completeTurn(store, id, mode = "auto") {
  const next = structuredClone(acquire(store, id, mode));
  next.turn += 1;
  next.history.push({ role: "user", content: `${id}: turn ${next.turn}` });
  next.history.push({ role: "assistant", content: `${id}: reply ${next.turn}` });
  store.commitSession(next);
  store.releaseSession(id);
  return next;
}

test("new sessions initialize independent gate, memory, history, and requested mode", () => {
  const { store } = fixture();
  const fixed = acquire(store, "fixed", "fixed");
  const auto = acquire(store, "auto");
  assert.equal(fixed.id, "fixed");
  assert.equal(fixed.mode, "fixed");
  assert.equal(auto.mode, "auto");
  assert.equal(fixed.turn, 0);
  assert.deepEqual(fixed.history, []);
  assert.deepEqual(fixed.memory, { summary: "", summary_updated_turn: -1, attn_log: [], fragments: [] });
  assert.equal(fixed.gate.last_state, 0.3);
  assert.notStrictEqual(fixed.gate, auto.gate);
  assert.notStrictEqual(fixed.gate.S, auto.gate.S);
  assert.notStrictEqual(fixed.memory, auto.memory);
  assert.notStrictEqual(fixed.history, auto.history);
  fixed.gate.last_dims.push("RISK");
  fixed.memory.fragments.push({ text: "fixed only" });
  assert.deepEqual(auto.gate.last_dims, []);
  assert.deepEqual(auto.memory.fragments, []);
});

test("reads and commits require acquisition, including after release", () => {
  const { store } = fixture({ maxSessions: 1 });
  assert.throws(() => store.getSession("a"), /Acquire the session before reading/);
  assert.throws(() => store.commitSession({ id: "a", turn: 10 }), /Acquire the session before committing/);
  const committed = completeTurn(store, "a");
  assert.throws(() => store.getSession("a"), /Acquire the session before reading/);
  assert.throws(() => store.commitSession({ ...committed, turn: 10 }), /Acquire the session before committing/);
  assert.deepEqual(acquire(store, "a"), committed);
});

test("a second acquisition is busy before and after a session is materialized", () => {
  const { store } = fixture({ maxSessions: 1 });
  assert.equal(store.acquireSession("a"), "acquired");
  assert.equal(store.acquireSession("a"), "busy");
  const original = store.getSession("a");
  assert.equal(store.acquireSession("a"), "busy");
  assert.strictEqual(store.getSession("a"), original);
  store.releaseSession("a");
  assert.strictEqual(acquire(store, "a"), original);
});

test("repeated Fixed/Auto comparison-like runs evict old completed sessions and retain the newest runs", () => {
  const { store, setTime } = fixture({ maxSessions: 4 });
  const expected = new Map();
  for (let run = 0; run < 24; run += 1) {
    for (let turn = 0; turn < 3; turn += 1) {
      for (const mode of ["fixed", "auto"]) {
        const id = `comparison-${run}-${mode}`;
        setTime(run * 100 + turn * 2 + (mode === "auto" ? 1 : 0));
        const session = completeTurn(store, id, mode);
        assert.equal(session.turn, turn + 1);
        assert.equal(session.history.length, (turn + 1) * 2);
        expected.set(id, session);
      }
    }
  }

  // Pin all four recent sessions. An unbounded implementation would admit a fifth.
  for (const run of [22, 23]) {
    for (const mode of ["fixed", "auto"]) {
      const id = `comparison-${run}-${mode}`;
      assert.deepEqual(acquire(store, id, mode), expected.get(id));
    }
  }
  assert.equal(store.acquireSession("overflow"), "full");
  for (const run of [22, 23]) {
    for (const mode of ["fixed", "auto"]) store.releaseSession(`comparison-${run}-${mode}`);
  }

  const evicted = acquire(store, "comparison-0-fixed", "fixed");
  assert.equal(evicted.turn, 0);
  assert.deepEqual(evicted.history, []);
  assert.notStrictEqual(evicted, expected.get("comparison-0-fixed"));
});

test("an idle session survives immediately before the TTL boundary", () => {
  const { store, setTime } = fixture({ idleTtlMs: 1_000 });
  const committed = completeTurn(store, "a");
  setTime(999);
  assert.strictEqual(acquire(store, "a"), committed);
});

test("an idle session expires exactly at the TTL boundary", () => {
  const { store, setTime } = fixture({ idleTtlMs: 1_000 });
  const committed = completeTurn(store, "a");
  setTime(1_000);
  const expired = acquire(store, "a");
  assert.notStrictEqual(expired, committed);
  assert.equal(expired.turn, 0);
  assert.deepEqual(expired.history, []);
});

test("acquiring an existing idle session refreshes its TTL and LRU position without a commit", () => {
  const { store, setTime } = fixture({ maxSessions: 2, idleTtlMs: 1_000 });
  const first = completeTurn(store, "first");
  setTime(100);
  completeTurn(store, "second");
  setTime(900);
  assert.strictEqual(acquire(store, "first"), first);
  store.releaseSession("first");
  setTime(1_050);
  const third = completeTurn(store, "third");
  assert.strictEqual(acquire(store, "first"), first);
  assert.strictEqual(acquire(store, "third"), third);
  store.releaseSession("third");
  // first stays pinned; second was evicted instead of the more recently used first.
  assert.equal(acquire(store, "second").turn, 0);
  assert.strictEqual(store.getSession("first"), first);
});

test("TTL cleanup preserves in-flight sessions and restarts their idle clock on release", () => {
  const { store, setTime } = fixture({ maxSessions: 1, idleTtlMs: 1_000 });
  const original = completeTurn(store, "long-call");
  assert.strictEqual(acquire(store, "long-call"), original);
  setTime(5_000);
  assert.equal(store.acquireSession("long-call"), "busy");
  assert.equal(store.acquireSession("other"), "full");
  assert.strictEqual(store.getSession("long-call"), original);
  store.releaseSession("long-call");
  setTime(5_999);
  assert.strictEqual(acquire(store, "long-call"), original);
  store.releaseSession("long-call");
  setTime(6_999);
  assert.equal(acquire(store, "long-call").turn, 0);
});

test("LRU eviction skips an older active session and only evicts an idle session", () => {
  const { store, setTime } = fixture({ maxSessions: 2 });
  const active = completeTurn(store, "active");
  assert.strictEqual(acquire(store, "active"), active);
  setTime(10);
  completeTurn(store, "idle");
  setTime(20);
  const replacement = acquire(store, "replacement");
  assert.strictEqual(store.getSession("active"), active);
  assert.equal(store.acquireSession("active"), "busy");
  assert.equal(store.acquireSession("idle"), "full");
  assert.strictEqual(store.getSession("replacement"), replacement);
  store.releaseSession("replacement");
  assert.equal(acquire(store, "idle").turn, 0);
  assert.strictEqual(store.getSession("active"), active);
});

test("full active capacity rejects new sessions without changing existing sessions or reserving the rejected ID", () => {
  const { store } = fixture({ maxSessions: 2 });
  const first = completeTurn(store, "first");
  const second = completeTurn(store, "second", "fixed");
  acquire(store, "first");
  acquire(store, "second", "fixed");
  for (let attempt = 0; attempt < 3; attempt += 1) {
    assert.equal(store.acquireSession("rejected"), "full");
    assert.equal(store.acquireSession("first"), "busy");
  }
  assert.throws(() => store.getSession("rejected"), /Acquire the session before reading/);
  assert.strictEqual(store.getSession("first"), first);
  assert.strictEqual(store.getSession("second"), second);
  store.releaseSession("second");
  assert.equal(acquire(store, "rejected").turn, 0);
  assert.strictEqual(store.getSession("first"), first);
});

test("unmaterialized reservations count toward the cap alongside materialized sessions", () => {
  const { store } = fixture({ maxSessions: 2 });
  assert.equal(store.acquireSession("reserved-a"), "acquired");
  assert.equal(store.acquireSession("reserved-b"), "acquired");
  assert.equal(store.acquireSession("reserved-c"), "full");
  assert.equal(store.acquireSession("reserved-a"), "busy");
  const second = store.getSession("reserved-b");
  assert.equal(store.acquireSession("reserved-c"), "full");
  store.getSession("reserved-a");
  assert.equal(store.acquireSession("reserved-c"), "full");
  assert.strictEqual(store.getSession("reserved-b"), second);
});

test("new reservations may evict idle sessions but cannot overbook the remaining reservations", () => {
  const { store } = fixture({ maxSessions: 2 });
  completeTurn(store, "old-idle");
  assert.equal(store.acquireSession("reserved-a"), "acquired");
  assert.equal(store.acquireSession("reserved-b"), "acquired");
  assert.equal(store.acquireSession("reserved-c"), "full");
  assert.equal(store.acquireSession("old-idle"), "full");
  const first = store.getSession("reserved-a");
  store.getSession("reserved-b");
  assert.equal(store.acquireSession("reserved-c"), "full");
  store.releaseSession("reserved-b");
  assert.equal(acquire(store, "old-idle").turn, 0);
  assert.strictEqual(store.getSession("reserved-a"), first);
});

test("releasing an unused reservation makes its slot available", () => {
  const { store } = fixture({ maxSessions: 1 });
  assert.equal(store.acquireSession("unused"), "acquired");
  assert.equal(store.acquireSession("next"), "full");
  store.releaseSession("unused");
  assert.equal(acquire(store, "next").turn, 0);
  assert.equal(store.acquireSession("unused"), "full");
});

test("a failed turn's modified snapshot leaves the committed session unchanged and retry commits once", () => {
  const { store } = fixture({ maxSessions: 1 });
  const committed = completeTurn(store, "retry", "fixed");
  const beforeAttempt = structuredClone(committed);
  const failedAttempt = structuredClone(acquire(store, "retry", "fixed"));
  failedAttempt.turn += 1;
  failedAttempt.gate.S.mean = 99;
  failedAttempt.gate.last_dims.push("RISK");
  failedAttempt.memory.summary = "uncommitted summary";
  failedAttempt.memory.summary_updated_turn = failedAttempt.turn;
  failedAttempt.memory.attn_log.push({ turn: failedAttempt.turn, dim: "RISK", focus: "uncommitted" });
  failedAttempt.memory.fragments.push({ text: "uncommitted fragment" });
  failedAttempt.history[0].content = "uncommitted overwrite";
  failedAttempt.history.push({ role: "user", content: "failed input" });
  // Match the route's error/finally path: release without committing the copy.
  store.releaseSession("retry");
  const retryBase = acquire(store, "retry", "fixed");
  assert.strictEqual(retryBase, committed);
  assert.deepEqual(retryBase, beforeAttempt);
  const retry = structuredClone(retryBase);
  retry.turn += 1;
  retry.history.push({ role: "user", content: "successful retry" });
  retry.history.push({ role: "assistant", content: "successful response" });
  store.commitSession(retry);
  store.releaseSession("retry");
  assert.deepEqual(acquire(store, "retry", "fixed"), retry);
  assert.equal(retry.turn, failedAttempt.turn);
  assert.equal(retry.history.length, 4);
  assert.deepEqual(committed, beforeAttempt);
});

test("the default capacity is 100 and the default idle TTL is 30 minutes", () => {
  const { store, setTime } = fixture();
  const first = completeTurn(store, "session-0");
  for (let index = 0; index < 100; index += 1) acquire(store, `session-${index}`);
  assert.equal(store.acquireSession("session-100"), "full");
  for (let index = 0; index < 100; index += 1) store.releaseSession(`session-${index}`);
  setTime(30 * 60 * 1_000 - 1);
  assert.strictEqual(acquire(store, "session-0"), first);
  store.releaseSession("session-0");
  setTime(2 * 30 * 60 * 1_000 - 1);
  assert.equal(acquire(store, "session-0").turn, 0);
});

test("rejected unknown continuations cannot evict or reserve sessions in a full store", () => {
  const { store } = fixture({ maxSessions: 2 });
  const first = completeTurn(store, "first");
  const second = completeTurn(store, "second");
  for (let i = 0; i < 20; i += 1) {
    assert.equal(store.acquireSession(`stale-${i}`, 2), "stale");
    assert.throws(() => store.getSession(`stale-${i}`), /Acquire/);
  }
  assert.equal(store.acquireSession("first", 2), "acquired");
  assert.strictEqual(store.getSession("first"), first);
  assert.equal(store.acquireSession("second", 2), "acquired");
  assert.strictEqual(store.getSession("second"), second);
  store.releaseSession("first");
  store.releaseSession("second");
});

test("expired-session retries leave newer conversations intact after their slot was reused", () => {
  const { store, setTime } = fixture({ maxSessions: 2, idleTtlMs: 100 });
  completeTurn(store, "expired");
  setTime(70);
  const newer = completeTurn(store, "newer");
  setTime(110);
  const replacement = completeTurn(store, "replacement");
  assert.equal(store.acquireSession("expired", 2), "stale");
  assert.equal(store.acquireSession("newer", 2), "acquired");
  assert.strictEqual(store.getSession("newer"), newer);
  assert.equal(store.acquireSession("replacement", 2), "acquired");
  assert.strictEqual(store.getSession("replacement"), replacement);
});

test("stale requests do not refresh LRU order but legitimate first turns may evict", () => {
  const { store } = fixture({ maxSessions: 2 });
  completeTurn(store, "oldest");
  const recent = completeTurn(store, "recent");
  assert.equal(store.acquireSession("oldest", 1), "stale");
  assert.equal(store.acquireSession("new", 1), "acquired");
  assert.equal(store.getSession("new").turn, 0);
  store.releaseSession("new");
  assert.equal(store.acquireSession("oldest", 2), "stale");
  assert.equal(store.acquireSession("recent", 2), "acquired");
  assert.strictEqual(store.getSession("recent"), recent);
});

test("turn validation occurs before full-capacity admission without disturbing active work", () => {
  const { store } = fixture({ maxSessions: 1 });
  const active = acquire(store, "active");
  assert.equal(store.acquireSession("active", 9), "busy");
  assert.equal(store.acquireSession("unknown", 2), "stale");
  assert.equal(store.acquireSession("new", 1), "full");
  assert.strictEqual(store.getSession("active"), active);
  store.releaseSession("active");
  assert.equal(store.acquireSession("new", 1), "acquired");
});
