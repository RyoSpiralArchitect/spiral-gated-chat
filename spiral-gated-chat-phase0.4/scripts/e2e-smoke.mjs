import assert from "node:assert/strict";
import { once } from "node:events";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import ts from "typescript";

// This regression suite ALWAYS uses mock. It never reads a key or calls a live LLM.
const root = fileURLToPath(new URL("..", import.meta.url));
const port = Number(process.env.E2E_PORT || 3100);
const baseUrl = `http://127.0.0.1:${port}`;
const logDir = await mkdtemp(path.join(tmpdir(), "spiral-gated-chat-e2e-"));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let child;
let output = "";
let currentLogDir = logDir;
let requests = 0;
const results = [];
const scenarioSource = await readFile(new URL("../lib/scenarios.ts", import.meta.url), "utf8");
const scenarioModule = ts.transpileModule(scenarioSource, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } });
const { scenarios } = await import(`data:text/javascript;base64,${Buffer.from(scenarioModule.outputText).toString("base64")}`);

async function startServer(failPurpose) {
  currentLogDir = path.join(logDir, failPurpose);
  output = "";
  child = spawn(process.execPath, [path.join(root, "node_modules/next/dist/bin/next"), "dev", "-p", String(port), "-H", "127.0.0.1"], {
    cwd: root,
    env: {
      ...process.env,
      PORT: String(port),
      NEXT_TELEMETRY_DISABLED: "1",
      SPIRAL_CHAT_PROVIDER: "mock",
      SPIRAL_CHAT_LOG_DIR: currentLogDir,
      SPIRAL_MOCK_FAIL_PURPOSE: failPurpose,
      SPIRAL_MOCK_DELAY_MS: "15",
    },
    detached: process.platform !== "win32",
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout.on("data", (chunk) => { output += chunk.toString(); });
  child.stderr.on("data", (chunk) => { output += chunk.toString(); });
  const deadline = Date.now() + 60_000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error(`Next exited before readiness (${child.exitCode})\n${output}`);
    try {
      const res = await fetch(`${baseUrl}/api/config`);
      if (res.ok) {
        const config = await res.json();
        assert.equal(config.provider, "mock");
        assert.equal(config.isMock, true);
        assert.equal(typeof config.model, "string");
        return;
      }
    } catch { /* Wait for Next's initial route compilation. */ }
    await sleep(200);
  }
  throw new Error(`Timed out waiting for ${baseUrl}\n${output}`);
}

async function stopServer() {
  if (!child || child.exitCode !== null) return;
  const current = child;
  const exited = once(current, "exit");
  const kill = (signal) => {
    try {
      if (process.platform !== "win32") process.kill(-current.pid, signal);
      else current.kill(signal);
    } catch (error) {
      if (error.code !== "ESRCH") throw error;
    }
  };
  kill("SIGTERM");
  const closed = await Promise.race([exited.then(() => true), sleep(3000).then(() => false)]);
  if (!closed) { kill("SIGKILL"); await exited; }
  child = null;
}

async function post(body, expectedStatus = 200, raw = false) {
  requests += 1;
  const res = await fetch(`${baseUrl}/api/step`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: raw ? body : JSON.stringify(body),
  });
  const value = await res.json();
  if (expectedStatus !== null) assert.equal(res.status, expectedStatus, JSON.stringify(value));
  return { status: res.status, body: value };
}

async function step(sessionId, userText, mode = "auto", extra = {}) {
  const { body } = await post({ sessionId, userText, mode, ...extra });
  assert.equal(body.sessionId, sessionId);
  assert.equal(body.mode, mode);
  assert.equal(body.debug.mode, mode);
  assert.equal(body.userText, userText.trim());
  assert.ok(body.assistantText);
  assert.equal(body.debug.provider.name, "mock");
  assert.equal(body.debug.provider.state_source, "heuristic_probe_fields");
  assert.equal(body.debug.log.saved, true);
  assertAccounting(body.debug.accounting, body.debug.provider.calls);
  const memory = body.debug.memory;
  assert.ok(Array.isArray(memory.context_used));
  assert.ok(Array.isArray(memory.attention_used));
  assert.ok(Array.isArray(memory.fragments.injected));
  assert.ok(memory.context_used.length <= memory.ctx_keep_msgs);
  assert.ok(memory.attention_used.length <= memory.attn_items);
  assert.ok(memory.fragments.injected.length <= memory.frag_items);
  assert.deepEqual(memory.context_used.at(-1), { role: "user", content: userText.trim() });
  return body;
}

function assertAccounting(accounting, calls) {
  assert.equal(accounting.call_count, calls.length);
  assert.equal(accounting.failed_calls, calls.filter((call) => call.status === "error").length);
  assert.ok(calls.every((call) => ["ok", "error"].includes(call.status)));
  assert.ok(calls.every((call) => call.provider === "mock"));
  const known = calls.filter((call) => typeof call.usage?.input_tokens === "number" && typeof call.usage?.output_tokens === "number");
  const tokens = known.reduce((sum, call) => sum + call.usage.input_tokens + call.usage.output_tokens, 0);
  assert.equal(accounting.known_total_tokens, tokens);
  assert.equal(accounting.unknown_usage_calls, calls.length - known.length);
  assert.equal(accounting.total_tokens, known.length === calls.length ? tokens : null);
  assert.equal(accounting.usage_kind, known.length === calls.length ? "mock_estimate" : known.length ? "partial" : "unknown");
  assert.equal(accounting.provider_latency_ms, calls.reduce((sum, call) => sum + call.latency_ms, 0));
  assert.ok(accounting.turn_latency_ms >= accounting.provider_latency_ms);
}

async function entries(sessionId) {
  const data = await readFile(path.join(currentLogDir, "sessions", `${sessionId}.jsonl`), "utf8");
  return data.trim().split("\n").map((line) => JSON.parse(line));
}

function fixedSettings(turn) {
  const { ctx_keep_msgs, summary_chars, attn_items, frag_items, summary_update_interval, summary_update_max_tokens } = turn.debug.memory;
  return { ...turn.debug.params, ctx_keep_msgs, summary_chars, attn_items, frag_items, summary_update_interval, summary_update_max_tokens };
}

// Remove identity/latency data only: the remaining deterministic values must replay.
function comparable(turn) {
  return {
    state: turn.debug.state,
    observed_state: turn.debug.observed_state,
    probe: turn.debug.probeText,
    viewpoint: turn.debug.viewpoint,
    params: turn.debug.params,
    settings: fixedSettings(turn),
    pulse: turn.debug.pulse,
    summary_used: turn.debug.summary_used,
    summary_stored: turn.debug.summary_stored,
    context_used: turn.debug.memory.context_used,
    attention_used: turn.debug.memory.attention_used,
    fragments: turn.debug.memory.fragments.injected.map(({ id, ...fragment }) => fragment),
    calls: turn.debug.provider.calls.map(({ purpose, status, usage }) => ({ purpose, status, usage })),
  };
}

async function mainSuite() {
  await startServer("main");
  const invalid = [
    null, [], {}, { sessionId: "bad-only" },
    { sessionId: 3, userText: "hello" },
    { sessionId: "../escape", userText: "hello" },
    { sessionId: "x".repeat(81), userText: "hello" },
    { sessionId: "bad", userText: null },
    { sessionId: "bad", userText: 3 },
    { sessionId: "bad", userText: "   " },
    { sessionId: "bad", userText: "x".repeat(12001) },
    { sessionId: "bad", userText: "hello", mode: "unknown" },
    { sessionId: "bad", userText: "hello", comparisonId: "../escape" },
    { sessionId: "bad", userText: "hello", scenarioId: 3 },
    { sessionId: "bad", userText: "hello", expectedTurn: 0 },
    { sessionId: "bad", userText: "hello", expectedTurn: 1.5 },
  ];
  for (const body of invalid) await post(body, 400);
  await post("{", 400, true);
  assert.deepEqual(await readdir(currentLogDir).catch(() => []), [], "invalid input must not write logs or call a provider");
  const defaultMode = (await post({ sessionId: "default-mode", userText: "hello" })).body;
  assert.equal(defaultMode.mode, "auto");
  assert.equal(defaultMode.turn, 1);
  results.push("config + 17 invalid bodies + default mode");

  const neutral = "次の作業を一つ決めよう。";
  const auto = [];
  const fixed = [];
  const comparison = { comparisonId: "paired-replay", scenarioId: "neutral-ten-turns" };
  for (let index = 0; index < 16; index += 1) {
    auto.push(await step("pair-auto", neutral, "auto", comparison));
    fixed.push(await step("pair-fixed", neutral, "fixed", comparison));
    assert.equal(auto[index].turn, index + 1);
    assert.equal(fixed[index].turn, index + 1);
  }
  const pulse = auto.slice(0, 10).find((turn) => turn.debug.pulse.triggered);
  assert.ok(pulse, "repeated neutral text must trigger a pulse within ten auto turns");
  assert.equal(pulse.debug.pulse.picked, 2);
  assert.equal(pulse.debug.viewpoint.pulse_changed, true);
  assert.notEqual(pulse.debug.probeText, pulse.debug.probeText_original);
  assert.ok(pulse.debug.provider.calls.some((call) => call.purpose === "explore"));
  assert.ok(pulse.debug.provider.calls.some((call) => call.purpose === "verify"));
  const phases = new Set(auto.flatMap((turn) => turn.debug.provider.calls.map((call) => call.purpose)));
  for (const purpose of ["probe", "main", "explore", "verify", "summary"]) assert.ok(phases.has(purpose), `missing ${purpose} coverage`);
  assert.ok(fixed.every((turn) => turn.debug.state === 0.5));
  assert.ok(fixed.every((turn) => !turn.debug.pulse.triggered));
  assert.ok(fixed.every((turn) => turn.debug.provider.calls.some((call) => call.purpose === "probe")));
  assert.ok(fixed.every((turn) => turn.debug.provider.calls.every((call) => !["explore", "verify"].includes(call.purpose))));
  for (const turn of fixed) assert.deepEqual(fixedSettings(turn), fixedSettings(fixed[0]));
  assert.ok(fixed.some((turn) => turn.debug.provider.calls.some((call) => call.purpose === "summary")));
  assert.ok(fixed.at(-1).debug.summary_stored.includes(neutral), "mock summary should reflect the actual conversation");
  results.push(`paired replay: Auto pulse on turn ${pulse.turn}; all five call phases accounted; Fixed stable`);

  const originalSnapshot = structuredClone(auto[0].debug.memory);
  await step("pair-auto", "予算は1000円。期限は明日。安全性の制約を確認して。", "auto", comparison);
  assert.deepEqual(auto[0].debug.memory, originalSnapshot);
  const persistedAuto = await entries("pair-auto");
  assert.deepEqual(persistedAuto[0].debug.memory, originalSnapshot, "old saved snapshots must not change when fragments are updated");
  assert.equal(persistedAuto[0].debug.memory.context_used.length, 1);
  assert.ok(persistedAuto[0].debug.memory.fragments.injected.every((fragment) => fragment.turn === 1));
  for (const [index, entry] of persistedAuto.entries()) {
    assert.equal(entry.schema_version, "spiral-gated-chat.turn.v2");
    assert.equal(entry.turn, index + 1);
    assert.equal(entry.mode, "auto");
    assert.equal(entry.status, "ok");
    assert.equal(entry.comparison_id, comparison.comparisonId);
    assert.equal(entry.scenario_id, comparison.scenarioId);
    assert.deepEqual(entry.accounting, entry.debug.accounting);
    assertAccounting(entry.accounting, entry.calls);
  }
  // An independent replay must be unaffected by both earlier sessions.
  for (let index = 0; index < 10; index += 1) {
    const repeated = await step("replay-auto", neutral);
    assert.deepEqual(comparable(repeated), comparable(auto[index]));
  }
  const isolation = await step("isolation", "独立した新しい会話です。");
  assert.equal(isolation.turn, 1);
  assert.equal(isolation.debug.memory.context_used.length, 1);
  assert.equal(isolation.debug.summary_used, null);
  assert.equal(isolation.debug.viewpoint.previous_dim, null);
  results.push("session isolation + deterministic rerun + immutable historical snapshots + JSONL v2");

  await post({ sessionId: "pair-fixed", userText: "hello", mode: "auto" }, 409);
  const stillFixed = await step("pair-fixed", "hello", "fixed", comparison);
  assert.equal(stillFixed.turn, 17, "mode mismatch must not mutate or lock the session");
  const guarded = await step("guarded", "hello", "auto", { expectedTurn: 1 });
  assert.equal(guarded.turn, 1);
  await post({ sessionId: "guarded", userText: "duplicate retry", expectedTurn: 1 }, 409);
  await post({ sessionId: "guarded", userText: "skipped turn", expectedTurn: 3 }, 409);
  assert.equal((await step("guarded", "next", "auto", { expectedTurn: 2 })).turn, 2);
  assert.equal((await entries("guarded")).length, 2);
  const concurrent = await Promise.all([
    post({ sessionId: "locked", userText: "first concurrent turn" }, null),
    post({ sessionId: "locked", userText: "second concurrent turn" }, null),
  ]);
  assert.deepEqual(concurrent.map((result) => result.status).sort(), [200, 409]);
  assert.equal((await step("locked", "after lock release")).turn, 2);
  results.push("immutable mode + expectedTurn duplicate protection + concurrent same-session lock + lock release");

  const beforeFailure = await step("rollback", "hello");
  const failure = (await post({ sessionId: "rollback", userText: "[mock:fail] failed turn must not survive", mode: "auto" }, 500)).body;
  assertAccounting(failure.accounting, failure.calls);
  assert.equal(failure.accounting.failed_calls, 1);
  assert.equal(failure.accounting.total_tokens, null);
  assert.deepEqual(failure.calls.map((call) => [call.purpose, call.status]), [["probe", "ok"], ["main", "error"]]);
  const retry = await step("rollback", "hello again");
  assert.equal(retry.turn, 2);
  assert.equal(retry.debug.memory.context_used.length, 3);
  assert.ok(!JSON.stringify(retry.debug.memory).includes("[mock:fail]"));
  assert.deepEqual(retry.debug.memory.context_used.slice(0, 2), [
    { role: "user", content: beforeFailure.userText },
    { role: "assistant", content: beforeFailure.assistantText },
  ]);
  const cleanFirst = await step("rollback-control", "hello");
  const cleanSecond = await step("rollback-control", "hello again");
  assert.deepEqual(comparable(beforeFailure), comparable(cleanFirst));
  assert.deepEqual(comparable(retry), comparable(cleanSecond), "failed turn must not alter gate, memory, or history");
  const failureLog = await entries("rollback");
  assert.deepEqual(failureLog.map(({ turn, status }) => ({ turn, status })), [{ turn: 1, status: "ok" }, { turn: 2, status: "error" }, { turn: 2, status: "ok" }]);
  assert.equal(failureLog[1].assistantText, "");
  assert.equal(failureLog[1].accounting.failed_calls, 1);
  const index = (await readFile(path.join(currentLogDir, "session-index.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(index.filter((entry) => entry.sessionId === "rollback").length, 1);
  assert.equal(index.filter((entry) => entry.sessionId === "pair-auto").length, 1);
  assert.ok(index.every((entry) => entry.log_path.endsWith(`${entry.sessionId}.jsonl`)));
  results.push("Main failure recorded + unknown usage + atomic rollback + retry uses same turn number");
  const repeatPreset = scenarios.find((scenario) => scenario.id === "repeated-perspective");
  assert.equal(repeatPreset.turns.length, 10);
  const presetTurns = [];
  for (const text of repeatPreset.turns) presetTurns.push(await step("actual-repeat-preset", text));
  assert.ok(presetTurns.some((turn) => turn.debug.pulse.triggered && turn.debug.viewpoint.pulse_changed), "actual repeated-perspective UI preset must demonstrate a changed pulse in mock mode");
  results.push("actual 10-turn UI repeated-perspective preset demonstrates a changed pulse");
  await stopServer();
}

async function optionalFailureSuite(purpose) {
  await startServer(purpose);
  const id = `optional-${purpose}`;
  const mode = purpose === "summary" ? "fixed" : "auto";
  const turns = [];
  for (let turn = 1; turn <= 12; turn += 1) {
    turns.push(await step(id, "次の作業を一つ決めよう。 [mock:fail]", mode));
  }
  const failed = turns.find((turn) => turn.debug.provider.calls.some((call) => call.purpose === purpose && call.status === "error"));
  assert.ok(failed, `${purpose} failure was not exercised`);
  assert.ok(failed.assistantText, `optional ${purpose} failure must preserve the Main response`);
  assert.equal(failed.debug.accounting.total_tokens, null);
  assert.equal(failed.debug.accounting.usage_kind, "partial");
  assert.equal(failed.debug.accounting.failed_calls, 1);
  assert.ok(failed.debug.notes.some((note) => note.includes("failed")));
  if (purpose === "explore") {
    assert.equal(failed.debug.pulse.triggered, false);
    assert.equal(failed.debug.viewpoint.pulse_changed, false);
    assert.ok(!failed.debug.provider.calls.some((call) => call.purpose === "verify"));
  }
  const persisted = await entries(id);
  assert.equal(persisted.length, 12);
  assert.ok(persisted.every((entry) => entry.status === "ok"));
  assert.equal(persisted[failed.turn - 1].accounting.failed_calls, 1);
  results.push(`${purpose} failure retained in accounting while successful Main turn commits`);
  await stopServer();
}

try {
  await mainSuite();
  await optionalFailureSuite("summary");
  await optionalFailureSuite("explore");
  console.log(JSON.stringify({ ok: true, provider: "mock", requests, checks: results, ...(process.env.KEEP_E2E_LOGS === "1" ? { logDir } : {}) }, null, 2));
} catch (error) {
  console.error(output);
  throw error;
} finally {
  await stopServer();
  if (process.env.KEEP_E2E_LOGS !== "1") await rm(logDir, { recursive: true, force: true });
}
