import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import ts from "typescript";

// Exercise the production writer against real temporary files, with no new loader.
const source = await readFile(new URL("../lib/logStore.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { appendTurnLog } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

async function fixture(t) {
  const dir = await mkdtemp(path.join(tmpdir(), "spiral-log-store-"));
  const previous = process.env.SPIRAL_CHAT_LOG_DIR;
  process.env.SPIRAL_CHAT_LOG_DIR = dir;
  t.after(async () => {
    if (previous === undefined) delete process.env.SPIRAL_CHAT_LOG_DIR;
    else process.env.SPIRAL_CHAT_LOG_DIR = previous;
    await rm(dir, { recursive: true, force: true });
  });
  return dir;
}

function payload(sessionId, turn = 1) {
  return {
    sessionId,
    turn,
    mode: "auto",
    accounting: {
      call_count: 0, failed_calls: 0, input_tokens: 0, output_tokens: 0,
      total_tokens: 0, known_total_tokens: 0, unknown_usage_calls: 0,
      provider_latency_ms: 0, turn_latency_ms: 0, usage_kind: "reported",
    },
    status: "ok",
    provider: "mock",
    model: "test-model",
    stateSource: "heuristic_probe_fields",
    latencyMs: 0,
    calls: [],
    userText: `${sessionId}: input ${turn}`,
    assistantText: `${sessionId}: output ${turn}`,
    debug: { turn },
  };
}

async function lines(filePath) {
  return (await readFile(filePath, "utf8")).trim().split("\n").map((line) => JSON.parse(line));
}

async function assertIndependentLogs(dir, ids) {
  const results = [];
  for (const id of ids) results.push(await appendTurnLog(payload(id)));
  assert.equal(new Set(results.map(({ filePath }) => filePath)).size, ids.length);
  assert.equal((await readdir(path.join(dir, "sessions"))).length, ids.length);
  const index = await lines(path.join(dir, "session-index.jsonl"));
  assert.equal(index.length, ids.length);
  for (const [i, result] of results.entries()) {
    const id = ids[i];
    assert.equal(path.dirname(result.filePath), path.join(dir, "sessions"));
    assert.equal(path.resolve(result.relativePath), result.filePath);
    assert.equal(result.sessionIndexPath, path.join(dir, "session-index.jsonl"));
    assert.equal(path.resolve(result.sessionIndexRelativePath), result.sessionIndexPath);
    const filename = path.basename(result.filePath);
    assert.match(filename, /^[a-zA-Z0-9_.-]+-[a-f0-9]{64}\.jsonl$/);
    assert.ok(filename.length <= 151, "readable prefix and filename must be bounded");
    assert.ok(filename.endsWith(`-${createHash("sha256").update(id).digest("hex")}.jsonl`));
    const rows = await lines(result.filePath);
    assert.equal(rows.length, 1);
    assert.equal(rows[0].sessionId, id, "turn log must retain the original ID");
    assert.equal(rows[0].userText, payload(id).userText);
    assert.equal(index[i].sessionId, id, "index must retain the original ID");
    assert.equal(index[i].log_path, result.relativePath);
    assert.equal(`${index[i].safe_session_id}.jsonl`, filename);
  }
  return results;
}

test("slash and underscore session IDs produce separate logs and index entries", async (t) => {
  const dir = await fixture(t);
  await assertIndependentLogs(dir, ["tenant/a", "tenant_a"]);
});

test("IDs sharing a truncated 80-character prefix produce separate logs", async (t) => {
  const dir = await fixture(t);
  const prefix = "a".repeat(80);
  await assertIndependentLogs(dir, [`${prefix}-first`, `${prefix}-second`]);
});

test("safe IDs are also hashed so they cannot impersonate another ID's log filename", async (t) => {
  const dir = await fixture(t);
  const firstId = "tenant/a";
  const filenameLikeId = `tenant_a-${createHash("sha256").update(firstId).digest("hex")}`;
  await assertIndependentLogs(dir, [firstId, filenameLikeId]);
});

test("repeated turns for the same original ID append to one log and index it once", async (t) => {
  const dir = await fixture(t);
  const id = "tenant/a";
  const results = [];
  for (let turn = 1; turn <= 3; turn += 1) results.push(await appendTurnLog(payload(id, turn)));
  assert.deepEqual(results, [results[0], results[0], results[0]]);
  assert.deepEqual(await readdir(path.join(dir, "sessions")), [path.basename(results[0].filePath)]);
  const rows = await lines(results[0].filePath);
  assert.deepEqual(rows.map(({ turn }) => turn), [1, 2, 3]);
  assert.ok(rows.every(({ sessionId }) => sessionId === id));
  const index = await lines(results[0].sessionIndexPath);
  assert.equal(index.length, 1);
  assert.equal(index[0].sessionId, id);
  assert.equal(index[0].first_turn, 1);
});

test("path-like, empty, and Unicode IDs stay inside the sessions directory", async (t) => {
  const dir = await fixture(t);
  await assertIndependentLogs(dir, ["../../escape", "/absolute/path", "windows\\path", "", "..", "会話/一", "会話_一"]);
  assert.deepEqual((await readdir(dir)).sort(), ["session-index.jsonl", "sessions"]);
});
