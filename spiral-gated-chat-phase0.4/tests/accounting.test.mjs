import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import ts from "typescript";

// Accounting is deliberately pure. Transpile with the project's existing TypeScript
// dependency so these small tests need no bundler, runtime loader, or extra package.
const source = await readFile(new URL("../lib/accounting.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
});
const { summarizeCalls } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);

const call = (purpose, usage, rest = {}) => ({
  purpose,
  provider: "openai",
  model: "test-model",
  status: "ok",
  latency_ms: 7,
  ...(usage === undefined ? {} : { usage }),
  ...rest,
});

test("counts every provider phase, not just the Main output", () => {
  const calls = ["probe", "explore", "verify", "main", "summary"].map((purpose, index) =>
    call(purpose, { input_tokens: (index + 1) * 10, output_tokens: index + 1 })
  );
  const result = summarizeCalls(calls, 44);
  assert.equal(result.call_count, 5);
  assert.equal(result.failed_calls, 0);
  assert.equal(result.input_tokens, 150);
  assert.equal(result.output_tokens, 15);
  assert.equal(result.total_tokens, 165);
  assert.equal(result.known_total_tokens, 165);
  assert.equal(result.unknown_usage_calls, 0);
  assert.equal(result.provider_latency_ms, 35);
  assert.equal(result.turn_latency_ms, 44);
  assert.equal(result.usage_kind, "reported");
});

test("failed calls remain in counts and make an otherwise known total partial", () => {
  const result = summarizeCalls([
    call("probe", { input_tokens: 10, output_tokens: 3 }),
    call("main", undefined, { status: "error", latency_ms: 13, error: "test failure" }),
  ], 26);
  assert.equal(result.call_count, 2);
  assert.equal(result.failed_calls, 1);
  assert.equal(result.input_tokens, null);
  assert.equal(result.output_tokens, null);
  assert.equal(result.total_tokens, null);
  assert.equal(result.known_total_tokens, 13);
  assert.equal(result.unknown_usage_calls, 1);
  assert.equal(result.provider_latency_ms, 20);
  assert.equal(result.usage_kind, "partial");
});

test("absent usage is unknown, never a fabricated zero-token call", () => {
  const result = summarizeCalls([call("main", undefined)], 8);
  assert.equal(result.total_tokens, null);
  assert.equal(result.input_tokens, null);
  assert.equal(result.output_tokens, null);
  assert.equal(result.known_total_tokens, 0);
  assert.equal(result.unknown_usage_calls, 1);
  assert.equal(result.usage_kind, "unknown");
});

test("explicit zero usage stays known", () => {
  const result = summarizeCalls([call("probe", { input_tokens: 0, output_tokens: 0, total_tokens: 0 })], 1);
  assert.equal(result.total_tokens, 0);
  assert.equal(result.unknown_usage_calls, 0);
  assert.equal(result.usage_kind, "reported");
});

test("a reported total can be known while the input/output split is unavailable", () => {
  const result = summarizeCalls([call("main", { total_tokens: 29 })], 8);
  assert.equal(result.total_tokens, 29);
  assert.equal(result.known_total_tokens, 29);
  assert.equal(result.input_tokens, null);
  assert.equal(result.output_tokens, null);
  assert.equal(result.unknown_usage_calls, 0);
});

test("one-sided usage preserves its known component without inventing a total", () => {
  const result = summarizeCalls([call("main", { input_tokens: 10 })], 8);
  assert.equal(result.input_tokens, 10);
  assert.equal(result.output_tokens, null);
  assert.equal(result.total_tokens, null);
  assert.equal(result.unknown_usage_calls, 1);
});

test("mock accounting is visibly labeled as an estimate", () => {
  const result = summarizeCalls([call("main", { input_tokens: 10, output_tokens: 4 }, { provider: "mock" })], 9);
  assert.equal(result.total_tokens, 14);
  assert.equal(result.usage_kind, "mock_estimate");
});

test("a failed mock call with unknown usage is partial rather than fully estimated", () => {
  const result = summarizeCalls([
    call("probe", { input_tokens: 10, output_tokens: 4 }, { provider: "mock" }),
    call("main", undefined, { provider: "mock", status: "error" }),
  ], 15);
  assert.equal(result.usage_kind, "partial");
  assert.equal(result.total_tokens, null);
  assert.equal(result.known_total_tokens, 14);
});

test("invalid token values are treated as unavailable", () => {
  for (const value of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
    const result = summarizeCalls([call("main", { input_tokens: value, output_tokens: 3 })], 8);
    assert.equal(result.input_tokens, null);
    assert.equal(result.total_tokens, null);
    assert.equal(result.unknown_usage_calls, 1);
  }
});
