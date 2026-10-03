import type { ProviderCallRecord } from "@/lib/providers/types";

export type TurnAccounting = {
  call_count: number;
  failed_calls: number;
  input_tokens: number | null;
  output_tokens: number | null;
  total_tokens: number | null;
  known_total_tokens: number;
  unknown_usage_calls: number;
  provider_latency_ms: number;
  turn_latency_ms: number;
  usage_kind: "mock_estimate" | "reported" | "partial" | "unknown";
};

function valid(value: number | null | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

export function getCallTokens(call: ProviderCallRecord): number | null {
  const explicit = valid(call.usage?.total_tokens);
  if (explicit !== null) return explicit;
  const input = valid(call.usage?.input_tokens);
  const output = valid(call.usage?.output_tokens);
  return input !== null && output !== null ? input + output : null;
}

/** Never treat an unreported or failed call as zero tokens. Includes every purpose. */
export function summarizeCalls(calls: ProviderCallRecord[], turnLatencyMs: number): TurnAccounting {
  const totals = calls.map(getCallTokens);
  const unknown = totals.filter((n) => n === null).length;
  const sumField = (key: "input_tokens" | "output_tokens") => {
    const values = calls.map((call) => valid(call.usage?.[key]));
    return values.some((n) => n === null) ? null : values.reduce<number>((sum, n) => sum + (n ?? 0), 0);
  };
  const knownTotal = totals.reduce<number>((sum, n) => sum + (n ?? 0), 0);
  return {
    call_count: calls.length,
    failed_calls: calls.filter((call) => call.status === "error").length,
    input_tokens: sumField("input_tokens"),
    output_tokens: sumField("output_tokens"),
    total_tokens: unknown ? null : knownTotal,
    known_total_tokens: knownTotal,
    unknown_usage_calls: unknown,
    provider_latency_ms: calls.reduce((sum, call) => sum + call.latency_ms, 0),
    turn_latency_ms: turnLatencyMs,
    usage_kind: unknown
      ? unknown === calls.length ? "unknown" : "partial"
      : calls.length && calls.every((call) => call.provider === "mock") ? "mock_estimate" : "reported",
  };
}
