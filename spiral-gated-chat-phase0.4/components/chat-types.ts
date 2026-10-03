export type GateMode = "fixed" | "auto";

export type Accounting = {
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

export type ProviderCall = {
  provider?: string;
  purpose: string;
  model: string;
  status?: "ok" | "error";
  latency_ms: number;
  usage?: { input_tokens?: number | null; output_tokens?: number | null; total_tokens?: number | null };
  error?: string;
};

export type MemoryFragment = { id: string; turn: number; dim: string | null; salience: number; text: string };

export type DebugPayload = {
  mode?: GateMode;
  probeText: string | null;
  probeText_original?: string | null;
  dim: string | null;
  focus: string | null;
  next?: string | null;
  state: number;
  observed_state?: number;
  viewpoint?: { previous_dim: string | null; previous_focus: string | null; changed: boolean; pulse_changed: boolean };
  accounting?: Accounting;
  pulse?: {
    triggered: boolean;
    stagnation_detected: boolean;
    repeating_dim: string | null;
    candidates_text: string | null;
    picked: number | null;
    selected_probe: string | null;
  };
  memory?: {
    ctx_keep_msgs: number;
    summary_chars: number;
    attn_items: number;
    frag_items?: number;
    summary_update_interval: number;
    summary_update_max_tokens: number;
    context_used?: { role: string; content: string }[];
    attention_used?: { turn: number; dim: string; focus: string; next?: string | null }[];
    fragments?: { total: number; injected: MemoryFragment[]; top: MemoryFragment[]; decay_factor: number };
  };
  summary_used?: string | null;
  summary_stored?: string | null;
  metrics: { surprisal: number | null; entropy: number | null; score: number | null; state_source?: string };
  provider?: { name: string; model: string; supports_token_logprobs: boolean; state_source: string; calls?: ProviderCall[] };
  params: { max_output_tokens: number; temperature: number; context_keep_msgs: number };
  log?: { saved: boolean; path: string | null; session_index_path?: string | null; error: string | null };
  notes: string[];
};

export type TurnResult = {
  sessionId: string;
  turn: number;
  mode: GateMode;
  userText: string;
  assistantText: string;
  debug: DebugPayload;
};

export type FailedTurn = {
  mode: GateMode;
  turn: number;
  userText: string;
  message: string;
  accounting?: Accounting;
  calls?: ProviderCall[];
};

export type PublicConfig = { provider: string; model: string; isMock: boolean };

export function decimal(value: number | null | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export function tokenLabel(accounting?: Accounting, isMock = false): string {
  if (!accounting) return "未取得";
  const estimate = (isMock || accounting.usage_kind === "mock_estimate") ? "約 " : "";
  if (accounting.total_tokens != null) return `${estimate}${accounting.total_tokens.toLocaleString("ja-JP")}`;
  if (accounting.unknown_usage_calls > 0 && accounting.known_total_tokens > 0) return `${estimate}${accounting.known_total_tokens.toLocaleString("ja-JP")} + 不明`;
  return "不明";
}

export function summarizeAccounting(items: (Accounting | undefined)[]): Accounting | undefined {
  const present = items.filter((item): item is Accounting => Boolean(item));
  if (!present.length) return undefined;
  const sum = (key: "call_count" | "failed_calls" | "known_total_tokens" | "unknown_usage_calls" | "provider_latency_ms" | "turn_latency_ms") => present.reduce((total, item) => total + item[key], 0);
  const nullableSum = (key: "input_tokens" | "output_tokens" | "total_tokens") => present.length < items.length || present.some((item) => item[key] == null) ? null : present.reduce((total, item) => total + (item[key] ?? 0), 0);
  return {
    call_count: sum("call_count"), failed_calls: sum("failed_calls"),
    input_tokens: nullableSum("input_tokens"), output_tokens: nullableSum("output_tokens"), total_tokens: nullableSum("total_tokens"),
    known_total_tokens: sum("known_total_tokens"), unknown_usage_calls: sum("unknown_usage_calls"),
    provider_latency_ms: sum("provider_latency_ms"), turn_latency_ms: sum("turn_latency_ms"),
    usage_kind: present.every((item) => item.usage_kind === "mock_estimate") ? "mock_estimate" : present.every((item) => item.usage_kind === "reported") ? "reported" : "partial",
  };
}
