import type { TokenLogprob } from "@/lib/types";

export type ProviderName = "openai" | "anthropic" | "mock";

export type ProviderTextPurpose = "probe" | "explore" | "verify" | "main" | "summary";

export type ProviderMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type ProviderUsage = {
  input_tokens?: number | null;
  output_tokens?: number | null;
  total_tokens?: number | null;
};

export type ProviderTextRequest = {
  purpose: ProviderTextPurpose;
  model?: string;
  input: ProviderMessage[];
  temperature: number;
  max_output_tokens: number;
  includeLogprobs?: boolean;
  topLogprobs?: number;
};

export type ProviderTextResponse = {
  text: string;
  logprobs: TokenLogprob[];
  model: string;
  usage?: ProviderUsage;
  requestId?: string | null;
  finishReason?: string | null;
};

export type LlmProvider = {
  name: ProviderName;
  model: string;
  capabilities: {
    tokenLogprobs: boolean;
  };
  createText(request: ProviderTextRequest): Promise<ProviderTextResponse>;
};

export type ProviderCallRecord = {
  purpose: ProviderTextPurpose;
  provider: ProviderName;
  model: string;
  latency_ms: number;
  usage?: ProviderUsage;
  request_id?: string | null;
  finish_reason?: string | null;
};

export type StateSource = "token_logprobs" | "heuristic_probe_fields" | "previous_state";
