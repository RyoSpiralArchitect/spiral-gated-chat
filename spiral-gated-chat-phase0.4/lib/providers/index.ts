import { createAnthropicProvider } from "@/lib/providers/anthropic";
import { createMockProvider } from "@/lib/providers/mock";
import { createOpenAIProvider } from "@/lib/providers/openai";
import type { LlmProvider, ProviderName } from "@/lib/providers/types";

function configuredProviderName(): ProviderName {
  const raw = (process.env.SPIRAL_CHAT_PROVIDER || process.env.LLM_PROVIDER || "openai").toLowerCase();
  if (raw === "anthropic" || raw === "claude") return "anthropic";
  if (raw === "mock") return "mock";
  return "openai";
}

/** Non-secret configuration; does not instantiate an SDK or require credentials. */
export function getProviderConfig() {
  const provider = configuredProviderName();
  const model = provider === "mock" ? process.env.MOCK_MODEL || "mock-gated-chat"
    : provider === "anthropic" ? process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6"
    : process.env.OPENAI_MODEL || "gpt-4.1";
  return { provider, model, isMock: provider === "mock" };
}

export function getProvider(): LlmProvider {
  const provider = configuredProviderName();
  if (provider === "anthropic") return createAnthropicProvider();
  if (provider === "mock") return createMockProvider();
  return createOpenAIProvider();
}

export type {
  LlmProvider,
  ProviderCallRecord,
  ProviderMessage,
  ProviderName,
  ProviderTextPurpose,
  ProviderTextRequest,
  ProviderTextResponse,
  ProviderUsage,
  StateSource,
} from "@/lib/providers/types";
