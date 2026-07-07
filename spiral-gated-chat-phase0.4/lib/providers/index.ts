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
