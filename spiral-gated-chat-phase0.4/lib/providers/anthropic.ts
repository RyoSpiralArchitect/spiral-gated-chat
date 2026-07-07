import type {
  LlmProvider,
  ProviderMessage,
  ProviderTextRequest,
  ProviderTextResponse,
  ProviderUsage,
} from "@/lib/providers/types";

type AnthropicMessage = {
  role: "user" | "assistant";
  content: string;
};

function clampTemperature(temperature: number): number {
  return Math.max(0, Math.min(1, temperature));
}

function splitMessages(input: ProviderMessage[]): { system: string | undefined; messages: AnthropicMessage[] } {
  const system = input
    .filter((message) => message.role === "system")
    .map((message) => message.content)
    .join("\n")
    .trim();

  const messages = input
    .filter((message) => message.role !== "system")
    .map((message) => ({
      role: message.role as "user" | "assistant",
      content: message.content,
    }));

  return { system: system || undefined, messages };
}

function extractText(data: any): string {
  if (!Array.isArray(data?.content)) return "";
  return data.content
    .filter((part: any) => part?.type === "text" && typeof part?.text === "string")
    .map((part: any) => part.text)
    .join("");
}

function normalizeUsage(usage: any): ProviderUsage | undefined {
  if (!usage) return undefined;
  const input = usage.input_tokens ?? null;
  const output = usage.output_tokens ?? null;
  return {
    input_tokens: input,
    output_tokens: output,
    total_tokens: typeof input === "number" && typeof output === "number" ? input + output : null,
  };
}

export function createAnthropicProvider(): LlmProvider {
  const apiKey = process.env.ANTHROPIC_API_KEY || process.env.CLAUDE_API_KEY;
  const authToken = process.env.ANTHROPIC_AUTH_TOKEN;
  const defaultModel = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";
  const baseUrl = (process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com/v1").replace(/\/+$/, "");
  const anthropicVersion = process.env.ANTHROPIC_VERSION || "2023-06-01";

  return {
    name: "anthropic",
    model: defaultModel,
    capabilities: {
      tokenLogprobs: false,
    },
    async createText(request: ProviderTextRequest): Promise<ProviderTextResponse> {
      if (!apiKey && !authToken) {
        throw new Error("ANTHROPIC_API_KEY, CLAUDE_API_KEY, or ANTHROPIC_AUTH_TOKEN is required when SPIRAL_CHAT_PROVIDER=anthropic");
      }

      const { system, messages } = splitMessages(request.input);
      const response = await fetch(`${baseUrl}/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "anthropic-version": anthropicVersion,
          ...(apiKey ? { "x-api-key": apiKey } : { authorization: `Bearer ${authToken}` }),
        },
        body: JSON.stringify({
          model: request.model || defaultModel,
          max_tokens: request.max_output_tokens,
          temperature: clampTemperature(request.temperature),
          ...(system ? { system } : {}),
          messages,
        }),
      });

      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Anthropic API ${response.status}: ${detail}`);
      }

      const data = await response.json();
      return {
        text: extractText(data),
        logprobs: [],
        model: data.model || request.model || defaultModel,
        usage: normalizeUsage(data.usage),
        requestId: response.headers.get("request-id") || response.headers.get("anthropic-request-id"),
        finishReason: data.stop_reason ?? null,
      };
    },
  };
}
