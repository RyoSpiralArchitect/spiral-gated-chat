import OpenAI from "openai";
import type {
  Response,
  ResponseCreateParamsNonStreaming,
  ResponseIncludable,
} from "openai/resources/responses/responses";
import type {
  LlmProvider,
  ProviderTextRequest,
  ProviderTextResponse,
  ProviderUsage,
} from "@/lib/providers/types";
import type { TokenLogprob } from "@/lib/types";

type ResponseCreateParamsWithLogprobs = Omit<ResponseCreateParamsNonStreaming, "include"> & {
  include?: Array<ResponseIncludable | "message.output_text.logprobs"> | null;
  top_logprobs?: number | null;
};

function normalizeTokenLogprobs(raw: any): TokenLogprob[] {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw as TokenLogprob[];
  if (Array.isArray(raw.content)) return raw.content as TokenLogprob[];
  if (Array.isArray(raw.tokens)) return raw.tokens as TokenLogprob[];
  return [];
}

function extractFirstTextAndLogprobs(response: any): { text: string; logprobs: TokenLogprob[] } {
  const msg = Array.isArray(response?.output)
    ? response.output.find((o: any) => o?.type === "message" && o?.role === "assistant")
    : null;
  const part = msg?.content?.find((c: any) => c?.type === "output_text") ?? msg?.content?.[0];
  const text: string = part?.text ?? "";
  const logprobs = normalizeTokenLogprobs(part?.logprobs);
  return { text, logprobs };
}

function normalizeUsage(usage: any): ProviderUsage | undefined {
  if (!usage) return undefined;
  return {
    input_tokens: usage.input_tokens ?? usage.prompt_tokens ?? null,
    output_tokens: usage.output_tokens ?? usage.completion_tokens ?? null,
    total_tokens: usage.total_tokens ?? null,
  };
}

async function createResponse(openai: OpenAI, params: ResponseCreateParamsWithLogprobs): Promise<Response> {
  // The API accepts output-text logprobs, but openai@4.104.0 has not added
  // "message.output_text.logprobs" to the ResponseIncludable union yet.
  return openai.responses.create(params as any) as Promise<Response>;
}

export function createOpenAIProvider(): LlmProvider {
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const defaultModel = process.env.OPENAI_MODEL || "gpt-4.1";

  return {
    name: "openai",
    model: defaultModel,
    capabilities: {
      tokenLogprobs: true,
    },
    async createText(request: ProviderTextRequest): Promise<ProviderTextResponse> {
      const response = await createResponse(client, {
        model: request.model || defaultModel,
        input: request.input,
        temperature: request.temperature,
        max_output_tokens: request.max_output_tokens,
        ...(request.includeLogprobs
          ? {
              include: ["message.output_text.logprobs"],
              top_logprobs: request.topLogprobs ?? 20,
            }
          : {}),
      });

      const { text, logprobs } = extractFirstTextAndLogprobs(response);
      return {
        text,
        logprobs,
        model: response.model || request.model || defaultModel,
        usage: normalizeUsage((response as any).usage),
        requestId: (response as any)._request_id ?? response.id ?? null,
        finishReason: (response as any).status ?? null,
      };
    },
  };
}
