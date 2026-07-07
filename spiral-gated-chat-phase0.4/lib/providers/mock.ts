import type { LlmProvider, ProviderTextRequest, ProviderTextResponse } from "@/lib/providers/types";

function mockProbe(text: string): string {
  const dim = /死|risk|危険|壊|不安|ログ|log/i.test(text) ? "RISK" : "UNCERTAINTY";
  return [
    `DIM: ${dim}`,
    dim === "RISK" ? "FOCUS: 脆弱な設計部分" : "FOCUS: 次に確認する対象",
    dim === "RISK" ? "NEXT: ログと単一障害点を確認する" : "NEXT: 条件を一つ確認する",
    "WHY: 次の判断に必要だから。",
  ].join("\n");
}

function latestUserText(request: ProviderTextRequest): string {
  return [...request.input].reverse().find((message) => message.role === "user")?.content ?? "";
}

export function createMockProvider(): LlmProvider {
  const model = process.env.MOCK_MODEL || "mock-gated-chat";

  return {
    name: "mock",
    model,
    capabilities: {
      tokenLogprobs: false,
    },
    async createText(request: ProviderTextRequest): Promise<ProviderTextResponse> {
      const userText = latestUserText(request);
      let text = "";

      if (request.purpose === "probe") {
        text = mockProbe(userText);
      } else if (request.purpose === "explore") {
        text = [
          mockProbe(userText),
          "",
          "DIM: GOAL\nFOCUS: 次の具体手順\nNEXT: 一番小さい確認を選ぶ\nWHY: 作業を前に進めるため。",
          "",
          "DIM: OPPORTUNITY\nFOCUS: 観察できる価値\nNEXT: 保存されたログを読む\nWHY: 変化を比較できるから。",
        ].join("\n");
      } else if (request.purpose === "verify") {
        text = "PICK: 1";
      } else if (request.purpose === "summary") {
        text = "ユーザーは状態ゲート付きチャットのログ保存と動作確認を進めている。";
      } else {
        text = `mock response: ${userText.slice(0, 120)}`;
      }

      return {
        text,
        logprobs: [],
        model,
        usage: {
          input_tokens: request.input.reduce((sum, message) => sum + Math.ceil(message.content.length / 4), 0),
          output_tokens: Math.ceil(text.length / 4),
        },
        requestId: `mock-${request.purpose}`,
        finishReason: "stop",
      };
    },
  };
}
