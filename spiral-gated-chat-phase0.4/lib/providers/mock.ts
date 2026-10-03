import type { LlmProvider, ProviderTextRequest, ProviderTextResponse } from "@/lib/providers/types";

function mockProbe(text: string): string {
  const constraint = /必ず|絶対|条件|予算|以内|禁止|deadline|constraint|budget/i.test(text);
  const risk = /死|risk|危険|壊|不安|ログ|log/i.test(text);
  const greeting = /^(こんにちは|おはよう|ありがとう|やあ|hello|hi)[！!。\s]*$/i.test(text.trim());
  const dim = constraint || risk ? "RISK" : greeting ? "META" : "UNCERTAINTY";
  const focus = constraint ? `守る条件: ${text.slice(0, 48)}` : risk ? "脆弱な設計部分" : greeting ? "短いあいさつ" : "次に確認する対象";
  return [
    `DIM: ${dim}`,
    `FOCUS: ${focus}`,
    `NEXT: ${constraint ? "条件を保った小さい案を選ぶ" : risk ? "ログと単一障害点を確認する" : greeting ? "ひとこと返す" : "条件を一つ確認する"}`,
    "WHY: 動作確認用の決定的なルール。",
  ].join("\n");
}

function latestUserText(request: ProviderTextRequest): string {
  return [...request.input].reverse().find((message) => message.role === "user")?.content ?? "";
}

export function createMockProvider(): LlmProvider {
  const model = process.env.MOCK_MODEL || "mock-gated-chat";
  return {
    name: "mock", model, capabilities: { tokenLogprobs: false },
    async createText(request: ProviderTextRequest): Promise<ProviderTextResponse> {
      const userText = latestUserText(request);
      // Explicit mock-only hooks for rollback and interrupted-request tests.
      const delay = Math.min(1000, Math.max(0, Number(process.env.SPIRAL_MOCK_DELAY_MS) || 0));
      if (delay) await new Promise((resolve) => setTimeout(resolve, delay));
      if (process.env.SPIRAL_MOCK_FAIL_PURPOSE === request.purpose && userText.includes("[mock:fail]")) {
        throw new Error(`Mock ${request.purpose} failure requested`);
      }
      let text = "";
      if (request.purpose === "probe") {
        text = mockProbe(userText);
      } else if (request.purpose === "explore") {
        text = [
          "DIM: NOVELTY\nFOCUS: まだ試していない方法\nNEXT: 手順を逆から考える\nWHY: 違う入口を試すため。",
          "DIM: GOAL\nFOCUS: 次の具体手順\nNEXT: 一番小さい確認を選ぶ\nWHY: 作業を前に進めるため。",
          "DIM: OPPORTUNITY\nFOCUS: 観察できる価値\nNEXT: 小さい試作品を見せる\nWHY: 変化を比較できるから。",
        ].join("\n\n");
      } else if (request.purpose === "verify") {
        text = "PICK: 2";
      } else if (request.purpose === "summary") {
        const current = userText.match(/\nUser: (.*)/)?.[1] ?? "";
        const previous = userText.match(/^Previous summary: (.*)/)?.[1] ?? "";
        text = `${previous && previous !== "(empty)" ? `${previous} / ` : ""}${current}`.slice(-140);
      } else {
        const frame = request.input.find((message) => message.content.startsWith("Current attention frame"))?.content ?? "";
        const focus = frame.match(/^FOCUS=(.*)$/m)?.[1] ?? "次の一歩";
        const next = frame.match(/^NEXT=(.*)$/m)?.[1] ?? "小さく試す";
        text = `［mock の返答例］${userText.slice(0, 80)}\n注目点: ${focus}\n次の一歩: ${next}`;
      }
      return {
        text, logprobs: [], model,
        // Character-count estimates, never billed or measured provider tokens.
        usage: {
          input_tokens: request.input.reduce((sum, message) => sum + Math.ceil(message.content.length / 4), 0),
          output_tokens: Math.ceil(text.length / 4),
        },
        requestId: `mock-${request.purpose}`, finishReason: "stop",
      };
    },
  };
}
