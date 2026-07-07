import { access, appendFile, mkdir } from "node:fs/promises";
import path from "node:path";
import type { ProviderCallRecord, ProviderName, StateSource } from "@/lib/providers/types";

export type TurnLogPayload = {
  sessionId: string;
  turn: number;
  provider: ProviderName;
  model: string;
  stateSource: StateSource;
  latencyMs: number;
  calls: ProviderCallRecord[];
  userText: string;
  assistantText: string;
  debug: unknown;
};

export type TurnLogResult = {
  filePath: string;
  relativePath: string;
  sessionIndexPath: string;
  sessionIndexRelativePath: string;
};

function safeSessionId(sessionId: string): string {
  const safe = sessionId.replace(/[^a-zA-Z0-9_.-]/g, "_").slice(0, 80);
  return safe || "session";
}

function logDirectory(): string {
  const configured = process.env.SPIRAL_CHAT_LOG_DIR || "logs";
  return path.isAbsolute(configured) ? configured : path.join(process.cwd(), configured);
}

async function fileExists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

export async function appendTurnLog(payload: TurnLogPayload): Promise<TurnLogResult> {
  const dir = logDirectory();
  const sessionsDir = path.join(dir, "sessions");
  await mkdir(sessionsDir, { recursive: true });

  const safeId = safeSessionId(payload.sessionId);
  const filePath = path.join(sessionsDir, `${safeId}.jsonl`);
  const sessionIndexPath = path.join(dir, "session-index.jsonl");
  const isNewSessionLog = !(await fileExists(filePath));
  const entry = {
    schema_version: "spiral-gated-chat.turn.v1",
    ts: new Date().toISOString(),
    sessionId: payload.sessionId,
    turn: payload.turn,
    provider: payload.provider,
    model: payload.model,
    state_source: payload.stateSource,
    latency_ms: payload.latencyMs,
    calls: payload.calls,
    userText: payload.userText,
    assistantText: payload.assistantText,
    debug: payload.debug,
  };

  if (isNewSessionLog) {
    await appendFile(
      sessionIndexPath,
      `${JSON.stringify({
        schema_version: "spiral-gated-chat.session.v1",
        ts: entry.ts,
        sessionId: payload.sessionId,
        safe_session_id: safeId,
        provider: payload.provider,
        model: payload.model,
        first_turn: payload.turn,
        log_path: path.relative(process.cwd(), filePath) || filePath,
      })}\n`,
      "utf8"
    );
  }

  await appendFile(filePath, `${JSON.stringify(entry)}\n`, "utf8");

  return {
    filePath,
    relativePath: path.relative(process.cwd(), filePath) || filePath,
    sessionIndexPath,
    sessionIndexRelativePath: path.relative(process.cwd(), sessionIndexPath) || sessionIndexPath,
  };
}
