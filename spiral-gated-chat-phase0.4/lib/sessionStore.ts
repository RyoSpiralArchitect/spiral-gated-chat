import { createInitialGateState } from "@/lib/gating";
import type { GateState, MemoryState } from "@/lib/types";

export type StoredMessage = {
  role: "user" | "assistant";
  content: string;
};

export type Session = {
  id: string;
  gate: GateState;
  memory: MemoryState;
  history: StoredMessage[]; // for context building
  turn: number;
};

const sessions = new Map<string, Session>();

export function getSession(sessionId: string): Session {
  let s = sessions.get(sessionId);
  if (!s) {
    s = {
      id: sessionId,
      gate: createInitialGateState(),
      memory: { summary: "", summary_updated_turn: -1, attn_log: [], fragments: [] },
      history: [],
      turn: 0,
    };
    sessions.set(sessionId, s);
  }
  return s;
}
