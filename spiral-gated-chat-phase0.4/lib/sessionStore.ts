import { createInitialGateState } from "@/lib/gating";
import type { GateState, MemoryState } from "@/lib/types";

export type StoredMessage = {
  role: "user" | "assistant";
  content: string;
};

export type Session = {
  id: string;
  mode: "auto" | "fixed";
  gate: GateState;
  memory: MemoryState;
  history: StoredMessage[]; // for context building
  turn: number;
};

const sessions = new Map<string, Session>();

export function getSession(sessionId: string, mode: "auto" | "fixed" = "auto"): Session {
  let s = sessions.get(sessionId);
  if (!s) {
    s = {
      id: sessionId,
      mode,
      gate: createInitialGateState(),
      memory: { summary: "", summary_updated_turn: -1, attn_log: [], fragments: [] },
      history: [],
      turn: 0,
    };
    sessions.set(sessionId, s);
  }
  return s;
}

// A turn works on a snapshot and commits only after Main succeeds. Concurrent
// requests for the same session fail fast rather than mixing histories.
const inFlight = new Set<string>();
export function acquireSession(sessionId: string): boolean {
  if (inFlight.has(sessionId)) return false;
  inFlight.add(sessionId);
  return true;
}
export function releaseSession(sessionId: string): void { inFlight.delete(sessionId); }
export function commitSession(session: Session): void { sessions.set(session.id, session); }
