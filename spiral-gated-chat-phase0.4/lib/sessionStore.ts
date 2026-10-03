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
  history: StoredMessage[];
  turn: number;
};

const MAX_SESSIONS = 100;
const IDLE_TTL_MS = 30 * 60 * 1000;

/** Process-local LRU storage. Cleanup is lazy: no background timer is needed. */
export function createSessionStore({
  maxSessions = MAX_SESSIONS,
  idleTtlMs = IDLE_TTL_MS,
  now = Date.now,
}: { maxSessions?: number; idleTtlMs?: number; now?: () => number } = {}) {
  const sessions = new Map<string, { session: Session; lastUsedAt: number }>();
  const inFlight = new Set<string>();

  function touch(id: string) {
    const entry = sessions.get(id);
    if (!entry) return;
    entry.lastUsedAt = now();
    sessions.delete(id);
    sessions.set(id, entry); // insertion order is least-recently-used first
  }

  function pruneExpired() {
    const cutoff = now() - idleTtlMs;
    for (const [id, entry] of sessions) {
      if (!inFlight.has(id) && entry.lastUsedAt <= cutoff) sessions.delete(id);
    }
  }

  function acquireSession(id: string): "acquired" | "busy" | "full" {
    pruneExpired();
    if (inFlight.has(id)) return "busy";
    // Include reservations not materialized by getSession yet, so the cap also
    // holds between acquiring a slot and creating its session.
    const reserved = [...inFlight].filter((key) => !sessions.has(key)).length;
    if (!sessions.has(id) && sessions.size + reserved >= maxSessions) {
      const oldestIdle = [...sessions.keys()].find((key) => !inFlight.has(key));
      if (oldestIdle === undefined) return "full";
      sessions.delete(oldestIdle);
    }
    inFlight.add(id);
    touch(id);
    return "acquired";
  }

  function getSession(id: string, mode: "auto" | "fixed" = "auto"): Session {
    if (!inFlight.has(id)) throw new Error("Acquire the session before reading it");
    let entry = sessions.get(id);
    if (!entry) {
      entry = {
        session: {
          id, mode, gate: createInitialGateState(),
          memory: { summary: "", summary_updated_turn: -1, attn_log: [], fragments: [] },
          history: [], turn: 0,
        },
        lastUsedAt: now(),
      };
      sessions.set(id, entry);
    }
    return entry.session;
  }

  function commitSession(session: Session): void {
    if (!inFlight.has(session.id)) throw new Error("Acquire the session before committing it");
    sessions.set(session.id, { session, lastUsedAt: now() });
    touch(session.id);
  }

  function releaseSession(id: string): void {
    touch(id); // A long-running call starts its idle period when it finishes.
    inFlight.delete(id);
    pruneExpired();
  }

  return { acquireSession, getSession, commitSession, releaseSession };
}

// Turns still work on snapshots and commit only after required phases succeed.
export const { acquireSession, getSession, commitSession, releaseSession } = createSessionStore();
