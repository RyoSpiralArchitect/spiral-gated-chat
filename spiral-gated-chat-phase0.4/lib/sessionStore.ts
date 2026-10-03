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
  const provisional = new Map<string, Session>();

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

  function acquireSession(id: string, expectedTurn?: number): "acquired" | "busy" | "full" | "stale" {
    pruneExpired();
    if (inFlight.has(id)) return "busy";
    // Reject unknown/expired/stale continuations before reserving a slot or
    // evicting another conversation. The lock below makes this check atomic.
    const nextTurn = (sessions.get(id)?.session.turn ?? 0) + 1;
    if (expectedTurn !== undefined && expectedTurn !== nextTurn) return "stale";
    // Bound active working turns separately from the retained-session cache.
    // A new turn stays provisional and cannot evict anything until it commits.
    if (inFlight.size >= maxSessions) return "full";
    inFlight.add(id);
    touch(id);
    return "acquired";
  }

  function getSession(id: string, mode: "auto" | "fixed" = "auto"): Session {
    if (!inFlight.has(id)) throw new Error("Acquire the session before reading it");
    const entry = sessions.get(id);
    if (entry) return entry.session;
    let session = provisional.get(id);
    if (!session) {
      session = {
        id, mode, gate: createInitialGateState(),
        memory: { summary: "", summary_updated_turn: -1, attn_log: [], fragments: [] },
        history: [], turn: 0,
      };
      provisional.set(id, session);
    }
    return session;
  }

  function commitSession(session: Session): void {
    if (!inFlight.has(session.id)) throw new Error("Acquire the session before committing it");
    if (!sessions.has(session.id) && sessions.size >= maxSessions) {
      const oldestIdle = [...sessions.keys()].find((id) => !inFlight.has(id));
      // A new in-flight turn occupies a slot, so at most maxSessions-1 retained
      // sessions can be locked. A full cache must therefore have an idle entry.
      if (oldestIdle === undefined) throw new Error("No idle session available for commit");
      sessions.delete(oldestIdle);
    }
    sessions.set(session.id, { session, lastUsedAt: now() });
    provisional.delete(session.id);
    touch(session.id);
  }

  function releaseSession(id: string): void {
    provisional.delete(id); // Failed first turns leave neither state nor a pinned mode.
    touch(id); // A long-running call starts its idle period when it finishes.
    inFlight.delete(id);
    pruneExpired();
  }

  return { acquireSession, getSession, commitSession, releaseSession };
}

// Turns still work on snapshots and commit only after required phases succeed.
export const { acquireSession, getSession, commitSession, releaseSession } = createSessionStore();
