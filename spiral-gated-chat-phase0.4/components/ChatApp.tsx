"use client";

import { useEffect, useMemo, useState } from "react";

type Role = "user" | "assistant";

type ChatMessage = {
  id: string;
  role: Role;
  text: string;
};

type DebugPayload = {
  probeText: string | null;
  probeText_original?: string | null;
  dim: string | null;
  focus: string | null;
  next?: string | null;
  pulse?: {
    triggered: boolean;
    stagnation_detected: boolean;
    repeating_dim: string | null;
    candidates_text: string | null;
    picked: number | null;
    selected_probe: string | null;
  };
  memory?: {
    ctx_keep_msgs: number;
    summary_chars: number;
    attn_items: number;
    frag_items?: number;
    summary_update_interval: number;
    summary_update_max_tokens: number;
    fragments?: {
      total: number;
      decay_factor: number;
      top_salience: number;
      last_add: {
        added: boolean;
        merged: boolean;
        id: string;
        salience: number;
        text: string;
      } | null;
      injected: { id: string; turn: number; dim: string | null; salience: number; text: string }[];
      top: { id: string; turn: number; dim: string | null; salience: number; text: string }[];
    };
  };
  summary_used?: string | null;
  summary_stored?: string | null;
  metrics: {
    surprisal: number | null;
    entropy: number | null;
    score: number | null;
  };
  state: number;
  meta?: {
    meta_cap_stage: number;
    meta_cap: number | null;
    recent_meta_share: number | null;
  };
  params: {
    max_output_tokens: number;
    temperature: number;
    context_keep_msgs: number;
  };
  notes: string[];
};

function pretty(n: number | null | undefined, digits = 3): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return n.toFixed(digits);
}

export default function ChatApp() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [debug, setDebug] = useState<DebugPayload | null>(null);

  useEffect(() => {
    let id = window.localStorage.getItem("spiral_session_id");
    if (!id) {
      id = crypto.randomUUID();
      window.localStorage.setItem("spiral_session_id", id);
    }
    setSessionId(id);
  }, []);

  const statePct = useMemo(() => {
    const s = debug?.state ?? 0;
    return Math.max(0, Math.min(1, s)) * 100;
  }, [debug]);

  async function send() {
    if (!sessionId) return;
    const text = input.trim();
    if (!text) return;
    setInput("");
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", text };
    setMessages((m) => [...m, userMsg]);
    setBusy(true);

    try {
      const res = await fetch("/api/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, userText: text }),
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
      const data = (await res.json()) as { assistantText: string; debug: DebugPayload };
      const asstMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        text: data.assistantText,
      };
      setMessages((m) => [...m, asstMsg]);
      setDebug(data.debug);
    } catch (e: any) {
      const asstMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        text:
          "[error] API呼び出しに失敗しました。サーバログ／APIキー／モデル名を確認してね。\n" +
          String(e?.message ?? e),
      };
      setMessages((m) => [...m, asstMsg]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ display: "flex", height: "100vh", width: "100vw" }}>
      {/* Chat */}
      <div style={{ flex: 1.2, borderRight: "1px solid #eee", display: "flex", flexDirection: "column" }}>
        <div style={{ padding: 16, borderBottom: "1px solid #eee" }}>
          <div style={{ fontWeight: 700 }}>Spiral Gated Chat — Phase0.4 (Salience fragments)</div>
          <div style={{ fontSize: 12, color: "#666" }}>session: {sessionId ?? "…"}</div>
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
          {messages.length === 0 ? (
            <div style={{ color: "#888", fontSize: 14 }}>
              例: 「交差点。人が増えた。信号音。風。」 / 「この設計、どこが死にやすい？」
            </div>
          ) : null}

          {messages.map((m) => (
            <div key={m.id} style={{ marginBottom: 12 }}>
              <div style={{ fontSize: 12, color: "#666", marginBottom: 4 }}>{m.role}</div>
              <div
                style={{
                  whiteSpace: "pre-wrap",
                  background: m.role === "user" ? "#f7f7ff" : "#f7fff7",
                  border: "1px solid #eee",
                  borderRadius: 12,
                  padding: 12,
                }}
              >
                {m.text}
              </div>
            </div>
          ))}
        </div>

        <div style={{ padding: 16, borderTop: "1px solid #eee", display: "flex", gap: 8 }}>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="観測/質問を入力…"
            rows={2}
            style={{ flex: 1, resize: "none", padding: 10, borderRadius: 10, border: "1px solid #ddd" }}
            onKeyDown={(e) => {
              if ((e.ctrlKey || e.metaKey) && e.key === "Enter") send();
            }}
            disabled={busy}
          />
          <button
            onClick={send}
            disabled={busy || !sessionId}
            style={{
              width: 120,
              borderRadius: 10,
              border: "1px solid #ddd",
              background: busy ? "#f0f0f0" : "#fff",
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            {busy ? "…" : "Send"}
          </button>
        </div>
      </div>

      {/* Debug */}
      <div style={{ flex: 0.8, display: "flex", flexDirection: "column" }}>
        <div style={{ padding: 16, borderBottom: "1px solid #eee" }}>
          <div style={{ fontWeight: 700 }}>state / probe</div>
          <div style={{ fontSize: 12, color: "#666" }}>
            state = {pretty(debug?.state ?? 0)}
          </div>
          <div style={{ marginTop: 8, height: 10, background: "#f0f0f0", borderRadius: 999 }}>
            <div style={{ width: `${statePct}%`, height: 10, background: "#111", borderRadius: 999 }} />
          </div>
        </div>

        <div style={{ flex: 1, overflow: "auto", padding: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>DIM</div>
              <div style={{ fontSize: 18, fontWeight: 700 }}>{debug?.dim ?? "—"}</div>
            </div>
            <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>FOCUS</div>
              <div style={{ fontSize: 14, fontWeight: 600, whiteSpace: "pre-wrap" }}>{debug?.focus ?? "—"}</div>
            </div>
            <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>NEXT</div>
              <div style={{ fontSize: 14, fontWeight: 600, whiteSpace: "pre-wrap" }}>{debug?.next ?? "—"}</div>
            </div>
            <div style={{ border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>memory / pulse</div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                ctx msgs: {debug?.memory?.ctx_keep_msgs ?? debug?.params.context_keep_msgs ?? "—"}
                <br />
                summary chars: {debug?.memory?.summary_chars ?? "—"}
                <br />
                attn items: {debug?.memory?.attn_items ?? "—"}
                <br />
                frag items: {debug?.memory?.frag_items ?? "—"}
                <br />
                summary interval: {debug?.memory?.summary_update_interval ?? "—"} / maxTok{" "}
                {debug?.memory?.summary_update_max_tokens ?? "—"}
              </div>
              {debug?.memory?.fragments ? (
                <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
                  fragments: {debug.memory.fragments.total} (decay×{pretty(debug.memory.fragments.decay_factor, 2)})
                  <br />
                  top salience: {pretty(debug.memory.fragments.top_salience, 2)}
                </div>
              ) : null}
              <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                pulse: {debug?.pulse?.triggered ? "ON" : "off"}
              </div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 4 }}>
                META cap: stage {debug?.meta?.meta_cap_stage ?? "—"} /{" "}
                {debug?.meta
                  ? debug.meta.meta_cap === null
                    ? "unlocked"
                    : `cap ${debug.meta.meta_cap}`
                  : "—"}
                {debug?.meta?.recent_meta_share !== null && debug?.meta?.recent_meta_share !== undefined
                  ? ` (share ${pretty(debug.meta.recent_meta_share, 2)})`
                  : ""}
              </div>
            </div>
          </div>

          <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
            <div style={{ fontSize: 12, color: "#666" }}>metrics</div>
            <div style={{ fontSize: 13, marginTop: 6 }}>
              surprisal: {pretty(debug?.metrics.surprisal)}
              <br />
              entropy: {pretty(debug?.metrics.entropy)}
              <br />
              score: {pretty(debug?.metrics.score)}
            </div>
          </div>

          <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
            <div style={{ fontSize: 12, color: "#666" }}>mapped params (main)</div>
            <div style={{ fontSize: 13, marginTop: 6 }}>
              max_output_tokens: {debug?.params.max_output_tokens ?? "—"}
              <br />
              temperature: {pretty(debug?.params.temperature, 2)}
              <br />
              context_keep_msgs: {debug?.params.context_keep_msgs ?? "—"}
            </div>
          </div>

          {debug?.summary_used || debug?.summary_stored ? (
            <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>summary</div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>used in prompt:</div>
              <div style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{debug?.summary_used ?? "—"}</div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 8 }}>stored:</div>
              <div style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>{debug?.summary_stored ?? "—"}</div>
            </div>
          ) : null}

          {debug?.pulse ? (
            <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>exploration pulse</div>
              <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
                stagnation: {debug.pulse.stagnation_detected ? "yes" : "no"}
                <br />
                repeating_dim: {debug.pulse.repeating_dim ?? "—"}
                <br />
                triggered: {debug.pulse.triggered ? "YES" : "no"}
                <br />
                picked: {debug.pulse.picked ?? "—"}
              </div>
              {debug.pulse.selected_probe ? (
                <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, margin: "8px 0 0 0" }}>
                  {debug.pulse.selected_probe}
                </pre>
              ) : null}
            </div>
          ) : null}

          {debug?.memory?.fragments ? (
            <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>memory fragments (salience)</div>
              {debug.memory.fragments.last_add ? (
                <div style={{ fontSize: 12, color: "#666", marginTop: 6 }}>
                  last add: {debug.memory.fragments.last_add.added ? "added" : ""}
                  {debug.memory.fragments.last_add.merged ? "merged" : ""}
                  {" "}s={pretty(debug.memory.fragments.last_add.salience, 2)}
                </div>
              ) : null}

              <div style={{ marginTop: 8, fontSize: 12, color: "#666" }}>injected this turn:</div>
              {debug.memory.fragments.injected?.length ? (
                <ul style={{ margin: "6px 0 0 18px", padding: 0, fontSize: 12 }}>
                  {debug.memory.fragments.injected.map((f) => (
                    <li key={f.id}>
                      {f.dim ? `(${f.dim}) ` : ""}
                      {f.text} <span style={{ color: "#999" }}>(s={pretty(f.salience, 2)})</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div style={{ fontSize: 12, color: "#999", marginTop: 4 }}>(none)</div>
              )}

              <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>top of bank:</div>
              {debug.memory.fragments.top?.length ? (
                <ul style={{ margin: "6px 0 0 18px", padding: 0, fontSize: 12 }}>
                  {debug.memory.fragments.top.map((f) => (
                    <li key={f.id}>
                      {f.dim ? `(${f.dim}) ` : ""}
                      {f.text} <span style={{ color: "#999" }}>(s={pretty(f.salience, 2)})</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div style={{ fontSize: 12, color: "#999", marginTop: 4 }}>(empty)</div>
              )}
            </div>
          ) : null}

          <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
            <div style={{ fontSize: 12, color: "#666" }}>probe (effective)</div>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, margin: 0 }}>{debug?.probeText ?? "—"}</pre>
          </div>

          {debug?.probeText_original && debug.probeText_original !== debug.probeText ? (
            <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>probe (original)</div>
              <pre style={{ whiteSpace: "pre-wrap", fontSize: 12, margin: 0 }}>{debug.probeText_original}</pre>
            </div>
          ) : null}

          {debug?.notes?.length ? (
            <div style={{ marginTop: 12, border: "1px solid #eee", borderRadius: 12, padding: 12 }}>
              <div style={{ fontSize: 12, color: "#666" }}>notes</div>
              <ul style={{ margin: "6px 0 0 18px", padding: 0, fontSize: 12 }}>
                {debug.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
