"use client";

import { useEffect, useRef, useState } from "react";
import { scenarios } from "@/lib/scenarios";
import TurnDetails, { AccountingDetails } from "./TurnDetails";
import { decimal, summarizeAccounting, tokenLabel, type Accounting, type FailedTurn, type GateMode, type ProviderCall, type PublicConfig, type TurnResult } from "./chat-types";

type RunStatus = "idle" | "running" | "complete" | "stopped" | "error";
type ComparisonRun = {
  id: string;
  scenarioId: string;
  sessions: Record<GateMode, string>;
  turns: Record<GateMode, TurnResult[]>;
  status: RunStatus;
  error?: FailedTurn;
};
type Selection = { scope: "comparison" | "manual"; mode: GateMode; turn: number };
type PendingTurn = { scope: "comparison" | "manual"; mode: GateMode; turn: number; userText: string };

class StepError extends Error {
  accounting?: Accounting;
  calls?: ProviderCall[];
  constructor(message: string, accounting?: Accounting, calls?: ProviderCall[]) {
    super(message);
    this.name = "StepError";
    this.accounting = accounting;
    this.calls = calls;
  }
}

async function requestTurn(payload: { sessionId: string; userText: string; mode: GateMode; expectedTurn: number; comparisonId?: string; scenarioId?: string }, signal: AbortSignal): Promise<TurnResult> {
  const response = await fetch("/api/step", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload), signal,
  });
  const data = await response.json().catch(() => null);
  if (!response.ok) throw new StepError(data?.error || `応答を取得できませんでした (HTTP ${response.status})`, data?.accounting, data?.calls);
  if (!data?.debug || typeof data.assistantText !== "string" || !Number.isInteger(data.turn)) throw new StepError("応答の形式を確認できませんでした。新しい実行でやり直してください。");
  return data as TurnResult;
}

function SpiralMark() {
  return <svg viewBox="0 0 36 36" fill="none" aria-hidden="true"><path d="M28.5 24.5c-3.8 6.5-14 8.8-20 2.6C2.2 20.8 5.2 8.9 13.9 6.2c7.2-2.3 14.8 2.4 14.8 9.7 0 6-5.3 10.3-10.8 8.6-4.4-1.3-6.4-6.5-3.3-9.9 2.2-2.5 6.6-1.5 7.1 1.7.3 1.8-1.1 3.1-2.8 2.5" stroke="currentColor" strokeWidth="2.1" strokeLinecap="round" /></svg>;
}

function Arrow({ down = false }: { down?: boolean }) {
  return <span aria-hidden="true">{down ? "↓" : "↗"}</span>;
}

function TurnCard({ result, selected, onSelect }: { result: TurnResult; selected: boolean; onSelect: () => void }) {
  const d = result.debug;
  const fragments = d.memory?.fragments?.injected?.length ?? 0;
  return <button type="button" className={`turn-card ${selected ? "selected" : ""}`} onClick={onSelect} aria-pressed={selected} aria-label={`${result.mode === "auto" ? "自動" : "固定"} ターン ${result.turn} の詳細を表示`} data-testid={`${result.mode}-turn-${result.turn}`}>
    <span className="turn-topline"><span>TURN {String(result.turn).padStart(2, "0")}</span><span className="turn-state">state <strong>{decimal(d.state)}</strong></span></span>
    <span className="user-utterance">{result.userText}</span>
    <span className="assistant-label"><span className="tiny-spiral" aria-hidden="true">◎</span> 応答</span>
    <span className="assistant-response">{result.assistantText || "（空の応答）"}</span>
    <span className="turn-focus"><span className="dim-pill">{d.dim ?? "—"}</span><span>{d.focus ?? "注目点なし"}</span></span>
    <span className="turn-footer"><span>記憶 {fragments} 件 · 会話 {d.memory?.context_used?.length ?? "—"} 件</span><span className={d.viewpoint?.changed ? "shift-tag" : ""}>{d.viewpoint?.changed ? "↗ 視点変化" : d.pulse?.triggered ? "◎ 探索パルス" : "詳細を見る →"}</span></span>
    {d.pulse?.triggered && d.viewpoint?.changed ? <span className="pulse-note">◎ 探索パルス {d.viewpoint.pulse_changed ? "· 視点が変化" : "· 選択後の視点は同じ"}</span> : null}
  </button>;
}

function PendingCard({ pending }: { pending: PendingTurn }) {
  return <div className="pending-card" role="status"><span className="turn-topline">TURN {String(pending.turn).padStart(2, "0")}<span className="working-dots"><i /><i /><i /></span></span><p>{pending.userText}</p><span>Probe → 記憶を選択 → 応答を生成</span></div>;
}

function FailureCard({ error }: { error: FailedTurn }) {
  return <div className="failure-card" role="alert"><strong>ターン {error.turn} で実行が止まりました</strong><p className="failed-prompt">{error.userText}</p><p>{error.message}</p><small>完了した結果は残っています。自動で再試行はしません。応答を取得できなかった呼び出しの使用量は不明で、表示中の累計は全量を表さない場合があります。</small>{error.accounting ? <details><summary>失敗までの呼び出しを確認</summary><AccountingDetails accounting={error.accounting} calls={error.calls} /></details> : null}</div>;
}

function LaneSummary({ turns, error }: { turns: TurnResult[]; error?: FailedTurn }) {
  const accounting = summarizeAccounting([...turns.map((turn) => turn.debug.accounting), ...(error?.accounting ? [error.accounting] : [])]);
  const calls = [...turns.flatMap((turn) => turn.debug.provider?.calls ?? []), ...(error?.calls ?? [])];
  const isMock = accounting?.usage_kind === "mock_estimate" || Boolean(calls.length && calls.every((call) => call.provider === "mock"));
  return <div className="lane-summary" aria-label="この実行の累計"><div><span>呼び出し</span><strong>{accounting?.call_count ?? "—"}<small> 回</small></strong></div><div><span>{isMock ? "推定 tokens" : "合計 tokens"}</span><strong>{tokenLabel(accounting, isMock)}</strong></div><div><span>処理時間</span><strong>{accounting ? decimal(accounting.turn_latency_ms / 1000, 2) : "—"}<small> 秒</small></strong></div></div>;
}

function ComparisonLane({ mode, turns, total, selected, pending, error, onSelect }: { mode: GateMode; turns: TurnResult[]; total: number; selected: Selection | null; pending: PendingTurn | null; error?: FailedTurn; onSelect: (result: TurnResult) => void }) {
  return <section className={`comparison-lane ${mode}`} aria-label={mode === "fixed" ? "固定ゲートの結果" : "自動ゲートの結果"}>
    <div className="lane-heading"><div className="lane-name"><span className="lane-symbol" aria-hidden="true">{mode === "fixed" ? "＝" : "↗"}</span><div><h2>{mode === "fixed" ? "固定ゲート" : "自動ゲート"}</h2><p>{mode === "fixed" ? "state = 0.50 · 探索なし" : "state に応じて設定が変わる"}</p></div></div><span className="turn-count">{turns.length}<span> / {total}</span></span></div>
    <LaneSummary turns={turns} error={error} />
    {turns.length > 0 ? <div className="state-history" aria-label="state の履歴">{turns.map((turn) => <button key={turn.turn} title={`ターン ${turn.turn}: state ${decimal(turn.debug.state)}`} aria-label={`ターン ${turn.turn}、state ${decimal(turn.debug.state)} の詳細`} onClick={() => onSelect(turn)}><span style={{ height: `${Math.max(5, turn.debug.state * 100)}%` }} /><small>{turn.turn}</small></button>)}</div> : null}
    <div className="lane-turns">{turns.map((result) => <TurnCard key={`${result.sessionId}-${result.turn}`} result={result} selected={selected?.scope === "comparison" && selected.mode === mode && selected.turn === result.turn} onSelect={() => onSelect(result)} />)}
      {pending?.scope === "comparison" && pending.mode === mode ? <PendingCard pending={pending} /> : null}
      {error ? <FailureCard error={error} /> : null}
      {!turns.length && !(pending?.scope === "comparison" && pending.mode === mode) && !error ? <div className="lane-empty"><div className="empty-line long" /><div className="empty-line" /><div className="empty-line short" /><p>{mode === "fixed" ? "設定を固定した会話が\nここに並びます" : "ゲートが変化する会話が\nここに並びます"}</p><span>同じ入力 · 独立したセッション</span></div> : null}
    </div>
  </section>;
}

export default function ChatApp() {
  const [view, setView] = useState<"comparison" | "manual">("comparison");
  const [scenarioId, setScenarioId] = useState(scenarios[0].id);
  const [config, setConfig] = useState<PublicConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [configLoading, setConfigLoading] = useState(true);
  const [comparison, setComparison] = useState<ComparisonRun | null>(null);
  const [manualTurns, setManualTurns] = useState<TurnResult[]>([]);
  const [manualMode, setManualMode] = useState<GateMode>("auto");
  const [manualSession, setManualSession] = useState<string | null>(null);
  const [manualError, setManualError] = useState<FailedTurn | null>(null);
  const [manualInterrupted, setManualInterrupted] = useState(false);
  const [input, setInput] = useState("");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [busy, setBusy] = useState(false);
  const operation = useRef({ generation: 0, busy: false, controller: null as AbortController | null });
  const manualSessionRef = useRef<string | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  async function loadConfig(signal?: AbortSignal) {
    setConfigLoading(true);
    setConfigError(null);
    try {
      const response = await fetch("/api/config", { cache: "no-store", signal });
      if (!response.ok) throw new Error("実行環境を取得できませんでした");
      const data = await response.json();
      if (typeof data.provider !== "string" || typeof data.model !== "string" || typeof data.isMock !== "boolean") throw new Error("実行環境の情報が不完全です");
      if (!signal?.aborted) setConfig(data);
    } catch (error) {
      if (!signal?.aborted) { setConfig(null); setConfigError(error instanceof Error ? error.message : "実行環境を取得できませんでした"); }
    } finally {
      if (!signal?.aborted) setConfigLoading(false);
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    void loadConfig(controller.signal);
    return () => { controller.abort(); operation.current.generation += 1; operation.current.controller?.abort(); };
  }, []);

  const scenario = scenarios.find((item) => item.id === scenarioId) ?? scenarios[0];
  const displayedScenario = scenarios.find((item) => item.id === comparison?.scenarioId) ?? scenario;
  const selectedResult = selected?.scope === "comparison" ? comparison?.turns[selected.mode].find((turn) => turn.turn === selected.turn) ?? null : selected?.scope === "manual" ? manualTurns.find((turn) => turn.turn === selected.turn) ?? null : null;
  const hasResults = view === "comparison" ? Boolean(comparison && (comparison.turns.fixed.length || comparison.turns.auto.length || comparison.error)) : Boolean(manualTurns.length || manualError);

  function beginOperation() {
    if (operation.current.busy) return null;
    operation.current.busy = true;
    operation.current.generation += 1;
    const generation = operation.current.generation;
    const controller = new AbortController();
    operation.current.controller = controller;
    setBusy(true);
    return { generation, controller };
  }

  function stopOperation() {
    const wasBusy = operation.current.busy;
    operation.current.generation += 1;
    operation.current.controller?.abort();
    operation.current.controller = null;
    operation.current.busy = false;
    setBusy(false);
    setPending(null);
    if (wasBusy) {
      setComparison((current) => current?.status === "running" ? { ...current, status: "stopped" } : current);
      if (view === "manual") setManualInterrupted(true);
    }
  }

  function finishOperation(generation: number) {
    if (operation.current.generation !== generation) return;
    operation.current.busy = false;
    operation.current.controller = null;
    setBusy(false);
    setPending(null);
  }

  function reset() {
    stopOperation();
    setSelected(null);
    if (view === "comparison") setComparison(null);
    else {
      const sessionId = crypto.randomUUID();
      manualSessionRef.current = sessionId;
      setManualSession(sessionId);
      setManualTurns([]);
      setManualError(null);
      setManualInterrupted(false);
      setInput("");
      inputRef.current?.focus();
    }
  }

  async function runComparison() {
    if (!config || configLoading) return;
    const active = beginOperation();
    if (!active) return;
    const run: ComparisonRun = { id: crypto.randomUUID(), scenarioId: scenario.id, sessions: { fixed: crypto.randomUUID(), auto: crypto.randomUUID() }, turns: { fixed: [], auto: [] }, status: "running" };
    setComparison(run);
    setSelected(null);
    let attempted: PendingTurn | null = null;
    try {
      for (const [index, userText] of scenario.turns.entries()) {
        for (const mode of ["fixed", "auto"] as const) {
          if (operation.current.generation !== active.generation) return;
          attempted = { scope: "comparison", mode, turn: index + 1, userText };
          setPending(attempted);
          const result = await requestTurn({ sessionId: run.sessions[mode], userText, mode, expectedTurn: index + 1, comparisonId: run.id, scenarioId: scenario.id }, active.controller.signal);
          if (operation.current.generation !== active.generation) return;
          setComparison((current) => current?.id === run.id ? { ...current, turns: { ...current.turns, [mode]: [...current.turns[mode], result] } } : current);
          setSelected((current) => current ?? { scope: "comparison", mode, turn: result.turn });
        }
      }
      if (operation.current.generation === active.generation) setComparison((current) => current?.id === run.id ? { ...current, status: "complete" } : current);
    } catch (error) {
      if (operation.current.generation !== active.generation) return;
      const failure: FailedTurn = { mode: attempted?.mode ?? "fixed", turn: attempted?.turn ?? 1, userText: attempted?.userText ?? "", message: error instanceof Error ? error.message : String(error), accounting: error instanceof StepError ? error.accounting : undefined, calls: error instanceof StepError ? error.calls : undefined };
      setComparison((current) => current?.id === run.id ? { ...current, status: "error", error: failure } : current);
    } finally { finishOperation(active.generation); }
  }

  async function sendManual() {
    const userText = input.trim();
    if (!config || configLoading || !userText || manualInterrupted || manualError) return;
    const active = beginOperation();
    if (!active) return;
    const sessionId = manualSessionRef.current ?? crypto.randomUUID();
    manualSessionRef.current = sessionId;
    setManualSession(sessionId);
    const turn = manualTurns.length + 1;
    setPending({ scope: "manual", mode: manualMode, turn, userText });
    try {
      const result = await requestTurn({ sessionId, userText, mode: manualMode, expectedTurn: turn }, active.controller.signal);
      if (operation.current.generation !== active.generation) return;
      setManualTurns((current) => [...current, result]);
      setSelected({ scope: "manual", mode: manualMode, turn: result.turn });
      setInput("");
    } catch (error) {
      if (operation.current.generation !== active.generation) return;
      setManualError({ mode: manualMode, turn, userText, message: error instanceof Error ? error.message : String(error), accounting: error instanceof StepError ? error.accounting : undefined, calls: error instanceof StepError ? error.calls : undefined });
    } finally { finishOperation(active.generation); }
  }

  function switchView(next: "comparison" | "manual") {
    if (operation.current.busy) return;
    setView(next);
    setSelected(null);
  }

  function downloadRun() {
    const data = { exportedAt: new Date().toISOString(), config, note: "表示中の実行結果のエクスポート。会話の復元機能ではありません。", ...(view === "comparison" ? { comparison, scenario: displayedScenario } : { sessionId: manualSession, mode: manualMode, turns: manualTurns, error: manualError, interrupted: manualInterrupted }) };
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `spiral-${view}-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <div className="app-shell">
    <header className="topbar"><a className="brand" href="/" aria-label="Spiral ホーム"><span className="brand-mark"><SpiralMark /></span><span>spiral<span className="brand-divider" /><small>GATED CHAT LAB</small></span></a><div className="environment" role="status"><span className={`status-dot ${config ? "ready" : ""}`} /><span>{configLoading ? "実行環境を確認中" : config ? `${config.isMock ? "MOCK" : config.provider.toUpperCase()} · ${config.model}` : "実行環境を取得できません"}</span><span className="version">PHASE 0.4</span></div></header>
    <main>
      <section className="intro"><div className="intro-copy"><div className="section-kicker"><span className="eyebrow-line" /> OBSERVE THE CONVERSATION</div><h1>会話の、その奥を見る<span>。</span></h1><p>いま何に注目し、どの記憶を使ったのか。<br className="mobile-break" />応答と一緒に、会話の変化をたどる実験室。</p></div><div className="process-note" aria-label="処理の流れ"><div><span>01</span><strong>Probe</strong><small>注目点を見つける</small></div><Arrow /><div><span>02</span><strong>Gate</strong><small>使う設定を調整</small></div><Arrow /><div><span>03</span><strong>Response</strong><small>応答と記録</small></div></div></section>
      {configError ? <div className="config-error" role="alert"><span>{configError}。確認できるまで実行できません。</span><button className="text-button" onClick={() => void loadConfig()} disabled={configLoading}>再取得</button></div> : null}
      <nav className="workspace-tabs" aria-label="実験モード"><button className={view === "comparison" ? "active" : ""} aria-current={view === "comparison" ? "page" : undefined} onClick={() => switchView("comparison")} disabled={busy}>比較ラボ <span>COMPARE</span></button><button className={view === "manual" ? "active" : ""} aria-current={view === "manual" ? "page" : undefined} onClick={() => switchView("manual")} disabled={busy}>自由に話す <span>CHAT</span></button><span className="tab-caption">{view === "comparison" ? "同じ会話、2つのゲート" : "一つの会話を、じっくり観察"}</span></nav>
      {view === "comparison" ? <section className="scenario-panel" aria-labelledby="scenario-heading"><div className="section-heading"><h2 id="scenario-heading"><span className="step-number">1</span>会話のシナリオを選ぶ</h2><span>両側に同じ入力を順番に送ります</span></div><div className="scenario-grid">{scenarios.map((item) => <button className={`scenario-card ${scenarioId === item.id ? "active" : ""}`} key={item.id} aria-pressed={scenarioId === item.id} disabled={busy} onClick={() => setScenarioId(item.id)}><span className="scenario-top"><span>EXPERIMENT {item.number}</span><span>{item.turns.length} TURNS</span></span><strong>{item.title}</strong><span className="scenario-description">{item.description}</span><span className="scenario-select" aria-hidden="true">{scenarioId === item.id ? "✓" : "↗"}</span></button>)}</div><div className="run-controls"><div className="run-description"><span className="field-label">観察のヒント</span><p>{scenario.watchFor}</p><details className="script-preview"><summary>入力する{scenario.turns.length}ターンを見る</summary><ol>{scenario.turns.map((text, index) => <li key={index}>{text}</li>)}</ol></details></div><button className="primary-button" onClick={() => void runComparison()} disabled={busy || !config || configLoading} data-testid="run-comparison">{busy ? "比較を実行中…" : comparison ? "新しい比較を実行" : "比較をはじめる"}<span aria-hidden="true">→</span></button></div><div className="method-note"><span aria-hidden="true">ⓘ</span><p>両側で同じprovider・modelとProbe・記憶・要約の処理を使います。固定側はstate 0.50、探索なし。Probeを省く比較ではありません。</p></div></section> : <section className="manual-setup"><div><h2>自由に話して、変化を追う</h2><p>途中の応答を選んで、その時点で使われた記憶を振り返れます。</p></div><label>ゲートの設定<select value={manualMode} onChange={(event) => setManualMode(event.target.value as GateMode)} disabled={busy || manualTurns.length > 0 || Boolean(manualError) || manualInterrupted}><option value="auto">自動ゲート</option><option value="fixed">固定 state 0.50</option></select></label></section>}
      <div className="results-heading"><div><h2><span className="step-number">{view === "comparison" ? "2" : "↗"}</span>{view === "comparison" ? "応答を見比べる" : "会話の記録"}</h2>{view === "comparison" && comparison ? <span className={`run-status ${comparison.status}`} role="status">{comparison.status === "running" ? `${pending?.turn ?? 1} / ${displayedScenario.turns.length} ターンを実行中` : comparison.status === "complete" ? `${displayedScenario.turns.length} ターン完了` : comparison.status === "stopped" ? "中断 · 完了した結果を表示" : "一部失敗 · 完了した結果を表示"}</span> : <span>カードを選ぶと詳細を表示</span>}</div><div className="result-actions">{busy ? <button className="stop-button" onClick={stopOperation}>■ 実行を止める</button> : null}<button className="text-button" onClick={downloadRun} disabled={!hasResults}>JSONを保存 <Arrow down /></button><button className="text-button" onClick={reset}>{view === "comparison" ? "リセット" : "新しい会話"} <span aria-hidden="true">↻</span></button></div></div>
      {view === "comparison" && comparison?.status === "stopped" ? <p className="run-notice" role="status">中断した実行は再開しません。次の比較は新しいセッションで始まります。表示中の累計は完了済み応答の分だけで、中断した呼び出しの使用量は不明です。サーバー処理が完了している可能性もあります。</p> : null}
      {view === "comparison" && comparison && comparison.scenarioId !== scenarioId ? <p className="run-notice">表示中の結果: {displayedScenario.title}。上で選んだシナリオは次の実行に使います。</p> : null}
      <div className="workspace"><div className="conversation-area">{view === "comparison" ? <div className="comparison-grid">{(["fixed", "auto"] as const).map((mode) => <ComparisonLane key={mode} mode={mode} turns={comparison?.turns[mode] ?? []} total={displayedScenario.turns.length} selected={selected} pending={pending} error={comparison?.error?.mode === mode ? comparison.error : undefined} onSelect={(result) => setSelected({ scope: "comparison", mode, turn: result.turn })} />)}</div> : <section className="manual-chat" aria-label="自由な会話"><div className="manual-chat-heading"><span className={`mode-badge ${manualMode}`}>{manualMode === "auto" ? "自動ゲート" : "固定 0.50"}</span><span>{manualTurns.length} TURNS</span></div>{manualTurns.length || manualError ? <LaneSummary turns={manualTurns} error={manualError ?? undefined} /> : null}<div className="manual-turns">{manualTurns.map((result) => <TurnCard key={`${result.sessionId}-${result.turn}`} result={result} selected={selected?.scope === "manual" && selected.turn === result.turn} onSelect={() => setSelected({ scope: "manual", mode: result.mode, turn: result.turn })} />)}{pending?.scope === "manual" ? <PendingCard pending={pending} /> : null}{manualError ? <FailureCard error={manualError} /> : null}{!manualTurns.length && !pending && !manualError ? <div className="manual-empty"><span className="empty-spiral"><SpiralMark /></span><h3>小さな問いから、はじめよう。</h3><p>考えていることや、気になっていることをひとつ。</p><button onClick={() => { setInput("考えがまとまりません。最初の一歩を一緒に考えて。"); inputRef.current?.focus(); }}>考えを整理したい <Arrow /></button><button onClick={() => { setInput("この設計で壊れやすい部分を見つけたい。何から確認すればいい？"); inputRef.current?.focus(); }}>設計を見直したい <Arrow /></button></div> : null}</div>{manualInterrupted || manualError ? <div className="manual-interrupted" role="status">このセッションは中断されています。「新しい会話」で始め直してください。応答未取得の呼び出しは使用量が不明で、表示中の累計に含まれない場合があります。</div> : null}<form className="composer" onSubmit={(event) => { event.preventDefault(); void sendManual(); }}><label htmlFor="chat-input" className="sr-only">メッセージ</label><textarea id="chat-input" ref={inputRef} value={input} rows={3} onChange={(event) => setInput(event.target.value)} placeholder="考えていることを、ここに。" disabled={busy || manualInterrupted || Boolean(manualError)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !event.nativeEvent.isComposing) { event.preventDefault(); void sendManual(); } }} /><div className="composer-footer"><span>⌘ / Ctrl + Enter で送信</span><button type="submit" className="primary-button" disabled={busy || !input.trim() || !config || configLoading || manualInterrupted || Boolean(manualError)}>{busy ? "生成中…" : "送信"} <span aria-hidden="true">↑</span></button></div></form></section>}</div><TurnDetails key={view} result={selectedResult} /></div>
      <footer className="page-footer"><p><strong>観察するためのプロトタイプ。</strong>stateはProbeのDIM行の信号に基づく制御値で、思考の深さの測定値ではありません。{config?.isMock ? "Mockの応答・トークン概算・ローカル処理時間は動作確認用です。品質・費用削減の評価には使えません。" : "呼び出しにはProbe・探索・要約も含みます。設定の違いから、品質や費用削減は断定できません。"}</p><div><span>毎回、新しいセッションで比較</span><span>画面の履歴は再読込でリセット</span></div></footer>
    </main>
  </div>;
}
