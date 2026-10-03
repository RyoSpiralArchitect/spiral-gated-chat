"use client";

import { useState } from "react";
import { decimal, tokenLabel, type Accounting, type ProviderCall, type TurnResult } from "./chat-types";

type DetailTab = "focus" | "memory" | "calls";

const purposeNames: Record<string, string> = { probe: "Probe · 注目点", main: "Main · 応答", explore: "Explore · 探索", verify: "Verify · 選択", summary: "Summary · 要約" };

export function AccountingDetails({ accounting, calls }: { accounting?: Accounting; calls?: ProviderCall[] }) {
  if (!accounting) return <p className="muted empty-note">使用量の情報はありません</p>;
  const isMock = accounting.usage_kind === "mock_estimate" || Boolean(calls?.length && calls.every((call) => call.provider === "mock"));
  return (
    <div className="detail-stack">
      <div className="usage-grid">
        <div><span>合計トークン</span><strong>{tokenLabel(accounting, isMock)}</strong></div>
        <div><span>呼び出し回数</span><strong>{accounting.call_count}<small> 回</small></strong></div>
        <div><span>入力</span><strong>{accounting.input_tokens?.toLocaleString("ja-JP") ?? "不明"}</strong></div>
        <div><span>出力</span><strong>{accounting.output_tokens?.toLocaleString("ja-JP") ?? "不明"}</strong></div>
      </div>
      <dl className="data-list">
        <div><dt>プロバイダー処理時間</dt><dd>{decimal(accounting.provider_latency_ms / 1000, 3)} 秒</dd></div>
        <div><dt>ターン処理時間</dt><dd>{decimal(accounting.turn_latency_ms / 1000, 3)} 秒</dd></div>
        <div><dt>失敗した呼び出し</dt><dd>{accounting.failed_calls} 回</dd></div>
        <div><dt>使用量が不明な呼び出し</dt><dd>{accounting.unknown_usage_calls} 回</dd></div>
      </dl>
      <p className="disclosure">{isMock ? "Mockのトークン数は文字数からの概算です。時間はローカル処理の測定値で、実モデルの速度・品質・料金は評価できません。" : "報告された全呼び出しを集計しています。不明な使用量を0として扱いません。単一の実行から品質や費用の優劣は判断できません。"}</p>
      {calls?.length ? <div className="call-list">{calls.map((call, index) => <div className="call-row" key={`${call.purpose}-${index}`}>
        <div><strong>{purposeNames[call.purpose] ?? call.purpose}</strong><span>{call.model}</span></div>
        <div><span className={call.status === "error" ? "text-error" : ""}>{call.status === "error" ? "失敗" : "完了"}</span><span>{decimal(call.latency_ms, 0)} ms</span></div>
      </div>)}</div> : null}
    </div>
  );
}

export default function TurnDetails({ result }: { result: TurnResult | null }) {
  const [tab, setTab] = useState<DetailTab>("focus");
  if (!result) return <aside className="inspector inspector-empty" aria-label="ターンの詳細">
    <div className="section-kicker">TURN INSPECTOR</div>
    <h2>会話の内側をのぞく</h2>
    <div className="inspection-illustration" aria-hidden="true"><span /><span /><span /><i /></div>
    <p>応答カードを選ぶと、そのターンの注目点・使われた記憶・呼び出しの内訳を確認できます。</p>
    <ol className="how-to"><li>会話例を選んで実行</li><li>左右の応答を見比べる</li><li>気になるターンを選択</li></ol>
    <div className="inspector-footnote">state はゲートの制御値です。<br />思考の深さを測る指標ではありません。</div>
  </aside>;

  const d = result.debug;
  const fragments = d.memory?.fragments?.injected ?? [];
  const context = d.memory?.context_used;
  const attention = d.memory?.attention_used;
  const stateSource = d.provider?.state_source ?? d.metrics.state_source ?? "未取得";
  const hasPrevious = Boolean(d.viewpoint?.previous_dim || d.viewpoint?.previous_focus);

  return <aside className="inspector" aria-label={`ターン ${result.turn} の詳細`} data-testid="turn-inspector">
    <div className="inspector-title">
      <div><div className="section-kicker">TURN INSPECTOR</div><h2>ターン {String(result.turn).padStart(2, "0")}</h2></div>
      <span className={`mode-badge ${result.mode}`}>{result.mode === "auto" ? "自動ゲート" : "固定 0.50"}</span>
    </div>
    <p className="inspector-question">{result.userText}</p>
    <div className="detail-tabs" role="tablist" aria-label="詳細の種類">
      {([["focus", "注目点"], ["memory", "使った記憶"], ["calls", "呼び出し"]] as const).map(([value, label]) => <button key={value} role="tab" id={`detail-tab-${value}`} aria-controls={`detail-panel-${value}`} aria-selected={tab === value} tabIndex={tab === value ? 0 : -1} onKeyDown={(event) => {
        const values: DetailTab[] = ["focus", "memory", "calls"];
        const index = values.indexOf(value);
        const next = event.key === "ArrowRight" ? values[(index + 1) % values.length] : event.key === "ArrowLeft" ? values[(index + values.length - 1) % values.length] : event.key === "Home" ? values[0] : event.key === "End" ? values[values.length - 1] : null;
        if (next) { event.preventDefault(); setTab(next); document.getElementById(`detail-tab-${next}`)?.focus(); }
      }} onClick={() => setTab(value)}>{label}</button>)}
    </div>
    <div role="tabpanel" id={`detail-panel-${tab}`} aria-labelledby={`detail-tab-${tab}`} className="detail-content">
      {tab === "focus" ? <div className="detail-stack">
        <div className="state-display">
          <div><span>適用された state</span><strong>{decimal(d.state)}</strong></div>
          <div className="state-track" role="meter" aria-label="適用されたstate" aria-valuenow={d.state} aria-valuemin={0} aria-valuemax={1}><span style={{ width: `${Math.max(0, Math.min(1, d.state)) * 100}%` }} /></div>
          <div className="scale-labels"><span>0.00</span><span>1.00</span></div>
          <p>Probeから算出した制御値 {decimal(d.observed_state)}<br /><span className="source-label">{stateSource}</span></p>
        </div>
        <div className="focus-block"><span className="field-label">DIM · 観点</span><span className="dim-pill">{d.dim ?? "未取得"}</span><span className="field-label">FOCUS · 今、注目していること</span><p>{d.focus ?? "未取得"}</p><span className="field-label">NEXT · 次に見ること</span><p>{d.next ?? "未取得"}</p></div>
        <div className={`viewpoint-box ${d.viewpoint?.changed ? "changed" : ""}`}>
          <strong>{!hasPrevious ? "最初の視点" : d.viewpoint?.changed ? "前のターンから視点が変化" : "前のターンと同じ視点"}</strong>
          {hasPrevious ? <p>{d.viewpoint?.previous_dim} · {d.viewpoint?.previous_focus}<br /><span aria-hidden="true">↓ </span>{d.dim} · {d.focus}</p> : <p>次のターンから、DIMとFOCUSの変化を比較します。</p>}
          <span>{d.pulse?.triggered ? d.viewpoint?.pulse_changed ? "探索パルスで視点が変化" : "探索パルス発動 · 選択後の視点は同じ" : result.mode === "fixed" ? "固定側の探索パルスは無効" : d.pulse?.stagnation_detected ? "停滞を検出 · パルス未発動" : "探索パルスなし"}</span>
        </div>
        <dl className="data-list">
          <div><dt>出力上限</dt><dd>{d.params.max_output_tokens} tokens</dd></div>
          <div><dt>Temperature</dt><dd>{decimal(d.params.temperature, 3)}</dd></div>
          <div><dt>会話保持の上限</dt><dd>{d.params.context_keep_msgs} messages</dd></div>
        </dl>
        <details className="technical-details"><summary>Probeと制御の詳細</summary><pre>{d.probeText ?? "Probeなし"}</pre>{d.probeText_original !== d.probeText ? <><h4>探索前のProbe</h4><pre>{d.probeText_original}</pre></> : null}{d.pulse?.candidates_text ? <><h4>探索候補 · 選択 {d.pulse.picked ?? "—"}</h4><pre>{d.pulse.candidates_text}</pre></> : null}<dl className="data-list"><div><dt>Surprisal</dt><dd>{decimal(d.metrics.surprisal, 3)}</dd></div><div><dt>Entropy</dt><dd>{decimal(d.metrics.entropy, 3)}</dd></div><div><dt>Score</dt><dd>{decimal(d.metrics.score, 3)}</dd></div></dl>{d.notes.length ? <ul>{d.notes.map((note, index) => <li key={index}>{note}</li>)}</ul> : null}</details>
      </div> : null}
      {tab === "memory" ? <div className="detail-stack">
        <p className="disclosure">このターンのMain入力に実際に渡した内容です。保存されている記憶すべてではありません。現在の入力や注目点も含みます。</p>
        <section className="memory-section"><div className="memory-heading"><h3>記憶の断片</h3><span>{fragments.length} 件使用 / 上限 {d.memory?.frag_items ?? "—"}</span></div>{fragments.length ? fragments.map((fragment) => <article className="memory-fragment" key={fragment.id}><div><span>T{fragment.turn} · {fragment.dim ?? "—"}</span><span>salience {decimal(fragment.salience)}</span></div><p>{fragment.text}</p></article>) : <p className="empty-note">このターンでは使用していません</p>}</section>
        <section className="memory-section"><div className="memory-heading"><h3>要約</h3><span>{d.summary_used?.length ?? 0} 文字使用</span></div><p className={d.summary_used ? "memory-text" : "empty-note"}>{d.summary_used || "このターンでは使用していません"}</p></section>
        <section className="memory-section"><div className="memory-heading"><h3>注目点の履歴</h3><span>{attention?.length ?? "—"} 件使用 / 上限 {d.memory?.attn_items ?? "—"}</span></div>{attention?.length ? <ol className="attention-list">{attention.map((entry, index) => <li key={`${entry.turn}-${index}`}><span>T{entry.turn} · {entry.dim}</span><p>{entry.focus}</p>{entry.next ? <small>次に: {entry.next}</small> : null}</li>)}</ol> : <p className="empty-note">{attention ? "このターンでは使用していません" : "使用した履歴の情報はありません"}</p>}</section>
        <section className="memory-section"><div className="memory-heading"><h3>送信した会話</h3><span>{context?.length ?? "—"} 件使用 / 上限 {d.memory?.ctx_keep_msgs ?? "—"}</span></div>{context?.length ? <div className="context-list">{context.map((message, index) => <details key={index}><summary>{message.role === "user" ? "あなた" : message.role === "assistant" ? "アシスタント" : message.role}<span>{message.content.slice(0, 34)}{message.content.length > 34 ? "…" : ""}</span></summary><p>{message.content}</p></details>)}</div> : <p className="empty-note">使用した会話の情報はありません</p>}</section>
      </div> : null}
      {tab === "calls" ? <div className="detail-stack"><AccountingDetails accounting={d.accounting} calls={d.provider?.calls} /><div className="log-status"><strong>{d.log?.saved ? "JSONLログを保存済み" : "JSONLログは未保存"}</strong><p>{d.log?.saved ? d.log.path : d.log?.error ?? "保存情報はありません"}</p><small>サーバー上のログです。この画面の履歴の再読込・復元には対応していません。</small></div><details className="technical-details"><summary>このターンのJSON</summary><pre>{JSON.stringify(result, null, 2)}</pre></details></div> : null}
    </div>
  </aside>;
}
