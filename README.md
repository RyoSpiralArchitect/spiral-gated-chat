# Spiral Gated Chat — Phase0.4 (Salience fragments)

Probe (<100 tokens) → state (0..1) → Main という **自己ゲート付きチャット**の最小実装です。

## できること
- 各ターンで短い **Probe** を同一モデルに生成させる
- Probe の **logprobs / top_logprobs** から surprisal / entropy を計算
- EMAベースラインとの差分から state∈[0,1] を作り、
  - max_output_tokens
  - temperature
  - context_keep_msgs（送る履歴量）
  を連動させる
- **探索パルス（ユーレカ/アナロジー）**: state/DIM が停滞したら
  - temp↑で別DIM案を3つ生成 → temp≈0の検証で1つ採用
  - 採用したプローブで stagnation を破る
- **グラデーション memory**: state に応じて連続的に
  - 送る履歴量（context_keep_msgs）
  - summary の注入量（summary_chars）
  - ATTENTION_LOG の注入量（attn_items）
  をスケール
- **Salience fragments（断片メモリ）**:
  - 短いメモ断片を複数保持し、各断片に salience（重要度）を付与
  - salience は state に応じて減衰（低stateほど速く忘れる）
  - main prompt には **最新ではなく salience上位K件**を注入（=「局所で捨てつつ、重要だけ残す」）
- **summary 更新も state 依存**:
  - 低stateほど更新間隔が長い／高stateほど頻繁に更新（追加computeを節約）
- **DIM重み付け**（まずは例どおり）
  - RISK は state を下駄履き（**state依存**: 低いほど持ち上げる）
  - META は state 上限を抑える（**段階的に緩める**: cap 0.55 → cap 0.65 → 解除）
- **探索パルス時の main 出力下限**
  - pulse で視点が切り替わった時だけ `max_output_tokens` に floor（例: 120）を入れて体感を出す
- UI 右パネルに state / probe / memory budgets / pulse を表示

## 起動
```bash
npm install
cp .env.local.example .env.local
# .env.local に OPENAI_API_KEY を入れる
npm run dev
```

- ブラウザ: http://localhost:3000
- 送信: ボタン or Ctrl/Cmd+Enter

## メモ
- Responses API で logprobs を取るには `include: ["message.output_text.logprobs"]` が必要です。
- もし logprobs が取れない場合、state は前回値を維持します（Phase0の安全策）。
