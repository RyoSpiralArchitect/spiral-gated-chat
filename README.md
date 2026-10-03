# Spiral Gated Chat — Phase 0.4 revival

短い Probe → state（0–1）→ Main という自己ゲート付きチャットを、同じ台本の **Auto / Fixed 比較**で観察するためのローカル実験です。会話、注目点の変化、記憶の注入、各ターンの呼び出し量・使用トークンを並べて確認できます。

**state はこのアプリの制御値です。モデル内部の思考量、理解度、回答品質を測った値ではありません。** Mock は決定的な動作確認用で、実モデルの品質や節約効果を示しません。

## 起動

`package.json` はリポジトリ直下ではなく、次のサブディレクトリにあります。Node.js 20 以降を使用してください。

```bash
cd spiral-gated-chat-phase0.4
npm ci
SPIRAL_CHAT_PROVIDER=mock npm run dev
```

http://localhost:3000 を開きます。Mock なら API キー、外部モデル呼び出し、課金は不要です。画面には利用中の provider / model と Mock の区別が表示されます。

### 実モデルを使う場合

同じサブディレクトリの `.env.local` にキーと provider を設定し、開発サーバーを再起動します。キーをコミットしないでください。

```bash
# OpenAI（既定 provider）
SPIRAL_CHAT_PROVIDER=openai OPENAI_API_KEY=... npm run dev

# Anthropic / Claude
SPIRAL_CHAT_PROVIDER=anthropic ANTHROPIC_API_KEY=... \
ANTHROPIC_MODEL=claude-sonnet-4-6 npm run dev
```

- OpenAI は Probe 先頭の DIM 行の token logprobs / top_logprobs から state を求めます。`include: ["message.output_text.logprobs"]` を指定します
- OpenAI で logprobs が得られない場合は、前回値を代用したことを `previous_state` として明示します
- Anthropic は native Messages API を使います。logprobs は得られないため、Probe fields 由来の `heuristic_probe_fields` と明示します。`CLAUDE_API_KEY` / `ANTHROPIC_AUTH_TOKEN` もキーの fallback として利用できます
- OpenAI SDK の自動再試行は無効です。失敗した試行を隠して、完全な token 集計に見せないためです
- 台本の実行は両モード分の API 呼び出しを行います。実 provider では費用が発生し得ます

## 小さな比較を試す

画面の台本を選び、比較を開始します。同じ user text を **Fixed → Auto** の順に各ターンへ渡し、それぞれ独立した新しい session を作ります。比較をもう一度実行しても、前の会話履歴は流用しません。

1. **気軽な会話**（`easy-chat`、3ターン）
   - 近所で気分転換 → 静かな場所で散歩と読書 → 出発前の準備
   - 注目点、返答、利用された記憶を読み比べます
2. **大切な条件が加わる**（`important-constraint`、4ターン）
   - 読書イベント → 12人・カフェ → 避難通路と車いすの通れる配置 → 条件を守る準備
   - 重要な制約への注目と、後のターンに実際に渡された履歴・断片を確認します
3. **同じ視点で足踏み**（`repeated-perspective`、10ターン）
   - 企画の方向性が決まらない相談を続けます
   - Auto の停滞検出、探索候補、検証で選んだ視点、実際に視点が変わったかを確認します。Mock は探索と視点変更を再現できる台本です。実モデルの探索発動は保証しません

完全な入力台本は [`lib/scenarios.ts`](spiral-gated-chat-phase0.4/lib/scenarios.ts) にあります。比較を止めると完了済みの結果は残りますが、通信中だった session を曖昧なまま再開しません。新しい比較を始めてください。個別の会話モードでも、各ターンの詳細を確認できます。

### Fixed の比較条件

Fixed は「Probe なし」や「記憶なし」のベースラインではありません。

- 同じ provider / model、同じ Probe、同じ Main の基本 prompt と記憶ルールを使用します
- 生成・記憶の制御 state を **0.50** に固定します。Probe 由来の `observed_state` は別に表示・保存します
- 固定値は Main 出力上限 290 tokens、temperature 0.375、履歴上限 10 messages、summary 上限 172 characters、attention 上限 2 entries、fragment 上限 4 entries です
- summary 更新間隔は12ターン、更新出力上限は53 tokens。初期の更新ターンが −1 のため最初は11ターン目に更新し、その後12ターン間隔です
- 探索パルスだけを無効にします。記憶の追加、減衰、重要度による選択、summary 更新は残します
- 上限が同じでも、利用可能な履歴・summary・断片の数や中身はターンごとに違います

Auto は従来のゲート計算を使い、出力上限、temperature、履歴量、summary・attention・fragment の予算を state に連動させます。停滞した場合は Explore → Verify を追加し、採用された視点で Main を生成します。会話が進むと両側の返答と履歴は分岐します。これは少数の会話を観察する実験であり、統計的な品質評価や速度ベンチマークではありません。

## 何を数えているか

各ターンの accounting は **Probe / Main / Explore / Verify / Summary の全呼び出し**を集計します。失敗した呼び出しも call count / failed calls / latency に含めます。

- `input_tokens` / `output_tokens` / `total_tokens`: provider が返した使用量。全体が不明なら `null` とし、0で補いません
- `known_total_tokens`: 使用量が分かる呼び出しだけの小計。完全な合計と区別してください
- `unknown_usage_calls`: token 合計の分からない呼び出し数
- `usage_kind`: 実 provider の報告は `reported`、Mock の文字数ベース概算は `mock_estimate`、一部不明は `partial`、全件不明は `unknown`
- `provider_latency_ms`: 各 provider 呼び出しの wall time の合計。失敗も含みます
- `turn_latency_ms`: サーバーで受信してから全 provider 処理を終えるまで。JSONL 保存時間とブラウザとの通信時間は含みません

Mock の tokens は文字数からの概算で、実際の tokenizer 使用量や課金額ではありません。実 provider の使用量も請求書の検算や金額換算を行っていません。不明値のある比較から節約率を推定しないでください。

各ターンには DIM / FOCUS と直前ターンからの変化、探索による変化を別々に残します。記憶欄は「今ある記憶」だけでなく、その Main に渡した `context_used`、`attention_used`、`summary_used`、注入 fragment の snapshot を表示します。後のターンが変わっても過去の snapshot はその時点の内容のままです。

## ログと session

- 会話と debug payload は `logs/sessions/<sessionId>.jsonl` に保存します
- `logs/session-index.jsonl` に session の provider / model / mode と保存先を記録します
- `SPIRAL_CHAT_LOG_DIR=/path/to/logs` で保存先を変更できます
- ターンの schema は `spiral-gated-chat.turn.v2`。`mode`、`comparison_id`、`scenario_id`、`status`、全 calls と accounting を含みます
- Main など必須処理の失敗は `status: "error"`、`assistantText: ""` で記録します。失敗で会話の turn / gate / memory を進めず、再試行は同じ turn 番号を使います。ログの各行を成功ターンとして数えず、`status` も確認してください
- Explore / Summary の失敗は任意処理の失敗として記録し、成功した Main の返答は残します。token 合計は不完全になります
- 同じ session の同時実行や途中での mode 変更は409で拒否します。`expectedTurn` が現在の次ターンと違う要求も、provider 呼び出し前に409で拒否します
- 空入力、型違い、不正ID、12,000文字を超える入力などは400で拒否します

session は最大100件まで保持し、上限に達すると最も古く使われた待機中の session を削除します。30分以上使われていない session も、次の取得・解放時にメモリから削除します。JSONL ログは残します。実行中のターンは削除せず、全枠が実行中なら新しい session の要求を503で返します。削除後の会話は復元せず、画面の「新しい会話」や新しい比較で始め直してください。

**session はサーバープロセス内のメモリにのみ保持します。** サーバーの再起動や開発時の reload で失われ、JSONL からの復元機能はありません。画面の履歴も永続的な保存・復元機能ではありません。複数 server process 間の session 共有や永続ロックもありません。ローカル実験用途です。

ログには入力、返答、記憶の断片が平文で入ります。秘密情報を入力しないでください。ログ保存に失敗した場合は debug に失敗を表示します。

## 検証

アプリのサブディレクトリで実行します。

```bash
npm run typecheck
npm run test:unit
npm run test:e2e
npm run build
# unit + e2e をまとめて実行
npm test
```

- Unit: 全5フェーズの集計、失敗・不明・部分 usage、明示的な0、Mock 概算、異常値。加えて session の件数上限・期限・LRU・実行中の保護を検証
- E2E: Mock のみで独自の Next dev server を起動。3種類の故障設定を順番に検証します。live API への切替はありません
- E2E: config / validation、10ターンの停滞と視点変更、実際のUI台本、Fixed の一定設定、summary 更新、独立 session と再実行、過去 snapshot、並行要求、重複要求、失敗・再試行の rollback、JSONL v2 を確認します
- `E2E_PORT=3101 npm run test:e2e` でポートを変更できます。`KEEP_E2E_LOGS=1` なら一時ログを残し、最後に保存先を出力します
- 同じ checkout の `.next` を共有するため、build / dev / E2E は同時に実行しないでください

GitHub Actions は全 PR と main への push で、Node.js 22 の単一 Ubuntu runner 上で install → typecheck → unit → Mock API E2E → build を順に実行します。API キーや外部モデル呼び出しは不要です。

Mock 専用のテストフックとして `SPIRAL_MOCK_DELAY_MS`（0–1000 ms）と `SPIRAL_MOCK_FAIL_PURPOSE`（`main` / `summary` / `explore` など）があります。後者は対象リクエストに `[mock:fail]` が含まれる場合だけ故障させます。実 provider には適用しません。
