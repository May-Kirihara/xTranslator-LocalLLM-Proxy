# Translator Proxy 引き継ぎ資料

## プロジェクト概要

xTranslator（SKYRIM MOD翻訳ツール）と llama.cpp サーバー間のプロキシサーバー。
xTranslator からの OpenAI API 互換リクエストを llama.cpp サーバーに中継し、リクエスト補正・レスポンスサニタイズを行う。

## ファイル構成

```
translatorproxy/
├── translator_proxy.py   # FastAPI メインサーバー (390行)
├── config.py             # TOML設定モジュール (169行)
├── pipeline.py           # マルチステップパイプライン処理 (215行)
├── config.example.toml   # サンプル設定ファイル
├── config.toml           # 実際の設定ファイル（ユーザー作成）
├── pyproject.toml        # uv/pip プロジェクト設定
├── README.md             # ユーザー向けドキュメント
└── handoff.md            # この引き継ぎ資料
```

## アーキテクチャ

### データフロー

```
xTranslator
    |
    | HTTP POST /v1/chat/completions
    v
[Translator Proxy (FastAPI/uvicorn)]
    |
    +-- reject_batch=true の場合 --> 400エラー返却 --> フォールバック強制
    |
    +-- crlf_batch=true の場合（CRLF分割バッチ処理）
    |       |
    |       +---> "{ORIGINAL_TEXT}\r\nDragon\r\nSword\r\nShield"
    |       |         |
    |       |         v (1. マーカー除去)
    |       +---> "\r\nDragon\r\nSword\r\nShield"
    |       |         |
    |       |         v (2. 先頭CRLF除去)
    |       +---> "Dragon\r\nSword\r\nShield"
    |       |         |
    |       |         v (3. CRLFで分割)
    |       +---> ["Dragon", "Sword", "Shield"]
    |       |         |
    |       |         v (4. 並列HTTP推論、各テキストにパイプライン適用)
    |       +---> ["ドラゴン", "剣", "盾"]
    |       |         |
    |       |         v (5. CRLFで結合)
    |       +---> "ドラゴン\r\n剣\r\n盾"
    |                 |
    |                 v
    |             Response
    |
    +-- pipeline.enabled=true の場合
    |       |
    |       +---> Step 1: translate (mode=zero_shot)
    |       |         |
    |       |         v
    |       +---> Step 2: review (mode=chain)
    |                 |
    |                 v
    |             Response
    |
    +-- pipeline.enabled=false の場合 --> パススルー
            |
            v
        llama.cpp サーバー
```

### 主要クラス/関数

#### config.py

| クラス | 説明 |
|--------|------|
| `ServerConfig` | サーバー設定（host, port, log_file, verbose, reject_batch, crlf_threshold） |
| `LLMConfig` | LLM接続設定（endpoint, model, max_tokens, timeout） |
| `PipelineStep` | パイプラインステップ設定（name, prompt, mode, enabled） |
| `PipelineConfig` | パイプライン全体設定（enabled, prompt_mode, steps, proper_nouns, original_marker） |
| `Config` | 全設定を統合。`load()`, `from_dict()` メソッド |
| `find_config_file()` | 設定ファイルの自動検索 |

#### pipeline.py

| クラス/関数 | 説明 |
|-------------|------|
| `RunawayDetectedError` | LLM暴走（鸚鵡返し）検出時の例外 |
| `Pipeline` | パイプライン実行エンジン |
| `Pipeline.format_proper_nouns()` | 固有名詞辞書を文字列にフォーマット |
| `Pipeline.build_prompt()` | プレースホルダー置換 (`{input}`, `{original}`, `{proper_nouns}`) |
| `Pipeline._check_runaway()` | 鸚鵡返しをチェック（プロンプト先頭N文字と応答を比較） |
| `Pipeline.execute_step()` | 単一ステップ実行（ストリーミング対応・暴走検出付き） |
| `Pipeline.execute()` | パイプライン全体実行 |
| `extract_user_content()` | messages から user role のコンテンツ抽出 |

#### translator_proxy.py

| 関数 | 説明 |
|------|------|
| `apply_placeholder_replacements()` | プレースホルダーを変換（LLMに送る前） |
| `restore_placeholder_replacements()` | プレースホルダーを復元（LLM応答後） |
| `setup_logging()` | ログ設定（ファイル + コンソール） |
| `parse_json()` | JSON パース（エラー時は None、エラーログ出力） |
| `fix_completion_request()` | リクエスト補正（model, max_tokens, stream, CRLF正規化） |
| `sanitize_response()` | レスポンスから `__verbose` 等を除去 |
| `build_response_from_content()` | パイプライン結果から OpenAI 互換レスポンス構築 |
| `estimate_tokens()` | テキストのトークン数を概算 |
| `calculate_parallel_limit()` | コンテキスト制限に基づいて並列数を計算 |
| `process_crlf_batch()` | CRLF分割バッチ処理のメイン関数 |
| `process_single_text_with_pipeline()` | 単一テキストをパイプライン処理（空入力・暴走検出付き） |
| `_check_runaway_passthrough()` | パススルー用の鸚鵡返しチェック |
| `process_single_text_passthrough()` | 単一テキストをパススルー処理（ストリーミング・暴走検出付き） |
| `passthrough()` | メインルーティングハンドラ |
| `main()` | CLI引数処理、設定読み込み、サーバー起動 |

## 設定パラメータ一覧

### [server]

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `host` | str | "0.0.0.0" | リッスンホスト |
| `port` | int | 18080 | リッスンポート |
| `log_dir` | str\|null | null | ログディレクトリ（起動ごとにタイムスタンプ付きファイルを作成） |
| `verbose` | bool | false | 詳細ログ有効化 |
| `reject_batch` | bool | false | バッチリクエスト拒否 |
| `crlf_batch` | bool | false | CRLF分割バッチ処理を有効化 |
| `crlf_batch_max_parallel` | int | 8 | CRLF分割時の最大並列数（ハード上限） |
| `crlf_batch_context_limit` | int | 0 | 総コンテキスト制限（0で無制限）。並列数はmin(max_parallel, context_limit/推定トークン数)で決定 |
| `crlf_error_marker` | str | "[TRANSLATION_ERROR]" | 推論失敗時のマーカー |
| `runaway_detection` | bool | true | LLM暴走検出の有効化（ストリーミングモードを使用） |
| `runaway_prefix_length` | int | 20 | 鸚鵡返し検出の比較文字数 |
| `runaway_ng_words` | list[str] | [] | NGワードリスト（単純文字列マッチ、XMLタグ等もそのまま指定可能） |
| `empty_input_as_error` | bool | true | 空入力をエラー扱いにする |
| `placeholder_replacements` | list[list[str]] | [] | プレースホルダー変換ルール（LLM送信前に変換、応答後に復元） |
| `crlf_threshold` | int | 0 | CRLF検出閾値（crlf_batch=false時のみ有効） |

### [llm]

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `endpoint` | str | (設定必須) | llama.cpp エンドポイント |
| `model` | str | (設定必須) | モデル名 |
| `max_tokens` | int | (設定必須) | 最大トークン数 |
| `timeout` | int | 300 | HTTPタイムアウト（秒） |

### [pipeline]

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `enabled` | bool | false | パイプライン有効化 |
| `prompt_mode` | str | "replace" | プロンプトモード ("replace"\|"extend") |
| `original_marker` | str | "{ORIGINAL_TEXT}" | 原文マーカー（除去対象） |
| `proper_nouns` | dict | {} | 固有名詞辞書 |

### [[pipeline.steps]]

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `name` | str | (必須) | ステップ名 |
| `prompt` | str | (必須) | プロンプトテンプレート |
| `enabled` | bool | true | ステップ有効化 |
| `mode` | str | "chain" | 入力モード ("chain"\|"zero_shot") |
| `endpoint` | str\|null | null | ステップ固有エンドポイント |
| `model` | str\|null | null | ステップ固有モデル |
| `max_tokens` | int\|null | null | ステップ固有トークン数 |

## プロンプトプレースホルダー

| プレースホルダー | 説明 |
|-----------------|------|
| `{input}` | 現在の入力（mode=chain: 前ステップ出力, mode=zero_shot: 原文） |
| `{original}` | 原文（常に最初の入力テキスト） |
| `{proper_nouns}` | 固有名詞リスト（"- 英語 → 日本語" 形式） |

## 既知の問題と対策

### xTranslator「仮想配列が破損」エラー（XMLバッチ処理）

**原因**: LLM が XML 形式ではなくプレーンテキストで応答するため、xTranslator のバッチ処理が失敗する

**対策**: `reject_batch = true` を設定し、最初からフォールバックモード（単一リクエスト）で動作させる

### xTranslator「仮想配列が破損」エラー（CRLF分割バッチ処理）

**原因**: レスポンスの先頭に余分な空行（`\r\n`）が入ると、xTranslator が期待する行数と一致しなくなる

**対策**: `process_crlf_batch()` 内でマーカー除去後に先頭のCRLFも除去している。この処理を削除・変更しないこと

### 空クエリ問題

**原因**: xTranslator が空文字列を送信することがある

**対策**: `original_marker = "{ORIGINAL_TEXT}"` を設定し、xTranslator 側でクエリ先頭にマーカーを付加。プロキシ側で自動除去する

### LLM暴走（鸚鵡返し・XMLタグ生成）問題

**原因**: LLMが入力プロンプトをそのまま出力し始めたり、XMLタグを無限に生成し続ける「暴走」が発生することがある

**対策**:
1. **鸚鵡返し検出**: `runaway_detection = true` を設定（デフォルト有効）。ストリーミングモードでLLMの応答を受信しながら、プロンプトの先頭N文字（`runaway_prefix_length`、デフォルト20文字）と比較。一致を検出した時点でストリームを打ち切り、翻訳エラー扱いにする

2. **NGワード検出**: `runaway_ng_words` にNGワードリストを設定。XMLタグ等のメタ文字もそのまま指定可能（単純文字列マッチ）。チャンク境界を跨ぐNGワードも検出可能（末尾最適化あり）
   ```toml
   runaway_ng_words = [
       '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
       "<EDID>",
   ]
   ```

**重要**: NGワード検出は**LLMの応答（assistantロール）のみ**をチェック対象とする。プロンプト（userロール）に翻訳例としてXMLが含まれていても、LLMの応答にXML宣言が出力されれば暴走として検出される。期待される応答は翻訳テキストのみであり、XML宣言やタグヘッダーは暴走のサインである。

### XMLタグ風マーカーの誤認問題

**原因**: 入力テキストに `<L_F>` のようなXMLタグ風の改行マーカーが含まれていると、LLMがXML形式の入力と誤認し、XML形式で応答しようとして暴走する

**対策**: `placeholder_replacements` を設定し、XMLタグ風マーカーを別の形式に変換してからLLMに送信する。応答後に元の形式に復元される
```toml
placeholder_replacements = [
    ["<L_F>", "[LF]"],      # 改行マーカー
    ["<BR>", "[BR]"],       # 改行タグ
]
```

この設定により、`<L_F>` は `[LF]` に変換されてLLMに送信され、応答の `[LF]` は `<L_F>` に復元される。

### 空入力の翻訳

**原因**: 空文字列や空白・改行のみの入力をLLMに送ると、意味のない応答や暴走を引き起こす

**対策**: `empty_input_as_error = true` を設定（デフォルト有効）。空入力を検出した時点で即座にエラーマーカーを返す

### CRLF分割バッチ処理が極端に遅い

**原因**: xTranslator の入力形式は `{ORIGINAL_TEXT}\r\n行1\r\n行2\r\n...` であり、マーカーは最初のCRLFの**前**に付加される。CRLF分割を先に行うと、最初のテキストがマーカーのみになり、マーカー除去後は空文字列として処理される。空文字列を翻訳しようとすると、LLM がプロンプト内の例を再生成しようとして数分かかる

**対策**: `process_crlf_batch()` 内でCRLF分割の**前に**マーカーを除去している。この順序を変更しないこと

## 開発・デバッグ

### 起動方法

```bash
# 設定ファイル自動検出
uv run translator-proxy

# 設定ファイル指定
uv run translator-proxy --config config.toml

# 詳細ログ有効化（logs/ ディレクトリにタイムスタンプ付きファイルを作成）
uv run translator-proxy --verbose --log-dir logs
```

ログファイルは起動ごとに `logs/proxy_YYYYMMDD_HHMMSS.log` の形式で作成されます。

### ログ確認

```bash
# リアルタイム監視
tail -f proxy.log

# パイプラインステップ実行確認
grep "Pipeline step" proxy.log

# エラー確認
grep -i error proxy.log

# CRLF分割バッチ処理のデバッグ
grep "CRLF" proxy.log

# リクエスト/レスポンス全文確認（デバッグログが有効な場合）
grep -A 10 "CRLF REQUEST FULL" proxy.log
grep -A 10 "CRLF RESPONSE FULL" proxy.log
```

### CRLF分割バッチ処理のトラブルシューティング

1. **処理が極端に遅い（数分かかる）場合**
   - ログで `CRLF batch item X: sending text (0 chars)` を確認
   - 空文字列が処理されている場合、マーカー除去のロジックに問題がある

2. **xTranslator で「仮想配列が破損」エラーが出る場合**
   - ログで `CRLF RESPONSE FULL` を確認
   - レスポンスの先頭に空行がないか確認
   - 入力と出力の行数が一致しているか確認

3. **一部の行が翻訳されない場合**
   - LLM が翻訳せずに原文をそのまま返している可能性
   - プロンプトの調整が必要

### テスト方法

```bash
# 単一リクエストテスト
curl -X POST http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role": "user", "content": "Hello world"}]
  }'
```

## 将来の拡張ポイント

1. **話者別プロンプト** (`[speakers]`): config.example.toml にコメントアウト状態で定義済み
2. **並列ステップ実行**: 現在は順次実行のみ
3. **キャッシュ機能**: 同一入力の結果をキャッシュ
4. **ストリーミング対応**: 現在は `stream: false` 固定

## 依存関係

```toml
[project]
dependencies = [
    "httpx>=0.24",
    "fastapi>=0.100",
    "uvicorn>=0.22",
    "tomli>=2.0; python_version < '3.11'",
]
```

Python 3.11 以降は標準ライブラリの `tomllib` を使用。

## 連絡事項

- このプロキシは xTranslator の API 互換性問題を解決するために開発された
- llama.cpp サーバーは別プロセスで起動済みの前提
- パイプラインを有効にする場合は `pipeline.enabled = true` を明示的に設定すること
