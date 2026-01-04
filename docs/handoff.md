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
| `Pipeline` | パイプライン実行エンジン |
| `Pipeline.format_proper_nouns()` | 固有名詞辞書を文字列にフォーマット |
| `Pipeline.build_prompt()` | プレースホルダー置換 (`{input}`, `{original}`, `{proper_nouns}`) |
| `Pipeline.execute_step()` | 単一ステップ実行（HTTP POST to LLM） |
| `Pipeline.execute()` | パイプライン全体実行 |
| `extract_user_content()` | messages から user role のコンテンツ抽出 |

#### translator_proxy.py

| 関数 | 説明 |
|------|------|
| `setup_logging()` | ログ設定（ファイル + コンソール） |
| `parse_json()` | JSON パース（エラー時は None、エラーログ出力） |
| `fix_completion_request()` | リクエスト補正（model, max_tokens, stream, CRLF正規化） |
| `sanitize_response()` | レスポンスから `__verbose` 等を除去 |
| `build_response_from_content()` | パイプライン結果から OpenAI 互換レスポンス構築 |
| `passthrough()` | メインルーティングハンドラ |
| `main()` | CLI引数処理、設定読み込み、サーバー起動 |

## 設定パラメータ一覧

### [server]

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `host` | str | "0.0.0.0" | リッスンホスト |
| `port` | int | 18080 | リッスンポート |
| `log_file` | str\|null | null | ログファイルパス |
| `verbose` | bool | false | 詳細ログ有効化 |
| `reject_batch` | bool | false | バッチリクエスト拒否 |
| `crlf_threshold` | int | 0 | CRLF検出閾値（0で無効） |

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

### xTranslator「仮想配列が破損」エラー

**原因**: LLM が XML 形式ではなくプレーンテキストで応答するため、xTranslator のバッチ処理が失敗する

**対策**: `reject_batch = true` を設定し、最初からフォールバックモード（単一リクエスト）で動作させる

### 空クエリ問題

**原因**: xTranslator が空文字列を送信することがある

**対策**: `original_marker = "{ORIGINAL_TEXT}"` を設定し、xTranslator 側でクエリ先頭にマーカーを付加。プロキシ側で自動除去する

## 開発・デバッグ

### 起動方法

```bash
# 設定ファイル自動検出
uv run translator-proxy

# 設定ファイル指定
uv run translator-proxy --config config.toml

# 詳細ログ有効化
uv run translator-proxy --verbose --log proxy.log
```

### ログ確認

```bash
# リアルタイム監視
tail -f proxy.log

# パイプラインステップ実行確認
grep "Pipeline step" proxy.log

# エラー確認
grep -i error proxy.log
```

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
