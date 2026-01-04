# xTranslator-LocalLLM-Proxy

ローカルLLM(llama.cpp)による翻訳の為のxTranslator (SKYRIM MOD翻訳ツール) と llama.cpp サーバー間のプロキシサーバー。

## 概要

xTranslator が OpenAI API 互換エンドポイントにリクエストを送信する際、不正なパラメータや形式の問題が発生し、llama.cpp サーバーでパースできないことがあります。このプロキシはリクエストを補正し、レスポンスをサニタイズすることで互換性を確保します。

## 機能

### リクエスト補正
- モデル名を llama.cpp で使用するモデルに自動置換
- `max_tokens` のデフォルト値を 4096 に設定（出力が途切れるのを防止）
- `stream: false` を設定
<!-- - CRLF (`\r\n`) を LF (`\n`) に正規化 -->

### レスポンスサニタイズ
- llama.cpp 固有の `__verbose` フィールドを除去
- `_` で始まる不要なフィールドを除去
- OpenAI API 互換形式に整形

### バッチリクエスト対応
- 配列形式のリクエストを個別に処理
- 各リクエストを順次 llama.cpp に送信
- レスポンスを配列としてまとめて返却
- `reject_batch = true` でバッチリクエストを拒否し、xTranslator のフォールバックモード（単一リクエスト）を強制
- `crlf_threshold` でCRLF数によるバッチ検出・拒否が可能

### CRLF分割バッチ処理 (新機能)
xTranslator は CRLF (`\r\n`) 区切りで複数のテキストを1リクエストで送信します。この機能を有効にすると：

- CRLF で分割して個別のテキストとして並列処理
- llama.cpp の並列スロット (-np) を活用した高速化
- 結果を CRLF で結合して返却（元の形式を維持）
- コンテキスト制限に基づく並列数の自動調整
- エラー時はマーカー（`[TRANSLATION_ERROR]`）で失敗箇所を明示

### ログ機能
- リクエスト/レスポンスの詳細をファイルに記録
- デバッグ用の詳細ログ

### マルチステップパイプライン (新機能)
- 複数のプロンプトを順次適用（翻訳→校正→ポストプロセス）
- 各ステップで異なるLLMエンドポイント/モデルを使用可能
- xTranslatorのプロンプトを置換または拡張可能
- TOML設定ファイルで柔軟に設定

### LLM暴走検出 (新機能)
- LLMがプロンプトを鸚鵡返しする「暴走」をリアルタイム検出
- NGワードによるXMLタグ等の無限生成検出
- ストリーミングモードで応答を受信しながら、プロンプト先頭との比較を実行
- 暴走検出時は即座にストリームを打ち切り、翻訳エラー扱い
- 空入力（空白・改行のみ）の自動検出・エラー処理

## インストール

```bash
cd translatorproxy
uv sync
```

## 使用方法

### 設定ファイルを使用（推奨）

```bash
# サンプル設定をコピー
cp config.example.toml config.toml

# 設定ファイルを編集
vim config.toml

# 起動（config.toml を自動検出）
uv run translator-proxy
```

### CLIオプションで起動

```bash
# デフォルト設定で起動
uv run translator-proxy

# llama.cpp サーバーを指定
uv run translator-proxy --llama http://0.0.0.0:8080

# 設定ファイルを明示的に指定
uv run translator-proxy --config /path/to/config.toml

# ポートを指定
uv run translator-proxy --port 8081

# モデル名を指定
uv run translator-proxy --model my-model.gguf
```

### オプション一覧

```
-c, --config FILE    TOML設定ファイルパス (default: config.toml があれば使用)
-l, --llama URL      llama.cpp サーバーURL (設定ファイルより優先)
-p, --port PORT      プロキシサーバーのポート (設定ファイルより優先)
-H, --host HOST      プロキシサーバーのホスト (設定ファイルより優先)
-m, --model NAME     使用するモデル名 (設定ファイルより優先)
-t, --max-tokens N   デフォルトの max_tokens (設定ファイルより優先)
-L, --log FILE       ログファイルパス (設定ファイルより優先)
-v, --verbose        詳細ログを有効化 (設定ファイルより優先)
```

### 例

```bash
# ログファイルを指定して詳細ログを有効化
uv run translator-proxy --log proxy.log --verbose

# 設定ファイル + CLIオプションで上書き
uv run translator-proxy --config config.toml --port 9000
```

## 設定ファイル

### 基本構造

```toml
# サーバー設定
[server]
host = "0.0.0.0"
port = 18080
log_file = "proxy.log"
verbose = false
reject_batch = true      # バッチリクエストを拒否してフォールバック強制

# CRLF分割バッチ処理
crlf_batch = true        # CRLF分割バッチ処理を有効化
crlf_batch_max_parallel = 4      # 最大並列数（llama.cppの-npに合わせる）
crlf_batch_context_limit = 65536 # 総コンテキスト制限（llama.cppの-cに合わせる）
crlf_error_marker = "[TRANSLATION_ERROR]"  # エラー時のマーカー
crlf_threshold = 10      # CRLF検出閾値（crlf_batch=false時のみ有効）

# LLM暴走検出
runaway_detection = true    # 暴走検出の有効化（ストリーミングモードを使用）
runaway_prefix_length = 20  # 鸚鵡返し検出の比較文字数
empty_input_as_error = true # 空入力をエラー扱いにする
runaway_ng_words = ['<?xml version="1.0"', "<EDID>"]  # NGワードリスト

# LLM接続設定（デフォルト）
[llm]
endpoint = "http://0.0.0.0:8080"
model = "model-name.gguf"
max_tokens = 4096
timeout = 300            # HTTPタイムアウト（秒）

# パイプライン設定
[pipeline]
enabled = false          # true でマルチステップ処理を有効化
prompt_mode = "replace"  # "replace" または "extend"
original_marker = "{ORIGINAL_TEXT}"  # xTranslatorからの入力先頭から除去
```

### パイプラインステップ

```toml
[pipeline]
enabled = true
prompt_mode = "replace"

# ステップ1: 翻訳
[[pipeline.steps]]
name = "translate"
enabled = true
mode = "zero_shot"  # 原文を使用（最初のステップ向け）
prompt = """
以下の英語テキストを日本語に翻訳してください。
{input}
"""

# ステップ2: 校正
[[pipeline.steps]]
name = "review"
enabled = true
mode = "chain"  # 前ステップの出力を使用（デフォルト）
# ステップ固有のLLM設定（省略時はデフォルト使用）
endpoint = "http://another-server:8080"
model = "review-model.gguf"
prompt = """
以下の翻訳文を校正してください。
{input}
"""

# ステップ3: ポストプロセス
[[pipeline.steps]]
name = "postprocess"
enabled = false  # 無効化されたステップはスキップ
mode = "chain"
prompt = """
固有名詞の置き換えを行ってください。
{input}
"""
```

### プロンプトモード

- `replace`: xTranslatorからのプロンプトを無視し、ステップのプロンプトのみ使用
- `extend`: xTranslatorからのプロンプトにステップのプロンプトを追加

### ステップの入力モード

各ステップで `mode` パラメータを設定することで、そのステップへの入力を制御できます：

| モード | 説明 | 用途 |
|--------|------|------|
| `chain` | 前ステップの出力を `{input}` として使用（デフォルト） | 校正、ポストプロセスなど |
| `zero_shot` | 原文を `{input}` として使用 | 最初の翻訳ステップなど |

### 原文マーカー

xTranslator の空クエリ対策として、`original_marker` を設定できます。xTranslator 側で翻訳クエリの先頭に `{ORIGINAL_TEXT}` などのマーカーを付加し、プロキシ側で自動除去します：

```toml
[pipeline]
original_marker = "{ORIGINAL_TEXT}"  # 先頭から除去される
```

### CRLF分割バッチ処理

xTranslator は CRLF 区切りで複数のテキストを1リクエストで送信します。`crlf_batch = true` にすると、これを分割して並列処理します：

```toml
[server]
crlf_batch = true                # CRLF分割バッチ処理を有効化
crlf_batch_max_parallel = 4      # 最大並列数（llama.cppの-npスロット数に合わせる）
crlf_batch_context_limit = 65536 # 総コンテキスト制限（llama.cppの-cに合わせる、0で無制限）
crlf_error_marker = "[TRANSLATION_ERROR]"  # 推論失敗時のマーカー
```

**並列数の自動計算:**
- 並列数 = `min(max_parallel, context_limit / 推定トークン数)`
- プロンプトのトークン数を推定し、コンテキスト制限を超えないように自動調整
- llama.cpp の `-c`（コンテキストサイズ）と `-np`（並列スロット数）に合わせて設定

**処理フロー:**
```
入力: "Dragon\r\nSword\r\nShield"
    ↓ CRLF分割
["Dragon", "Sword", "Shield"]
    ↓ 並列推論（パイプライン適用）
["ドラゴン", "剣", "盾"]
    ↓ CRLF結合
出力: "ドラゴン\r\n剣\r\n盾"
```

### CRLF検出によるバッチ拒否

`crlf_batch = false` の場合、CRLF数による拒否が可能です：

```toml
[server]
crlf_batch = false
crlf_threshold = 10  # メッセージ内のCRLFがこの数以上なら拒否（0で無効）
```

これにより、xTranslator は自動的にフォールバックモード（単一文字列ごとの翻訳）に切り替わります。

### LLM暴走検出

LLMがプロンプトを鸚鵡返ししたり、XMLタグを無限に生成し続ける「暴走」を検出し、早期に打ち切ります：

```toml
[server]
runaway_detection = true    # 暴走検出の有効化（ストリーミングモードを使用）
runaway_prefix_length = 20  # 鸚鵡返し検出の比較文字数
empty_input_as_error = true # 空入力をエラー扱いにする

# NGワードリスト（XMLタグ等のメタ文字もそのまま指定可能）
runaway_ng_words = [
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
    "<EDID>",
]
```

**動作原理:**
- `runaway_detection = true` にすると、LLMへのリクエストがストリーミングモードになる
- **LLMの応答（assistantロール）のみ**をチェック対象とする
- レスポンスをチャンクごとに受信しながら、以下をチェック：
  1. プロンプトの先頭N文字と比較（鸚鵡返し検出）
  2. NGワードリストとの照合（XMLタグ等の無限生成検出）
- 検出した時点でストリームを即座に打ち切り、翻訳エラーマーカーを返す
- 空入力（空白・改行のみ）も事前にチェックし、エラー扱いにする
- チャンク境界を跨ぐNGワードも検出可能（末尾最適化あり）

**NGワードの選定:**
- プロンプトに翻訳例としてXMLが含まれていても、LLMの応答にXML宣言が出力されれば暴走として検出
- 期待される応答は翻訳テキストのみであり、XML宣言やタグヘッダーは暴走のサイン

### HTTPタイムアウト

LLMサーバーへのリクエストタイムアウトを設定できます：

```toml
[llm]
timeout = 300  # 秒（デフォルト: 300）
```

長い翻訳やコンテキスト長が大きい場合は、この値を増やしてください。

### プロンプト内のプレースホルダー

| プレースホルダー | 説明 |
|-----------------|------|
| `{input}` | 現在の入力テキスト（前ステップの出力、最初は原文） |
| `{original}` | 原文（最初の入力テキスト、常に同じ） |
| `{proper_nouns}` | 固有名詞リスト（辞書を文字列化したもの） |

### 固有名詞辞書

TOMLで辞書形式で定義し、プロンプト内で `{proper_nouns}` として参照できます：

```toml
[pipeline.proper_nouns]
"Dragonborn" = "ドラゴンボーン"
"Whiterun" = "ホワイトラン"
"Jarl" = "首長"
```

プロンプト内での展開例：

```
- Dragonborn → ドラゴンボーン
- Whiterun → ホワイトラン
- Jarl → 首長
```

## xTranslator の設定

xTranslator の API エンドポイント設定で、このプロキシのアドレスを指定:

```
http://<プロキシのIP>:18080/v1/chat/completions
```

## ログ確認

```bash
# リアルタイム監視
tail -f proxy.log

# 最新のログを確認
tail -100 proxy.log

# パイプラインステップの実行を確認
grep "Pipeline step" proxy.log
```

## 既知の制限

### xTranslator の「仮想配列が破損」エラー

xTranslator がバッチで複数の翻訳文字列を送信した際、LLM が期待する XML 形式で応答しない場合にこのエラーが発生します。

**原因:** LLM が XML 形式ではなく単純なテキストで翻訳結果を返すため、xTranslator が内部配列を構築できない

**動作:** xTranslator は自動的にフォールバックモードに切り替わり、1件ずつ翻訳を実行します。最終的には翻訳は完了しますが、バッチ処理より時間がかかります。

**対処法:**
- `reject_batch = true` を設定し、最初からフォールバックモードで動作させる（推奨）
- 現状維持（エラー後に自動でフォールバック）
- LLM のプロンプトを調整して XML 形式で応答するようにする

## アーキテクチャ

### パススルーモード（pipeline.enabled = false）

```
xTranslator
    |
    v
[Proxy Server (port 18080)]
    - リクエスト補正
    - レスポンスサニタイズ
    - ログ記録
    |
    v
llama.cpp Server (port 8080)
```

### パイプラインモード（pipeline.enabled = true）

```
xTranslator
    |
    v
[Proxy Server (port 18080)]
    |
    +---> Step 1: translate (LLM Server A)
    |         |
    |         v
    +---> Step 2: review (LLM Server B)
    |         |
    |         v
    +---> Step 3: postprocess (LLM Server A)
              |
              v
          Response
```

## 開発

```bash
# 依存関係のインストール
uv sync

# 開発モードで実行
uv run python translator_proxy.py --verbose
```

## ファイル構成

```
xTranslator-LocalLLM-Proxy/
├── translator_proxy.py   # メインサーバー
├── config.py             # 設定モジュール
├── pipeline.py           # パイプライン処理
├── config.example.toml   # サンプル設定
├── config.toml           # 実際の設定ファイル（ユーザー作成）
├── pyproject.toml        # プロジェクト設定
├── README.md             # このファイル
├── docs/
│   ├── handoff.md        # 開発者向け引き継ぎ資料
│   └── ToDo.md           # タスク管理
└── test/
    └── test_crlf_batch.py  # CRLF分割バッチ処理のテスト
```
