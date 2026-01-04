- [x] 1. xTranslatorのクエリはCRLFで改行されたものを一気に推論させる方式である。マルチバッチ推論可能なLLM環境がある場合、CRLFで改行されたものを複数のクエリに分割してバッチ推論を行い、最後にまとめる方法を取ることが可能と思われる。これを検討し、実装する。この際、xTranslatorへのレスポンスは元のCRLFを維持し、原文を置き換えたものである必要がある。
  - 実装完了: `crlf_batch = true` で有効化
  - 並列HTTPリクエスト方式（asyncio.gather + Semaphore）
  - パイプライン機能との統合済み
  - エラー時はマーカー（`[TRANSLATION_ERROR]`）を返す
  - 並列数の動的計算: `min(max_parallel, context_limit / 推定トークン数)`
    - `crlf_batch_max_parallel`: 最大並列数（ハード上限）
    - `crlf_batch_context_limit`: 総コンテキスト制限（llama.cppの-cに合わせる）
- [x] 2. 1で不要になったCRLF系の実装を整理
  - `crlf_threshold` は `crlf_batch = false` 時のみ有効（従来の拒否動作）
  - 両方の機能を共存可能に設計
- [x] 3. CRLF分割バッチ処理のバグ修正（2026-01-04）
  - **問題1: 並列実行時に極端に遅くなる（数分かかる）**
    - 原因: xTranslatorからの入力形式が `{ORIGINAL_TEXT}\r\n実際のテキスト\r\n...` であり、CRLF分割後に最初のテキストがマーカーのみになっていた。マーカー除去後は空文字列となり、LLMがプロンプト内のXML例全体を再生成しようとしていた
    - 対策: CRLF分割の**前に**マーカーを除去するように修正
  - **問題2: xTranslator「仮想配列が破損」エラー**
    - 原因: マーカー除去後に先頭のCRLFが残り、レスポンスの先頭に空行（`\r\n`）が入っていた。これによりxTranslatorが期待する行数と一致しなくなった
    - 対策: マーカー除去後に先頭のCRLFも除去するように修正
  - **注意点**: xTranslatorの入力形式は `{ORIGINAL_TEXT}\r\n行1\r\n行2\r\n...` であり、マーカーは最初のCRLFの前に付加される
- [x] 4. LLM暴走検出機能の実装（2026-01-04）
  - **目的**: LLMがプロンプトを鸚鵡返しする「暴走」を検出し、早期に打ち切る
  - **実装内容**:
    - `runaway_detection`: 暴走検出の有効/無効（デフォルト: true）
    - `runaway_prefix_length`: 比較文字数（デフォルト: 20）
    - `empty_input_as_error`: 空入力をエラー扱い（デフォルト: true）
  - **動作原理**:
    - ストリーミングモードでLLMの応答をチャンクごとに受信
    - プロンプトの先頭N文字と応答を比較
    - 一致検出時に即座にストリームを打ち切り、翻訳エラーマーカーを返す
  - **対象ファイル**: config.py, pipeline.py, translator_proxy.py, config.example.toml
- [x] 5. NGワードによる暴走検出機能の追加（2026-01-04）
  - **目的**: XMLタグ等を無限に生成し続ける暴走パターンを検出
  - **実装内容**:
    - `runaway_ng_words`: NGワードリスト（単純文字列マッチ、XMLタグ等のメタ文字もそのまま指定可能）
  - **動作原理**:
    - **LLMの応答（assistantロール）のみ**をチェック対象とする
    - ストリーミング受信中にNGワードの出現をチェック
    - チャンク境界を跨ぐNGワードも検出可能（末尾最適化）
    - 検出時に即座にストリームを打ち切り、翻訳エラーマーカーを返す
  - **重要な仕様**:
    - プロンプト（userロール）に翻訳例としてXMLが含まれていても、LLMの応答にXML宣言が出力されれば暴走として検出
    - 期待される応答は翻訳テキストのみであり、XML宣言やタグヘッダーは暴走のサイン
  - **設定例**:
    ```toml
    runaway_ng_words = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        "<EDID>",
    ]
    ```
  - **対象ファイル**: config.py, pipeline.py, translator_proxy.py, config.example.toml
- [x] 6. プレースホルダー変換機能の実装（2026-01-04）
  - **目的**: XMLタグ風のマーカー（`<L_F>` 等）がLLMにXMLとして誤認されるのを防ぐ
  - **背景**: 入力テキストに `<L_F>` のような改行マーカーが含まれていると、LLMがXMLとして解釈し、XML形式で応答する暴走が発生する
  - **実装内容**:
    - `placeholder_replacements`: 変換ルールリスト（`[[変換元, 変換先], ...]` 形式）
  - **動作原理**:
    - LLMに送信する前に変換元を変換先に置換（例: `<L_F>` → `[LF]`）
    - LLMの応答を受け取った後、変換先を変換元に復元（例: `[LF]` → `<L_F>`）
  - **設定例**:
    ```toml
    placeholder_replacements = [
        ["<L_F>", "[LF]"],
        ["<BR>", "[BR]"],
    ]
    ```
  - **対象ファイル**: config.py, translator_proxy.py, config.example.toml