"""
設定モジュール - TOML設定ファイルの読み込みと管理
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Python 3.11+ は tomllib、それ以前は tomli を使用
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass
class ServerConfig:
    """サーバー設定"""
    host: str = "0.0.0.0"
    port: int = 18080
    log_dir: str | None = None  # ログディレクトリ（タイムスタンプ付きファイルを作成）
    verbose: bool = False
    reject_batch: bool = False  # バッチリクエストを拒否してフォールバックさせる
    # CRLF検出によるバッチ拒否（0で無効、正数で閾値）- crlf_batchとは排他
    crlf_threshold: int = 0
    # CRLF分割バッチ処理（CRLFで分割して並列推論）
    crlf_batch: bool = False
    crlf_batch_max_parallel: int = 8  # 最大並列数（ハード上限）
    crlf_batch_context_limit: int = 0  # 総コンテキスト制限（0で無制限）
    crlf_error_marker: str = "[TRANSLATION_ERROR]"  # エラー時のマーカー
    # LLM暴走検出設定
    runaway_detection: bool = True  # 暴走検出の有効化
    runaway_prefix_length: int = 20  # 鸚鵡返し検出の比較文字数
    runaway_ng_words: list[str] = field(default_factory=list)  # NGワードリスト（単純文字列マッチ）
    empty_input_as_error: bool = True  # 空入力をエラー扱いにする
    # プレースホルダー変換設定（LLMに送る前に変換、応答後に復元）
    # 形式: [[変換元, 変換先], ...]  例: [["<L_F>", "[LF]"]]
    placeholder_replacements: list[list[str]] = field(default_factory=list)


@dataclass
class LLMConfig:
    """LLM接続設定（全て必須）"""
    endpoint: str
    model: str
    max_tokens: int
    timeout: int = 300  # HTTPリクエストタイムアウト（秒）


@dataclass
class PipelineStep:
    """パイプラインの各ステップ設定"""
    name: str
    prompt: str
    enabled: bool = True
    # 入力モード: "chain" = 前ステップの出力を使用, "zero_shot" = 原文を使用
    mode: Literal["chain", "zero_shot"] = "chain"
    # ステップ固有のLLM設定（省略時はデフォルトを使用）
    endpoint: str | None = None
    model: str | None = None
    max_tokens: int | None = None


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    enabled: bool = False
    prompt_mode: Literal["replace", "extend"] = "replace"
    steps: list[PipelineStep] = field(default_factory=list)
    # 固有名詞辞書: {原文: 翻訳}
    proper_nouns: dict[str, str] = field(default_factory=dict)
    # 原文マーカー（xTranslatorからの入力から除去される）
    original_marker: str = "{ORIGINAL_TEXT}"


@dataclass
class Config:
    """全体設定"""
    server: ServerConfig
    llm: LLMConfig
    pipeline: PipelineConfig

    @classmethod
    def load(cls, path: Path | str) -> Config:
        """TOMLファイルから設定を読み込む"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """辞書から設定を構築"""
        server_data = data.get("server", {})
        llm_data = data.get("llm", {})
        pipeline_data = data.get("pipeline", {})

        # サーバー設定
        server = ServerConfig(
            host=server_data.get("host", "0.0.0.0"),
            port=server_data.get("port", 18080),
            log_dir=server_data.get("log_dir"),
            verbose=server_data.get("verbose", False),
            reject_batch=server_data.get("reject_batch", False),
            crlf_threshold=server_data.get("crlf_threshold", 0),
            crlf_batch=server_data.get("crlf_batch", False),
            crlf_batch_max_parallel=server_data.get("crlf_batch_max_parallel", 8),
            crlf_batch_context_limit=server_data.get("crlf_batch_context_limit", 0),
            crlf_error_marker=server_data.get("crlf_error_marker", "[TRANSLATION_ERROR]"),
            runaway_detection=server_data.get("runaway_detection", True),
            runaway_prefix_length=server_data.get("runaway_prefix_length", 20),
            runaway_ng_words=server_data.get("runaway_ng_words", []),
            empty_input_as_error=server_data.get("empty_input_as_error", True),
            placeholder_replacements=server_data.get("placeholder_replacements", []),
        )

        # LLM設定（必須項目のチェック）
        if "endpoint" not in llm_data:
            raise ValueError("[llm] セクションに 'endpoint' が必要です")
        if "model" not in llm_data:
            raise ValueError("[llm] セクションに 'model' が必要です")
        if "max_tokens" not in llm_data:
            raise ValueError("[llm] セクションに 'max_tokens' が必要です")

        llm = LLMConfig(
            endpoint=llm_data["endpoint"],
            model=llm_data["model"],
            max_tokens=llm_data["max_tokens"],
            timeout=llm_data.get("timeout", 300),
        )

        # パイプライン設定
        steps = []
        for step_data in pipeline_data.get("steps", []):
            steps.append(PipelineStep(
                name=step_data.get("name", "unnamed"),
                prompt=step_data.get("prompt", ""),
                enabled=step_data.get("enabled", True),
                mode=step_data.get("mode", "chain"),
                endpoint=step_data.get("endpoint"),
                model=step_data.get("model"),
                max_tokens=step_data.get("max_tokens"),
            ))

        # 固有名詞辞書
        proper_nouns = pipeline_data.get("proper_nouns", {})

        pipeline = PipelineConfig(
            enabled=pipeline_data.get("enabled", False),
            prompt_mode=pipeline_data.get("prompt_mode", "replace"),
            steps=steps,
            proper_nouns=proper_nouns,
            original_marker=pipeline_data.get("original_marker", "{ORIGINAL_TEXT}"),
        )

        return cls(server=server, llm=llm, pipeline=pipeline)

    def get_step_llm_config(self, step: PipelineStep) -> tuple[str, str, int]:
        """ステップのLLM設定を取得（ステップ固有設定がなければデフォルト使用）"""
        endpoint = step.endpoint or self.llm.endpoint
        model = step.model or self.llm.model
        max_tokens = step.max_tokens or self.llm.max_tokens
        return endpoint, model, max_tokens


def find_config_file(cli_path: str | None = None) -> Path | None:
    """設定ファイルを検索する

    優先順位:
    1. CLIで指定されたパス
    2. カレントディレクトリの config.toml
    3. スクリプトと同じディレクトリの config.toml
    """
    if cli_path:
        path = Path(cli_path)
        if path.exists():
            return path
        return None

    # カレントディレクトリ
    cwd_config = Path.cwd() / "config.toml"
    if cwd_config.exists():
        return cwd_config

    # スクリプトと同じディレクトリ
    script_dir = Path(__file__).parent
    script_config = script_dir / "config.toml"
    if script_config.exists():
        return script_config

    return None
