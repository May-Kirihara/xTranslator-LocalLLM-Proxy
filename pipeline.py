"""
パイプライン処理モジュール - マルチステップ翻訳処理
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from config import Config, PipelineStep

logger = logging.getLogger("translator-proxy")


class RunawayDetectedError(Exception):
    """LLM暴走（鸚鵡返し）検出時の例外"""

    def __init__(self, message: str, partial_content: str = ""):
        super().__init__(message)
        self.partial_content = partial_content


class Pipeline:
    """マルチステップ翻訳パイプライン"""

    def __init__(self, config: Config):
        self.config = config

    def is_enabled(self) -> bool:
        """パイプラインが有効かどうか"""
        return self.config.pipeline.enabled

    def get_active_steps(self) -> list[PipelineStep]:
        """有効なステップのリストを返す"""
        return [s for s in self.config.pipeline.steps if s.enabled]

    def format_proper_nouns(self) -> str:
        """固有名詞辞書を文字列にフォーマット"""
        proper_nouns = self.config.pipeline.proper_nouns
        if not proper_nouns:
            return ""

        lines = []
        for original, translated in proper_nouns.items():
            lines.append(f"- {original} → {translated}")
        return "\n".join(lines)

    def build_prompt(
        self,
        step: PipelineStep,
        input_text: str,
        original_text: str,
        original_messages: list[dict] | None = None,
    ) -> str:
        """ステップ用のプロンプトを構築する

        Args:
            step: パイプラインステップ
            input_text: 入力テキスト（前ステップの出力または原文）
            original_text: 原文（最初の入力テキスト）
            original_messages: xTranslatorからの元メッセージ（extendモード用）

        プレースホルダー:
            {input} - 現在の入力（前ステップの出力）
            {original} - 原文
            {proper_nouns} - 固有名詞リスト
        """
        prompt_mode = self.config.pipeline.prompt_mode

        # プレースホルダーを置換
        def replace_placeholders(text: str) -> str:
            text = text.replace("{input}", input_text)
            text = text.replace("{original}", original_text)
            text = text.replace("{proper_nouns}", self.format_proper_nouns())
            return text

        if prompt_mode == "extend" and original_messages:
            # extendモード: 元のプロンプトにステップのプロンプトを追加
            original_content = ""
            for msg in original_messages:
                if msg.get("role") == "user":
                    original_content = msg.get("content", "")
                    break

            # 元のプロンプトとステップのプロンプトを結合
            combined = f"{original_content}\n\n---\n\n{step.prompt}"
            return replace_placeholders(combined)
        else:
            # replaceモード: ステップのプロンプトのみ使用
            return replace_placeholders(step.prompt)

    def build_messages(self, prompt: str) -> list[dict]:
        """Chat Completion用のmessages配列を構築"""
        return [{"role": "user", "content": prompt}]

    def _check_runaway(self, prompt: str, content: str, prev_content_len: int = 0) -> tuple[bool, str]:
        """鸚鵡返し・NGワードをチェック

        Args:
            prompt: LLMに送信したプロンプト
            content: LLMからの応答（途中または全体）
            prev_content_len: 前回チェック時のcontent長（NGワード検出の最適化用）

        Returns:
            (検出フラグ, 検出理由)
        """
        if not self.config.server.runaway_detection:
            return False, ""

        # 1. 鸚鵡返しチェック（プロンプト先頭との比較）
        prefix_len = self.config.server.runaway_prefix_length
        if prefix_len > 0:
            prompt_prefix = prompt[:prefix_len]
            content_prefix = content[:prefix_len]

            if len(content_prefix) >= prefix_len and content_prefix == prompt_prefix:
                return True, "prompt echo"

        # 2. NGワードチェック（末尾最適化）
        # LLMの応答にNGワードが含まれていたら暴走として検出
        ng_words = self.config.server.runaway_ng_words
        if ng_words:
            # NGワードの最大長を取得
            max_ng_len = max(len(w) for w in ng_words)

            # チェック対象範囲を計算（新しく追加された部分 + NGワード最大長のバッファ）
            # これにより、チャンク境界を跨ぐNGワードも検出可能
            if prev_content_len > max_ng_len:
                check_start = prev_content_len - max_ng_len
            else:
                check_start = 0
            check_content = content[check_start:]

            for ng_word in ng_words:
                if ng_word in check_content:
                    return True, f"NG word: {ng_word}"

        return False, ""

    async def execute_step(
        self,
        client: httpx.AsyncClient,
        step: PipelineStep,
        input_text: str,
        original_text: str,
        original_messages: list[dict] | None = None,
    ) -> str:
        """単一ステップを実行（ストリーミング対応・暴走検出付き）

        Args:
            client: HTTPクライアント
            step: 実行するステップ
            input_text: 入力テキスト（前ステップの出力）
            original_text: 原文
            original_messages: xTranslatorからの元メッセージ

        Returns:
            LLMからの応答テキスト

        Raises:
            RunawayDetectedError: 鸚鵡返しを検出した場合
        """
        endpoint, model, max_tokens = self.config.get_step_llm_config(step)

        prompt = self.build_prompt(step, input_text, original_text, original_messages)
        messages = self.build_messages(prompt)

        # 暴走検出が有効な場合はストリーミングを使用
        use_streaming = self.config.server.runaway_detection

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": use_streaming,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json" if not use_streaming else "text/event-stream",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

        logger.info(f"  Pipeline step '{step.name}': sending to {endpoint} (prompt={len(prompt)} chars, max_tokens={max_tokens}, streaming={use_streaming})")
        logger.debug(f"  Prompt: {prompt[:200]}...")

        url = f"{endpoint}/v1/chat/completions"

        if use_streaming:
            # ストリーミングモード: チャンクごとに暴走検出
            content = ""
            prev_content_len = 0

            async with client.stream("POST", url, headers=headers, json=request_data) as response:
                if response.status_code != 200:
                    logger.error(f"  Step '{step.name}' failed: status={response.status_code}")
                    raise RuntimeError(f"Step '{step.name}' failed with status {response.status_code}")

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    # SSE形式: "data: {...}" or "data: [DONE]"
                    if line.startswith("data: "):
                        data_str = line[6:]  # "data: " を除去
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                chunk_content = delta.get("content", "")
                                if chunk_content:
                                    content += chunk_content

                                    # 暴走検出: 鸚鵡返し・NGワードチェック
                                    detected, reason = self._check_runaway(prompt, content, prev_content_len)
                                    if detected:
                                        logger.warning(f"  Step '{step.name}': RUNAWAY DETECTED - {reason}")
                                        raise RunawayDetectedError(
                                            f"LLM runaway detected in step '{step.name}': {reason}",
                                            partial_content=content,
                                        )
                                    prev_content_len = len(content)
                        except json.JSONDecodeError:
                            continue

            logger.info(f"  Step '{step.name}' completed: {len(content)} chars")
            return content

        else:
            # 非ストリーミングモード（従来の処理）
            response = await client.post(url, headers=headers, json=request_data)

            if response.status_code != 200:
                logger.error(f"  Step '{step.name}' failed: status={response.status_code}")
                raise RuntimeError(f"Step '{step.name}' failed with status {response.status_code}")

            result = response.json()

            # 応答テキストを抽出
            choices = result.get("choices", [])
            if not choices:
                raise RuntimeError(f"Step '{step.name}' returned no choices")

            content = choices[0].get("message", {}).get("content", "")
            logger.info(f"  Step '{step.name}' completed: {len(content)} chars")

            return content

    async def execute(
        self,
        client: httpx.AsyncClient,
        original_messages: list[dict],
    ) -> str:
        """パイプライン全体を実行

        Args:
            client: HTTPクライアント
            original_messages: xTranslatorからの元メッセージ

        Returns:
            最終的な応答テキスト
        """
        active_steps = self.get_active_steps()

        if not active_steps:
            logger.warning("No active pipeline steps configured")
            return ""

        # 最初の入力はxTranslatorからのメッセージ内容
        original_text = ""
        for msg in original_messages:
            if msg.get("role") == "user":
                original_text = msg.get("content", "")
                break

        # 原文マーカーを除去
        marker = self.config.pipeline.original_marker
        if marker and original_text.startswith(marker):
            original_text = original_text[len(marker):]
            logger.debug(f"Removed original marker: {marker}")

        logger.info(f"Starting pipeline with {len(active_steps)} steps")

        # 各ステップを順次実行
        current_text = original_text
        for i, step in enumerate(active_steps):
            # 入力モードに応じて入力テキストを決定
            if step.mode == "zero_shot":
                input_text = original_text
                logger.info(f"Executing step {i + 1}/{len(active_steps)}: {step.name} (zero_shot)")
            else:  # chain
                input_text = current_text
                logger.info(f"Executing step {i + 1}/{len(active_steps)}: {step.name} (chain)")

            current_text = await self.execute_step(
                client,
                step,
                input_text,
                original_text,  # 原文は常に渡す
                original_messages if i == 0 else None,  # 最初のステップのみ元メッセージを渡す
            )

        return current_text


def extract_user_content(messages: list[dict]) -> str:
    """messagesからユーザーコンテンツを抽出"""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
