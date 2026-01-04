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

    async def execute_step(
        self,
        client: httpx.AsyncClient,
        step: PipelineStep,
        input_text: str,
        original_text: str,
        original_messages: list[dict] | None = None,
    ) -> str:
        """単一ステップを実行

        Args:
            client: HTTPクライアント
            step: 実行するステップ
            input_text: 入力テキスト（前ステップの出力）
            original_text: 原文
            original_messages: xTranslatorからの元メッセージ

        Returns:
            LLMからの応答テキスト
        """
        endpoint, model, max_tokens = self.config.get_step_llm_config(step)

        prompt = self.build_prompt(step, input_text, original_text, original_messages)
        messages = self.build_messages(prompt)

        request_data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Accept-Encoding": "identity",
            "Connection": "close",
        }

        logger.info(f"  Pipeline step '{step.name}': sending to {endpoint}")
        logger.debug(f"  Prompt: {prompt[:200]}...")

        url = f"{endpoint}/v1/chat/completions"
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
