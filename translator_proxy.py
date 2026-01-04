#!/usr/bin/env python3
"""
xTranslator to llama.cpp proxy server

Usage:
    uv run translator-proxy --config config.toml
    uv run translator-proxy --llama http://0.0.0.0:8080 --port 18080
    uv run translator-proxy --help
"""
import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request, Response
import httpx
import uvicorn

from config import Config, find_config_file
from pipeline import Pipeline, RunawayDetectedError

# グローバル設定（起動時に初期化）
config: Config | None = None
pipeline: Pipeline | None = None

app = FastAPI(title="Translator Proxy", description="xTranslator to llama.cpp proxy")


def apply_placeholder_replacements(text: str) -> str:
    """プレースホルダーを変換（LLMに送る前）

    例: <L_F> → [LF]
    """
    assert config is not None, "Config not initialized"
    for replacement in config.server.placeholder_replacements:
        if len(replacement) >= 2:
            text = text.replace(replacement[0], replacement[1])
    return text


def restore_placeholder_replacements(text: str) -> str:
    """プレースホルダーを復元（LLM応答後）

    例: [LF] → <L_F>
    """
    assert config is not None, "Config not initialized"
    for replacement in config.server.placeholder_replacements:
        if len(replacement) >= 2:
            text = text.replace(replacement[1], replacement[0])
    return text


logger = logging.getLogger("translator-proxy")


def setup_logging(log_dir: str | None, verbose: bool) -> str | None:
    """ログ設定

    Args:
        log_dir: ログディレクトリ（Noneの場合はログファイルを作成しない）
        verbose: 詳細ログの有効化

    Returns:
        作成されたログファイルのパス（ログファイルを作成しない場合はNone）
    """
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]
    log_file_path = None

    if log_dir:
        # ログディレクトリを作成（存在しない場合）
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # タイムスタンプ付きのログファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_path / f"proxy_{timestamp}.log"

        handlers.append(logging.FileHandler(log_file_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )

    return str(log_file_path) if log_file_path else None


def parse_json(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as e:
        logger.warning(f"JSON decode error (UTF-8): {e}")
        return None
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e.msg} at line {e.lineno}, col {e.colno}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error parsing JSON: {e}")
        return None


def fix_completion_request(item: dict):
    """個々のリクエストを補正する"""
    assert config is not None, "Config not initialized"
    item["model"] = config.llm.model
    item.setdefault("max_tokens", config.llm.max_tokens)
    item.setdefault("stream", False)

    # CRLF 正規化 <- 不要と思われるので一旦コメントアウト
    # msgs = item.get("messages")
    # if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
    #     c = msgs[0].get("content")
    #     if isinstance(c, str):
    #         msgs[0]["content"] = c.replace("\r\n", "\n")


def sanitize_response(response_data: dict) -> dict:
    """llama.cppのレスポンスをOpenAI互換形式にサニタイズする"""
    if not isinstance(response_data, dict):
        return response_data

    # __verbose フィールドを除去
    if "__verbose" in response_data:
        del response_data["__verbose"]

    # choices内の不要フィールドを除去
    choices = response_data.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                for key in list(choice.keys()):
                    if key.startswith("_"):
                        del choice[key]

    return response_data


def build_response_from_content(content: str, model: str) -> dict:
    """パイプライン結果からOpenAI互換レスポンスを構築"""
    return {
        "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def send_single_request(
    client: httpx.AsyncClient, path: str, item: dict, headers: dict, index: int
):
    """単一リクエストを送信し、結果を返す"""
    assert config is not None, "Config not initialized"
    try:
        logger.info(f"  Sending batch item {index}...")
        r = await client.request(
            "POST",
            f"{config.llm.endpoint}/{path}",
            headers=headers,
            json=item,
        )
        result = parse_json(r.content)
        if result:
            result = sanitize_response(result)
        logger.info(f"  Batch item {index} completed: status={r.status_code}")
        return (index, r.status_code, result if result else r.content.decode("utf-8", errors="replace"))
    except Exception as e:
        logger.error(f"  Batch item {index} failed: {e}")
        return (index, 500, {"error": str(e)})


async def process_with_pipeline(
    client: httpx.AsyncClient, messages: list[dict]
) -> dict:
    """パイプラインを使用してリクエストを処理"""
    assert config is not None, "Config not initialized"
    assert pipeline is not None, "Pipeline not initialized"

    try:
        result_content = await pipeline.execute(client, messages)
        return build_response_from_content(result_content, config.llm.model)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return {"error": str(e)}


async def process_single_text_with_pipeline(
    client: httpx.AsyncClient,
    text: str,
    semaphore: asyncio.Semaphore,
    index: int,
) -> tuple[int, str, bool]:
    """単一テキストをパイプライン処理（CRLF分割バッチ用）

    Returns:
        (インデックス, 結果テキスト, 成功フラグ)
    """
    assert config is not None, "Config not initialized"
    assert pipeline is not None, "Pipeline not initialized"

    async with semaphore:
        # 空入力チェック
        if config.server.empty_input_as_error and not text.strip():
            logger.warning(f"  CRLF batch item {index}: empty input detected")
            return (index, config.server.crlf_error_marker, False)

        try:
            # プレースホルダー変換（LLMに送る前）
            converted_text = apply_placeholder_replacements(text)
            if converted_text != text:
                logger.debug(f"  CRLF batch item {index}: placeholder converted ({len(text)} -> {len(converted_text)} chars)")
                logger.debug(f"    Before: {text[:100]}...")
                logger.debug(f"    After:  {converted_text[:100]}...")
            messages = [{"role": "user", "content": converted_text}]
            result = await pipeline.execute(client, messages)
            # プレースホルダー復元（LLM応答後）
            result = restore_placeholder_replacements(result)
            logger.debug(f"  CRLF batch item {index}: success ({len(result)} chars)")
            return (index, result, True)
        except RunawayDetectedError as e:
            logger.warning(f"  CRLF batch item {index}: runaway detected - {e}")
            return (index, config.server.crlf_error_marker, False)
        except Exception as e:
            logger.error(f"  CRLF batch item {index} failed: {e}")
            return (index, config.server.crlf_error_marker, False)


def _check_runaway_passthrough(prompt: str, content: str, prev_content_len: int = 0) -> tuple[bool, str]:
    """パススルー用の鸚鵡返し・NGワードチェック

    Args:
        prompt: LLMに送信したプロンプト（テキスト）
        content: LLMからの応答（途中または全体）
        prev_content_len: 前回チェック時のcontent長（NGワード検出の最適化用）

    Returns:
        (検出フラグ, 検出理由)
    """
    assert config is not None, "Config not initialized"

    if not config.server.runaway_detection:
        return False, ""

    # 1. 鸚鵡返しチェック（プロンプト先頭との比較）
    prefix_len = config.server.runaway_prefix_length
    if prefix_len > 0:
        prompt_prefix = prompt[:prefix_len]
        content_prefix = content[:prefix_len]

        if len(content_prefix) >= prefix_len and content_prefix == prompt_prefix:
            return True, "prompt echo"

    # 2. NGワードチェック（末尾最適化）
    # LLMの応答にNGワードが含まれていたら暴走として検出
    ng_words = config.server.runaway_ng_words
    if ng_words:
        max_ng_len = max(len(w) for w in ng_words)

        if prev_content_len > max_ng_len:
            check_start = prev_content_len - max_ng_len
        else:
            check_start = 0
        check_content = content[check_start:]

        for ng_word in ng_words:
            if ng_word in check_content:
                return True, f"NG word: {ng_word}"

    return False, ""


async def process_single_text_passthrough(
    client: httpx.AsyncClient,
    text: str,
    semaphore: asyncio.Semaphore,
    index: int,
    headers: dict,
) -> tuple[int, str, bool]:
    """単一テキストをパススルー処理（CRLF分割バッチ用、パイプライン無効時）
    ストリーミング対応・暴走検出付き

    Returns:
        (インデックス, 結果テキスト, 成功フラグ)
    """
    assert config is not None, "Config not initialized"

    async with semaphore:
        # 空入力チェック
        if config.server.empty_input_as_error and not text.strip():
            logger.warning(f"  CRLF batch item {index}: empty input detected")
            return (index, config.server.crlf_error_marker, False)

        try:
            # プレースホルダー変換（LLMに送る前）
            converted_text = apply_placeholder_replacements(text)

            # 暴走検出が有効な場合はストリーミングを使用
            use_streaming = config.server.runaway_detection

            request_data = {
                "model": config.llm.model,
                "messages": [{"role": "user", "content": converted_text}],
                "max_tokens": config.llm.max_tokens,
                "stream": use_streaming,
            }

            url = f"{config.llm.endpoint}/v1/chat/completions"

            if use_streaming:
                # ストリーミングモード: チャンクごとに暴走検出
                content = ""
                prev_content_len = 0
                stream_headers = {**headers, "Accept": "text/event-stream"}

                async with client.stream("POST", url, headers=stream_headers, json=request_data) as response:
                    if response.status_code != 200:
                        logger.error(f"  CRLF batch item {index} failed: status={response.status_code}")
                        return (index, config.server.crlf_error_marker, False)

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # SSE形式: "data: {...}" or "data: [DONE]"
                        if line.startswith("data: "):
                            data_str = line[6:]
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

                                        # 暴走検出: 鸚鵡返し・NGワードチェック（変換後テキストで比較）
                                        detected, reason = _check_runaway_passthrough(converted_text, content, prev_content_len)
                                        if detected:
                                            logger.warning(f"  CRLF batch item {index}: RUNAWAY DETECTED - {reason}")
                                            return (index, config.server.crlf_error_marker, False)
                                        prev_content_len = len(content)
                            except json.JSONDecodeError:
                                continue

                # プレースホルダー復元（LLM応答後）
                content = restore_placeholder_replacements(content)
                logger.debug(f"  CRLF batch item {index}: success ({len(content)} chars)")
                return (index, content, True)

            else:
                # 非ストリーミングモード（従来の処理）
                r = await client.post(url, headers=headers, json=request_data)

                if r.status_code != 200:
                    logger.error(f"  CRLF batch item {index} failed: status={r.status_code}")
                    return (index, config.server.crlf_error_marker, False)

                result = r.json()
                choices = result.get("choices", [])
                if not choices:
                    return (index, config.server.crlf_error_marker, False)

                content = choices[0].get("message", {}).get("content", "")
                # プレースホルダー復元（LLM応答後）
                content = restore_placeholder_replacements(content)
                logger.debug(f"  CRLF batch item {index}: success ({len(content)} chars)")
                return (index, content, True)

        except Exception as e:
            logger.error(f"  CRLF batch item {index} failed: {e}")
            return (index, config.server.crlf_error_marker, False)


def estimate_tokens(text: str) -> int:
    """テキストのトークン数を概算（日本語は1文字≒1トークン、英語は4文字≒1トークン）"""
    # 簡易推定: 日本語が多い場合は文字数に近く、英語が多い場合は1/4程度
    # 安全のため、文字数の1/2を使用（過大評価気味）
    return max(1, len(text) // 2)


def calculate_parallel_limit(sample_text: str, use_pipeline: bool) -> int:
    """コンテキスト制限に基づいて並列数を計算

    Args:
        sample_text: サンプルテキスト（最初の分割テキスト）
        use_pipeline: パイプラインを使用するかどうか

    Returns:
        許容される並列数
    """
    assert config is not None, "Config not initialized"

    max_parallel = config.server.crlf_batch_max_parallel
    context_limit = config.server.crlf_batch_context_limit

    if context_limit <= 0:
        return max_parallel

    # プロンプトのトークン数を推定
    if use_pipeline and pipeline is not None:
        active_steps = pipeline.get_active_steps()
        if active_steps:
            # 最初のステップのプロンプトでトークン数を推定
            step = active_steps[0]
            prompt = pipeline.build_prompt(step, sample_text, sample_text)
            estimated_tokens = estimate_tokens(prompt)
        else:
            estimated_tokens = estimate_tokens(sample_text)
    else:
        # パススルー時は入力テキスト + オーバーヘッドを推定
        estimated_tokens = estimate_tokens(sample_text) + 50

    if estimated_tokens <= 0:
        return max_parallel

    # コンテキスト制限から並列可能数を計算
    context_based_parallel = context_limit // estimated_tokens

    # max_parallelと比較して小さい方を採用（最低1）
    result = max(1, min(max_parallel, context_based_parallel))

    logger.info(f"  Parallel limit: {result} (estimated_tokens={estimated_tokens}, context_limit={context_limit}, max_parallel={max_parallel})")

    return result


async def process_crlf_batch(
    client: httpx.AsyncClient,
    content: str,
    use_pipeline: bool,
    headers: dict,
) -> str:
    """CRLF区切りのテキストを分割して並列処理し、結合して返す

    Args:
        client: HTTPクライアント
        content: CRLF区切りのテキスト
        use_pipeline: パイプラインを使用するかどうか
        headers: HTTPヘッダー

    Returns:
        CRLF区切りの結果テキスト
    """
    assert config is not None, "Config not initialized"

    # 原文マーカーを先に除去
    marker = config.pipeline.original_marker
    if marker and content.startswith(marker):
        content = content[len(marker):]
        logger.debug(f"Removed original marker from content: {marker}")

    # マーカー直後のCRLFも除去（先頭の空行を防ぐ）
    while content.startswith("\r\n"):
        content = content[2:]
        logger.debug("Removed leading CRLF")

    # CRLFで分割
    texts = content.split("\r\n")

    # 非空テキストから並列数を計算
    non_empty_texts = [t for t in texts if t.strip()]
    if non_empty_texts:
        parallel_limit = calculate_parallel_limit(non_empty_texts[0], use_pipeline)
    else:
        parallel_limit = config.server.crlf_batch_max_parallel

    logger.info(f"CRLF batch processing: {len(texts)} items (parallel_limit={parallel_limit})")

    # 空文字列の処理（そのまま保持）
    non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
    results = [""] * len(texts)  # 結果配列を初期化

    if not non_empty_indices:
        logger.warning("All CRLF batch items are empty")
        return content

    # セマフォで並列数を制限
    semaphore = asyncio.Semaphore(parallel_limit)

    # 並列タスクを作成
    tasks = []
    for i in non_empty_indices:
        if use_pipeline:
            task = process_single_text_with_pipeline(client, texts[i], semaphore, i)
        else:
            task = process_single_text_passthrough(client, texts[i], semaphore, i, headers)
        tasks.append(task)

    # 並列実行
    task_results = await asyncio.gather(*tasks)

    # 結果を元の順序で配置
    success_count = 0
    for idx, result_text, success in task_results:
        results[idx] = result_text
        if success:
            success_count += 1

    logger.info(f"CRLF batch completed: {success_count}/{len(non_empty_indices)} succeeded")

    # CRLFで結合して返す
    return "\r\n".join(results)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def passthrough(path: str, req: Request):
    assert config is not None, "Config not initialized"

    raw = await req.body()
    data = parse_json(raw) if raw else None

    # ログ
    logger.info(f"=== REQUEST [{datetime.now().isoformat()}] ===")
    logger.info(f"Path: {path}")
    logger.info(f"Data type: {type(data).__name__}")
    if isinstance(data, list):
        logger.info(f"Batch size: {len(data)}")
    logger.debug(f"Raw request: {raw[:2000].decode('utf-8', errors='replace') if raw else 'None'}...")

    # ヘッダを新規に組み立てる
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }

    # バッチリクエスト拒否（フォールバックさせる）
    if (
        config.server.reject_batch
        and path.endswith("v1/chat/completions")
        and isinstance(data, list)
    ):
        logger.info(f"Rejecting batch request ({len(data)} items) - forcing fallback")
        return Response(
            content=json.dumps({"error": "Batch requests are not supported"}, ensure_ascii=False),
            status_code=400,
            headers={"Content-Type": "application/json"},
        )

    # CRLF分割バッチ処理（crlf_batch=trueの場合）
    if (
        config.server.crlf_batch
        and path.endswith("v1/chat/completions")
        and isinstance(data, dict)
    ):
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "\r\n" in content:
                    logger.info(f"CRLF batch mode: detected CRLF in content ({len(content)} chars)")
                    logger.info(f"=== CRLF REQUEST FULL ===")
                    logger.info(f"{content}")
                    logger.info(f"=== END CRLF REQUEST ===")
                    use_pipeline = pipeline is not None and pipeline.is_enabled()

                    async with httpx.AsyncClient(timeout=config.llm.timeout) as client:
                        result_content = await process_crlf_batch(
                            client, content, use_pipeline, headers
                        )

                    logger.info(f"=== CRLF RESPONSE FULL ===")
                    logger.info(f"{result_content}")
                    logger.info(f"=== END CRLF RESPONSE ===")

                    response_data = build_response_from_content(result_content, config.llm.model)
                    response_json = json.dumps(response_data, ensure_ascii=False)
                    logger.info(f"=== FINAL JSON RESPONSE ===")
                    logger.info(f"{response_json}")
                    logger.info(f"=== END FINAL JSON RESPONSE ===")
                    return Response(
                        content=response_json,
                        status_code=200,
                        headers={"Content-Type": "application/json"},
                    )
                break

    # CRLF検出によるバッチ拒否（XMLバッチ処理の検出）- crlf_batch無効時のみ
    if (
        not config.server.crlf_batch
        and config.server.crlf_threshold > 0
        and path.endswith("v1/chat/completions")
        and isinstance(data, dict)
    ):
        messages = data.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                crlf_count = content.count("\r\n")
                if crlf_count >= config.server.crlf_threshold:
                    logger.info(f"Rejecting request with {crlf_count} CRLFs (threshold: {config.server.crlf_threshold}) - likely batch XML")
                    return Response(
                        content=json.dumps({"error": "Request appears to contain batch data"}, ensure_ascii=False),
                        status_code=200,
                        headers={"Content-Type": "application/json"},
                    )
                break

    # パイプラインが有効な場合
    if (
        path.endswith("v1/chat/completions")
        and pipeline is not None
        and pipeline.is_enabled()
    ):
        async with httpx.AsyncClient(timeout=config.llm.timeout) as client:
            # バッチ処理
            if isinstance(data, list):
                logger.info(f"Processing batch request with pipeline ({len(data)} items)...")
                responses = []
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        messages = item.get("messages", [])
                        result = await process_with_pipeline(client, messages)
                        responses.append(result)
                    else:
                        responses.append({"error": "Invalid item in batch"})

                return Response(
                    content=json.dumps(responses, ensure_ascii=False),
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                )

            # 単一リクエスト
            if isinstance(data, dict):
                messages = data.get("messages", [])
                result = await process_with_pipeline(client, messages)
                return Response(
                    content=json.dumps(result, ensure_ascii=False),
                    status_code=200,
                    headers={"Content-Type": "application/json"},
                )

    # パイプライン無効時は従来のパススルー処理
    # バッチ処理
    if path.endswith("v1/chat/completions") and isinstance(data, list):
        logger.info(f"Processing batch request with {len(data)} items...")

        for item in data:
            if isinstance(item, dict):
                fix_completion_request(item)

        responses = []
        async with httpx.AsyncClient(timeout=config.llm.timeout) as client:
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    _, _, result = await send_single_request(client, path, item, headers, i)
                    responses.append(result)
                else:
                    responses.append({"error": "Invalid item in batch"})

        logger.info(f"=== BATCH RESPONSE ===")
        logger.info(f"Processed {len(responses)} items")

        return Response(
            content=json.dumps(responses, ensure_ascii=False),
            status_code=200,
            headers={"Content-Type": "application/json"},
        )

    # 単一リクエスト
    if path.endswith("v1/chat/completions") and isinstance(data, dict):
        fix_completion_request(data)

    async with httpx.AsyncClient(timeout=config.llm.timeout) as client:
        r = await client.request(
            req.method,
            f"{config.llm.endpoint}/{path}",
            headers=headers,
            json=data if data is not None else None,
            content=None if data is not None else raw,
        )

    logger.info(f"=== RESPONSE ===")
    logger.info(f"Status: {r.status_code}")
    logger.debug(f"Response content: {r.content[:2000].decode('utf-8', errors='replace') if r.content else 'None'}...")

    # レスポンスをサニタイズ
    if path.endswith("v1/chat/completions"):
        response_data = parse_json(r.content)
        if response_data:
            sanitized = sanitize_response(response_data)
            sanitized_json = json.dumps(sanitized, ensure_ascii=False)
            logger.info(f"Sanitized response (removed __verbose)")
            return Response(
                content=sanitized_json,
                status_code=r.status_code,
                headers={"Content-Type": "application/json"},
            )

    return Response(
        content=r.content,
        status_code=r.status_code,
        headers={"Content-Type": r.headers.get("content-type", "application/json")},
    )


def main():
    global config, pipeline

    parser = argparse.ArgumentParser(
        description="xTranslator to llama.cpp proxy server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  translator-proxy --config config.toml
  translator-proxy --llama http://0.0.0.0:8080
  translator-proxy --port 8081 --model my-model.gguf
  translator-proxy --log proxy.log --verbose
        """,
    )
    parser.add_argument(
        "--config", "-c",
        help="TOML config file path (default: config.toml if exists)",
    )
    parser.add_argument(
        "--llama", "-l",
        help="llama.cpp server URL (overrides config)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        help="Proxy server port (overrides config)",
    )
    parser.add_argument(
        "--host", "-H",
        help="Proxy server host (overrides config)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Model name to use (overrides config)",
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        help="Default max_tokens (overrides config)",
    )
    parser.add_argument(
        "--log-dir", "-L",
        help="Log directory path (overrides config). Log files are created with timestamp.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (overrides config)",
    )

    args = parser.parse_args()

    # 設定ファイルの読み込み（必須）
    config_path = find_config_file(args.config)
    if not config_path:
        print("エラー: 設定ファイルが見つかりません", file=sys.stderr)
        print("  --config オプションで指定するか、config.toml を配置してください", file=sys.stderr)
        sys.exit(1)

    print(f"Loading config from: {config_path}")
    try:
        config = Config.load(config_path)
    except Exception as e:
        print(f"設定ファイルの読み込みエラー: {e}", file=sys.stderr)
        sys.exit(1)

    # CLIオプションで設定を上書き
    if args.llama:
        config.llm.endpoint = args.llama
    if args.model:
        config.llm.model = args.model
    if args.max_tokens:
        config.llm.max_tokens = args.max_tokens
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.log_dir:
        config.server.log_dir = args.log_dir
    if args.verbose:
        config.server.verbose = True

    # ログ設定
    log_file_path = setup_logging(config.server.log_dir, config.server.verbose)

    # パイプライン初期化
    pipeline = Pipeline(config)

    logger.info(f"Starting translator-proxy")
    logger.info(f"  llama.cpp: {config.llm.endpoint}")
    logger.info(f"  Model: {config.llm.model}")
    logger.info(f"  Max tokens: {config.llm.max_tokens}")
    logger.info(f"  Listen: {config.server.host}:{config.server.port}")
    if log_file_path:
        logger.info(f"  Log file: {log_file_path}")
    if pipeline.is_enabled():
        active_steps = pipeline.get_active_steps()
        logger.info(f"  Pipeline: enabled ({len(active_steps)} steps)")
        for step in active_steps:
            logger.info(f"    - {step.name}")
    else:
        logger.info(f"  Pipeline: disabled (passthrough mode)")

    # 暴走検出設定
    if config.server.runaway_detection:
        logger.info(f"  Runaway detection: enabled (prefix_length={config.server.runaway_prefix_length})")
        if config.server.runaway_ng_words:
            logger.info(f"  NG words: {len(config.server.runaway_ng_words)} configured")
            for ng_word in config.server.runaway_ng_words:
                logger.info(f"    - {ng_word[:50]}{'...' if len(ng_word) > 50 else ''}")
    else:
        logger.info(f"  Runaway detection: disabled")
    if config.server.empty_input_as_error:
        logger.info(f"  Empty input: treated as error")
    if config.server.placeholder_replacements:
        logger.info(f"  Placeholder replacements: {len(config.server.placeholder_replacements)} configured")
        for replacement in config.server.placeholder_replacements:
            if len(replacement) >= 2:
                logger.info(f"    - {replacement[0]} → {replacement[1]}")

    # サーバー起動
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="warning",  # uvicorn自体のログは抑制
    )


if __name__ == "__main__":
    main()
