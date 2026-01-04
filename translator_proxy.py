#!/usr/bin/env python3
"""
xTranslator to llama.cpp proxy server

Usage:
    uv run translator-proxy --config config.toml
    uv run translator-proxy --llama http://0.0.0.0:8080 --port 18080
    uv run translator-proxy --help
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from fastapi import FastAPI, Request, Response
import httpx
import uvicorn

from config import Config, find_config_file
from pipeline import Pipeline

# グローバル設定（起動時に初期化）
config: Config | None = None
pipeline: Pipeline | None = None

app = FastAPI(title="Translator Proxy", description="xTranslator to llama.cpp proxy")
logger = logging.getLogger("translator-proxy")


def setup_logging(log_file: str | None, verbose: bool):
    """ログ設定"""
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


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

    # CRLF 正規化
    msgs = item.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict):
        c = msgs[0].get("content")
        if isinstance(c, str):
            msgs[0]["content"] = c.replace("\r\n", "\n")


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

    # CRLF検出によるバッチ拒否（XMLバッチ処理の検出）
    if (
        config.server.crlf_threshold > 0
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
                        status_code=400,
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
        "--log", "-L",
        help="Log file path (overrides config)",
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
    if args.log:
        config.server.log_file = args.log
    if args.verbose:
        config.server.verbose = True

    # ログ設定
    setup_logging(config.server.log_file, config.server.verbose)

    # パイプライン初期化
    pipeline = Pipeline(config)

    logger.info(f"Starting translator-proxy")
    logger.info(f"  llama.cpp: {config.llm.endpoint}")
    logger.info(f"  Model: {config.llm.model}")
    logger.info(f"  Max tokens: {config.llm.max_tokens}")
    logger.info(f"  Listen: {config.server.host}:{config.server.port}")
    if pipeline.is_enabled():
        active_steps = pipeline.get_active_steps()
        logger.info(f"  Pipeline: enabled ({len(active_steps)} steps)")
        for step in active_steps:
            logger.info(f"    - {step.name}")
    else:
        logger.info(f"  Pipeline: disabled (passthrough mode)")

    # サーバー起動
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="warning",  # uvicorn自体のログは抑制
    )


if __name__ == "__main__":
    main()
