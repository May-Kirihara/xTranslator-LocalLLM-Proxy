#!/usr/bin/env python3
"""
CRLF分割バッチ処理のテストスクリプト

使用方法:
    # llama.cppサーバーへの直接テスト
    uv run python test/test_crlf_batch.py --direct

    # プロキシ経由のテスト（プロキシ起動後）
    uv run python test/test_crlf_batch.py --proxy

    # 両方のテスト
    uv run python test/test_crlf_batch.py --all

    # カスタムエンドポイント
    uv run python test/test_crlf_batch.py --direct --llm-endpoint http://<YOUR_LLM_SERVER>:8080
    uv run python test/test_crlf_batch.py --proxy --proxy-endpoint http://localhost:18080
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config


def load_config() -> Config:
    """設定ファイルを読み込む"""
    config_path = Path(__file__).parent.parent / "config.toml"
    if config_path.exists():
        return Config.load(config_path)

    # デフォルト設定
    example_path = Path(__file__).parent.parent / "config.example.toml"
    if example_path.exists():
        return Config.load(example_path)

    raise FileNotFoundError("config.toml または config.example.toml が見つかりません")


async def test_llm_health(endpoint: str) -> bool:
    """LLMサーバーのヘルスチェック"""
    print(f"\n=== LLMサーバーヘルスチェック: {endpoint} ===")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{endpoint}/health")
            if r.status_code == 200:
                print(f"✓ ステータス: OK")
                return True
            else:
                print(f"✗ ステータス: {r.status_code}")
                return False
    except Exception as e:
        print(f"✗ 接続エラー: {e}")
        return False


async def test_llm_slots(endpoint: str) -> dict | None:
    """LLMサーバーのスロット情報を取得"""
    print(f"\n=== LLMサーバースロット情報 ===")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{endpoint}/slots")
            if r.status_code == 200:
                slots = r.json()
                print(f"✓ 総スロット数: {len(slots)}")
                for slot in slots:
                    print(f"  - Slot {slot['id']}: n_ctx={slot['n_ctx']}, is_processing={slot['is_processing']}")
                return {"total_slots": len(slots), "slots": slots}
            else:
                print(f"✗ スロット情報取得失敗: {r.status_code}")
                return None
    except Exception as e:
        print(f"✗ エラー: {e}")
        return None


async def test_single_request(endpoint: str, prompt: str, max_tokens: int = 100) -> str | None:
    """単一リクエストのテスト"""
    print(f"\n=== 単一リクエストテスト ===")
    print(f"プロンプト: {prompt[:50]}...")

    request_data = {
        "model": "test",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    }

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{endpoint}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )

            if r.status_code == 200:
                result = r.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                tokens = result.get("usage", {}).get("prompt_tokens", 0)
                print(f"✓ ステータス: 200")
                print(f"  プロンプトトークン数: {tokens}")
                print(f"  レスポンス: {content[:100]}")
                return content
            else:
                print(f"✗ ステータス: {r.status_code}")
                print(f"  エラー: {r.text[:200]}")
                return None
    except Exception as e:
        print(f"✗ エラー: {e}")
        return None


async def test_parallel_requests(
    endpoint: str,
    prompts: list[str],
    max_tokens: int = 100
) -> list[tuple[int, str | None, str | None]]:
    """並列リクエストのテスト

    Returns:
        (インデックス, レスポンス, エラー) のリスト
    """
    print(f"\n=== 並列リクエストテスト ({len(prompts)}件) ===")

    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for i, prompt in enumerate(prompts):
            request_data = {
                "model": "test",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens
            }
            task = client.post(
                f"{endpoint}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            tasks.append((i, prompt, task))

        print(f"  {len(tasks)}件のリクエストを送信中...")

        # 並列実行
        results = await asyncio.gather(*[t[2] for t in tasks], return_exceptions=True)

        output = []
        success_count = 0
        for i, (idx, prompt, _) in enumerate(tasks):
            result = results[i]
            if isinstance(result, Exception):
                print(f"  [{idx}] ✗ 例外: {result}")
                output.append((idx, None, str(result)))
            elif result.status_code == 200:
                content = result.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  [{idx}] ✓ {prompt[:20]}... → {content[:30]}")
                output.append((idx, content, None))
                success_count += 1
            else:
                error = result.json().get("error", {}).get("message", result.text[:100])
                print(f"  [{idx}] ✗ ステータス {result.status_code}: {error}")
                output.append((idx, None, error))

        print(f"\n  結果: {success_count}/{len(tasks)} 成功")
        return output


async def test_crlf_batch_via_proxy(
    proxy_endpoint: str,
    texts: list[str]
) -> str | None:
    """プロキシ経由のCRLF分割バッチ処理テスト"""
    print(f"\n=== CRLF分割バッチ処理テスト（プロキシ経由） ===")

    # CRLFで結合
    content = "\r\n".join(texts)
    print(f"  入力テキスト数: {len(texts)}")
    print(f"  入力: {content[:50]}...")

    request_data = {
        "model": "test",
        "messages": [{"role": "user", "content": content}]
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                f"{proxy_endpoint}/v1/chat/completions",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )

            if r.status_code == 200:
                result = r.json()
                response_content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                response_texts = response_content.split("\r\n")

                print(f"✓ ステータス: 200")
                print(f"  出力テキスト数: {len(response_texts)}")

                # 入力と出力を比較
                for i, (inp, out) in enumerate(zip(texts, response_texts)):
                    status = "✓" if out and not out.startswith("[") else "✗"
                    print(f"  [{i}] {status} {inp[:15]}... → {out[:20]}")

                return response_content
            else:
                print(f"✗ ステータス: {r.status_code}")
                print(f"  エラー: {r.text[:200]}")
                return None
    except Exception as e:
        print(f"✗ エラー: {e}")
        return None


async def test_with_pipeline_prompt(config: Config, endpoint: str) -> str | None:
    """パイプラインプロンプトを使用したテスト"""
    print(f"\n=== パイプラインプロンプトテスト ===")

    if not config.pipeline.steps:
        print("  パイプラインステップが設定されていません")
        return None

    step = config.pipeline.steps[0]
    test_text = "Dragon"

    # プロンプトを構築
    prompt = step.prompt.replace("{input}", test_text)
    prompt = prompt.replace("{original}", test_text)
    prompt = prompt.replace("{proper_nouns}", "- Dragonborn → ドラゴンボーン")

    print(f"  プロンプト長: {len(prompt)} 文字")
    print(f"  推定トークン数: ~{len(prompt) // 2}")

    return await test_single_request(endpoint, prompt)


async def run_direct_tests(llm_endpoint: str, config: Config):
    """LLMサーバーへの直接テスト"""
    print("\n" + "=" * 60)
    print("LLMサーバー直接テスト")
    print("=" * 60)

    # ヘルスチェック
    if not await test_llm_health(llm_endpoint):
        print("\n✗ LLMサーバーに接続できません")
        return False

    # スロット情報
    await test_llm_slots(llm_endpoint)

    # 単一リクエスト
    await test_single_request(llm_endpoint, "Translate to Japanese: Hello")

    # パイプラインプロンプト
    await test_with_pipeline_prompt(config, llm_endpoint)

    # 並列リクエスト（短いプロンプト）
    await test_parallel_requests(
        llm_endpoint,
        [
            "Translate to Japanese: Dragon",
            "Translate to Japanese: Sword",
            "Translate to Japanese: Shield"
        ]
    )

    return True


async def run_proxy_tests(proxy_endpoint: str):
    """プロキシ経由のテスト"""
    print("\n" + "=" * 60)
    print("プロキシ経由テスト")
    print("=" * 60)

    # ヘルスチェック
    if not await test_llm_health(proxy_endpoint):
        print("\n✗ プロキシサーバーに接続できません")
        print("  プロキシを起動してください: uv run translator-proxy")
        return False

    # CRLF分割バッチ処理テスト
    await test_crlf_batch_via_proxy(
        proxy_endpoint,
        ["Dragon", "Sword", "Shield"]
    )

    # より多くのテキストでテスト
    await test_crlf_batch_via_proxy(
        proxy_endpoint,
        ["Hello", "World", "Test", "Dragon", "Knight"]
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="CRLF分割バッチ処理のテスト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  uv run python test/test_crlf_batch.py --direct
  uv run python test/test_crlf_batch.py --proxy
  uv run python test/test_crlf_batch.py --all
  uv run python test/test_crlf_batch.py --direct --llm-endpoint http://<YOUR_LLM_SERVER>:8080
        """
    )
    parser.add_argument("--direct", action="store_true", help="LLMサーバーへの直接テスト")
    parser.add_argument("--proxy", action="store_true", help="プロキシ経由のテスト")
    parser.add_argument("--all", action="store_true", help="全てのテストを実行")
    parser.add_argument("--llm-endpoint", help="LLMサーバーのエンドポイント")
    parser.add_argument("--proxy-endpoint", default="http://localhost:18080", help="プロキシのエンドポイント")

    args = parser.parse_args()

    if not (args.direct or args.proxy or args.all):
        parser.print_help()
        sys.exit(1)

    # 設定読み込み
    try:
        config = load_config()
        llm_endpoint = args.llm_endpoint or config.llm.endpoint
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        sys.exit(1)

    print("CRLF分割バッチ処理テスト")
    print(f"LLMエンドポイント: {llm_endpoint}")
    print(f"プロキシエンドポイント: {args.proxy_endpoint}")

    async def run():
        if args.direct or args.all:
            await run_direct_tests(llm_endpoint, config)

        if args.proxy or args.all:
            await run_proxy_tests(args.proxy_endpoint)

    asyncio.run(run())

    print("\n" + "=" * 60)
    print("テスト完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
