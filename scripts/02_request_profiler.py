#!/usr/bin/env python3
"""
Phase 1 & 2: E2E request profiler — client-side time decomposition.

Sends requests to vLLM server and measures:
  - t_request_build: time to construct the OpenAI request payload
  - t_network_connect: time from send to first byte (includes server processing)
  - t_ttft: time to first token (first SSE data event with content)
  - t_streaming: time from first token to last token
  - t_response_parse: time to parse final response + extract tool calls
  - t_total: end-to-end

Runs multiple scenarios: varying context length and concurrency.
"""
import argparse
import asyncio
import json
import os
import sys
import time
import statistics
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import aiohttp
except ImportError:
    print("pip install aiohttp")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("pip install transformers")
    sys.exit(1)


@dataclass
class RequestTiming:
    context_tokens: int = 0
    output_tokens: int = 0
    t_request_build_ms: float = 0
    t_network_to_first_byte_ms: float = 0
    t_ttft_ms: float = 0
    t_streaming_ms: float = 0
    t_response_parse_ms: float = 0
    t_total_ms: float = 0
    # Derived
    t_cpu_client_ms: float = 0  # request_build + response_parse
    t_server_ms: float = 0     # network_to_first_byte + streaming (includes GPU)
    error: Optional[str] = None


def generate_conversation(tokenizer, target_input_tokens: int, system_prompt_tokens: int = 2000):
    """Generate a multi-turn conversation that hits target input token count."""
    system_text = (
        "You are a helpful coding assistant with access to tools. "
        "You can read files, write files, run shell commands, and search code. "
        "Always think step by step before acting. "
    ) * 50  # ~2k tokens worth of system prompt

    system_ids = tokenizer.encode(system_text)[:system_prompt_tokens]
    system_text = tokenizer.decode(system_ids)

    # Fill remaining tokens with user/assistant turns
    remaining = target_input_tokens - system_prompt_tokens - 50  # buffer for formatting
    filler = (
        "Please analyze the following code and suggest improvements for performance, "
        "readability, and maintainability. Consider edge cases and error handling. "
        "The codebase uses Python 3.12 with asyncio for concurrent operations. "
    ) * 200

    filler_ids = tokenizer.encode(filler)[:max(remaining, 100)]
    filler_text = tokenizer.decode(filler_ids)

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": filler_text},
    ]
    return messages


async def send_request_profiled(
    session: aiohttp.ClientSession,
    url: str,
    messages: list,
    max_tokens: int = 128,
) -> RequestTiming:
    """Send a single request and capture fine-grained timing."""
    timing = RequestTiming()

    # 1. Request build time
    t0 = time.perf_counter()
    payload = {
        "model": os.environ.get("MODEL_NAME", "/work/models/MiniMax-M2.5"),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    payload_json = json.dumps(payload)
    timing.t_request_build_ms = (time.perf_counter() - t0) * 1000

    # Count input tokens (approximate from message content)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    timing.context_tokens = total_chars // 4  # rough estimate

    # 2. Send request
    t_send = time.perf_counter()
    first_byte_time = None
    first_token_time = None
    last_token_time = None
    token_count = 0
    full_response = ""

    try:
        async with session.post(
            url,
            data=payload_json,
            headers={"Content-Type": "application/json"},
        ) as resp:
            # First byte = headers received
            first_byte_time = time.perf_counter()
            timing.t_network_to_first_byte_ms = (first_byte_time - t_send) * 1000

            # Stream SSE
            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content") or delta.get("reasoning") or ""
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            last_token_time = time.perf_counter()
                            token_count += 1
                            full_response += content
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        timing.error = str(e)
        timing.t_total_ms = (time.perf_counter() - t0) * 1000
        return timing

    t_stream_done = time.perf_counter()

    # 3. Response parse time (simulate tool call extraction)
    t_parse_start = time.perf_counter()
    # Simulate parsing: check for tool calls, JSON extraction
    try:
        # Attempt to parse as JSON (tool call scenario)
        _ = json.loads(full_response) if full_response.strip().startswith("{") else None
    except (json.JSONDecodeError, ValueError):
        pass
    # Simulate structured output validation
    _ = len(full_response.split())
    timing.t_response_parse_ms = (time.perf_counter() - t_parse_start) * 1000

    # Fill timing
    if first_token_time:
        timing.t_ttft_ms = (first_token_time - t_send) * 1000
    if first_token_time and last_token_time:
        timing.t_streaming_ms = (last_token_time - first_token_time) * 1000
    timing.output_tokens = token_count
    timing.t_total_ms = (time.perf_counter() - t0) * 1000
    timing.t_cpu_client_ms = timing.t_request_build_ms + timing.t_response_parse_ms
    timing.t_server_ms = timing.t_total_ms - timing.t_cpu_client_ms

    return timing


async def run_concurrent_requests(
    url: str,
    messages_list: list,
    concurrency: int,
    max_tokens: int = 128,
) -> list[RequestTiming]:
    """Run multiple requests concurrently and collect timings."""
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for msgs in messages_list[:concurrency]:
            tasks.append(send_request_profiled(session, url, msgs, max_tokens))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    timings = []
    for r in results:
        if isinstance(r, Exception):
            t = RequestTiming(error=str(r))
            timings.append(t)
        else:
            timings.append(r)
    return timings


async def main():
    parser = argparse.ArgumentParser(description="E2E request profiler")
    parser.add_argument("--api-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", default="/work/models/MiniMax-M2.5")
    parser.add_argument("--output-dir", default="/work/results/cpu-gpu-codesign/phase1")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Iterations per (context, concurrency) pair")
    args = parser.parse_args()

    url = f"{args.api_endpoint}/v1/chat/completions"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Test matrix
    context_lengths = [1_000, 8_000, 32_000, 100_000]
    concurrency_levels = [1, 4, 16, 32]

    all_results = []

    for ctx_len in context_lengths:
        print(f"\n{'='*60}")
        print(f"Context: {ctx_len:,} tokens")
        print(f"{'='*60}")

        # Pre-generate messages for all concurrency levels
        max_conc = max(concurrency_levels)
        messages_list = []
        for i in range(max_conc):
            msgs = generate_conversation(tokenizer, ctx_len)
            messages_list.append(msgs)

        for conc in concurrency_levels:
            print(f"\n  Concurrency: {conc}")

            iter_timings = []
            for it in range(args.num_iters):
                timings = await run_concurrent_requests(
                    url, messages_list, conc, args.max_tokens
                )
                iter_timings.extend(timings)
                ok = sum(1 for t in timings if t.error is None)
                err = sum(1 for t in timings if t.error is not None)
                if err > 0:
                    print(f"    iter {it}: {ok} OK, {err} errors")

            # Aggregate
            valid = [t for t in iter_timings if t.error is None]
            if not valid:
                print(f"    ALL FAILED")
                continue

            agg = {
                "context_tokens": ctx_len,
                "concurrency": conc,
                "num_requests": len(valid),
                "num_errors": len(iter_timings) - len(valid),
            }

            for metric in [
                "t_request_build_ms", "t_network_to_first_byte_ms",
                "t_ttft_ms", "t_streaming_ms", "t_response_parse_ms",
                "t_total_ms", "t_cpu_client_ms", "t_server_ms",
            ]:
                values = [getattr(t, metric) for t in valid if getattr(t, metric) > 0]
                if values:
                    agg[f"{metric}_mean"] = statistics.mean(values)
                    agg[f"{metric}_p50"] = statistics.median(values)
                    agg[f"{metric}_p95"] = sorted(values)[int(0.95 * len(values))]
                    agg[f"{metric}_min"] = min(values)
                    agg[f"{metric}_max"] = max(values)

            # CPU vs GPU split
            if "t_total_ms_mean" in agg and agg["t_total_ms_mean"] > 0:
                cpu_pct = (agg.get("t_cpu_client_ms_mean", 0) / agg["t_total_ms_mean"]) * 100
                agg["cpu_client_pct"] = round(cpu_pct, 2)
                agg["server_pct"] = round(100 - cpu_pct, 2)

            all_results.append(agg)

            print(f"    Total: {agg.get('t_total_ms_mean', 0):.1f} ms mean")
            print(f"    TTFT:  {agg.get('t_ttft_ms_mean', 0):.1f} ms mean")
            print(f"    Build: {agg.get('t_request_build_ms_mean', 0):.2f} ms | "
                  f"Parse: {agg.get('t_response_parse_ms_mean', 0):.2f} ms")
            print(f"    CPU client: {agg.get('cpu_client_pct', 0):.1f}% | "
                  f"Server: {agg.get('server_pct', 0):.1f}%")

    # Save results
    output_file = os.path.join(output_dir, "request_profiler.json")
    with open(output_file, "w") as f:
        json.dump({"results": all_results}, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary table
    print(f"\n{'='*100}")
    print(f"{'Context':>10} | {'Conc':>5} | {'Total(ms)':>10} | {'TTFT(ms)':>10} | "
          f"{'Build(ms)':>10} | {'Parse(ms)':>10} | {'CPU%':>6} | {'Server%':>8}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['context_tokens']:>10,} | {r['concurrency']:>5} | "
              f"{r.get('t_total_ms_mean', 0):>10.1f} | "
              f"{r.get('t_ttft_ms_mean', 0):>10.1f} | "
              f"{r.get('t_request_build_ms_mean', 0):>10.2f} | "
              f"{r.get('t_response_parse_ms_mean', 0):>10.2f} | "
              f"{r.get('cpu_client_pct', 0):>6.1f} | "
              f"{r.get('server_pct', 0):>8.1f}")
    print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
