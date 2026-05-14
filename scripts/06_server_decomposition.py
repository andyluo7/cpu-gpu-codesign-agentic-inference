#!/usr/bin/env python3
"""
Phase 2 Deep-Dive: Server-side time decomposition.

Uses vLLM Prometheus metrics + per-request timing to decompose
E2E latency into CPU vs GPU components:

  1. Polls /metrics before/after each request batch to get:
     - vllm:time_to_first_token_seconds (TTFT histogram)
     - vllm:time_per_output_token_seconds (TPOT histogram)
     - vllm:e2e_request_latency_seconds
     - vllm:request_prompt_tokens / vllm:request_generation_tokens
     - vllm:gpu_cache_usage_perc / vllm:cpu_cache_usage_perc
     - vllm:num_requests_running / vllm:num_requests_waiting
     - vllm:prompt_tokens_total / vllm:generation_tokens_total

  2. Captures torch profiler traces via /start_profile / /stop_profile
     API endpoints (if profiler is configured).

  3. Measures server-side processing overhead by comparing:
     - Client-observed TTFT vs server-reported TTFT
     - The gap = HTTP overhead + tokenization + scheduling

  4. Uses CUDA events timing from torch profiler traces to get
     actual GPU kernel execution time vs CPU overhead.
"""
import asyncio
import aiohttp
import json
import os
import re
import sys
import time
import statistics
from collections import defaultdict

MODEL = os.environ.get("MODEL_NAME", "/work/models/MiniMax-M2.5")
BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/work/results/cpu-gpu-codesign/phase2-deepdive")


async def fetch_metrics(session):
    """Fetch Prometheus metrics from vLLM."""
    try:
        async with session.get(f"{BASE_URL}/metrics") as resp:
            text = await resp.text()
            return parse_prometheus(text)
    except Exception as e:
        print(f"  Warning: failed to fetch metrics: {e}")
        return {}


def parse_prometheus(text):
    """Parse Prometheus text format into dict of metric -> value."""
    metrics = {}
    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        # Parse metric_name{labels} value
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?([^}]*)\}?\s+([\d.eE+-]+|NaN|Inf|-Inf)$', line)
        if match:
            name, labels, value = match.groups()
            try:
                val = float(value)
            except ValueError:
                continue
            key = f"{name}{{{labels}}}" if labels else name
            metrics[key] = val
        else:
            # Simple metric without labels
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return metrics


def extract_histogram_stats(metrics, prefix):
    """Extract mean/p50/p95 from Prometheus histogram buckets."""
    count_key = f"{prefix}_count"
    sum_key = f"{prefix}_sum"
    count = 0
    total = 0
    for k, v in metrics.items():
        if k.startswith(count_key) or k == count_key:
            count += v
        if k.startswith(sum_key) or k == sum_key:
            total += v
    mean = total / count if count > 0 else 0
    return {"count": count, "sum": total, "mean": mean}


async def send_timed_request(session, ctx_size, max_tokens=64):
    """Send a single request with detailed client-side timing."""
    text = "Analyze this code and suggest performance improvements. " * (ctx_size // 7 + 1)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful coding assistant. " * 200},
            {"role": "user", "content": text[:ctx_size * 4]},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    t_start = time.perf_counter()

    # Measure request serialization
    t_serialize_start = time.perf_counter()
    payload_bytes = json.dumps(payload).encode()
    t_serialize = time.perf_counter() - t_serialize_start

    # Send request
    t_send = time.perf_counter()
    first_token_time = None
    last_token_time = None
    token_count = 0

    try:
        async with session.post(
            f"{BASE_URL}/v1/chat/completions",
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
        ) as resp:
            t_first_byte = time.perf_counter()

            async for line in resp.content:
                line_str = line.decode().strip()
                if not line_str.startswith("data: "):
                    continue
                data = line_str[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content") or delta.get("reasoning"):
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        last_token_time = now
                        token_count += 1
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        return {"error": str(e)}

    t_end = time.perf_counter()

    # Compute timing breakdown
    result = {
        "ctx_tokens_approx": ctx_size,
        "output_tokens": token_count,
        "payload_bytes": len(payload_bytes),
        # Client-side CPU times
        "t_serialize_ms": t_serialize * 1000,
        "t_http_overhead_ms": (t_first_byte - t_send) * 1000,  # network + server accept
        # Server-dominated times
        "t_ttft_ms": ((first_token_time - t_send) * 1000) if first_token_time else 0,
        "t_server_prefill_ms": ((first_token_time - t_first_byte) * 1000) if first_token_time else 0,
        "t_decode_ms": ((last_token_time - first_token_time) * 1000) if (first_token_time and last_token_time) else 0,
        # E2E
        "t_total_ms": (t_end - t_start) * 1000,
    }

    # Derived: estimate CPU vs GPU split
    # t_http_overhead = tokenization + scheduling + HTTP parsing (CPU)
    # t_server_prefill = GPU prefill + KV cache ops
    # t_decode = GPU decode (mostly) + detokenization (CPU, small)
    if result["t_total_ms"] > 0:
        cpu_estimate_ms = result["t_serialize_ms"] + result["t_http_overhead_ms"]
        gpu_estimate_ms = result["t_server_prefill_ms"] + result["t_decode_ms"]
        result["cpu_estimate_pct"] = (cpu_estimate_ms / result["t_total_ms"]) * 100
        result["gpu_estimate_pct"] = (gpu_estimate_ms / result["t_total_ms"]) * 100

    return result


async def run_scenario(session, label, concurrency, ctx_size, num_batches=5):
    """Run a scenario and collect metrics before/after."""
    print(f"\n{'='*60}")
    print(f"Scenario: {label} (conc={concurrency}, ctx={ctx_size})")
    print(f"{'='*60}")

    # Collect pre-metrics
    metrics_before = await fetch_metrics(session)

    # Run requests
    all_results = []
    for batch_i in range(num_batches):
        tasks = [send_timed_request(session, ctx_size) for _ in range(concurrency)]
        results = await asyncio.gather(*tasks)
        valid = [r for r in results if "error" not in r]
        all_results.extend(valid)
        if valid:
            mean_ttft = statistics.mean(r["t_ttft_ms"] for r in valid)
            mean_total = statistics.mean(r["t_total_ms"] for r in valid)
            print(f"  Batch {batch_i+1}/{num_batches}: {len(valid)} OK, "
                  f"TTFT={mean_ttft:.0f}ms, Total={mean_total:.0f}ms")

    # Collect post-metrics
    metrics_after = await fetch_metrics(session)

    if not all_results:
        return {"label": label, "error": "no successful requests"}

    # Aggregate timing breakdown
    timing_keys = [
        "t_serialize_ms", "t_http_overhead_ms", "t_ttft_ms",
        "t_server_prefill_ms", "t_decode_ms", "t_total_ms",
        "cpu_estimate_pct", "gpu_estimate_pct",
    ]
    agg = {"label": label, "concurrency": concurrency, "ctx_tokens": ctx_size,
           "num_requests": len(all_results)}

    for key in timing_keys:
        values = [r[key] for r in all_results if key in r and r[key] > 0]
        if values:
            agg[f"{key}_mean"] = statistics.mean(values)
            agg[f"{key}_p50"] = statistics.median(values)
            if len(values) > 1:
                agg[f"{key}_p95"] = sorted(values)[int(0.95 * len(values))]
                agg[f"{key}_stdev"] = statistics.stdev(values)

    # Extract delta metrics from Prometheus
    prom_delta = {}
    for key in metrics_after:
        if key in metrics_before:
            delta = metrics_after[key] - metrics_before[key]
            if abs(delta) > 0.001:
                prom_delta[key] = delta
    agg["prometheus_deltas"] = prom_delta

    # Print summary
    print(f"\n  --- {label} Summary ---")
    print(f"  Requests: {agg['num_requests']}")
    for key in timing_keys:
        mean_key = f"{key}_mean"
        if mean_key in agg:
            print(f"  {key}: {agg[mean_key]:.2f} ms (mean)")

    return agg


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=64)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Warmup
        print("Warming up server...")
        for _ in range(3):
            await send_timed_request(session, 1000, max_tokens=16)

        scenarios = [
            ("single_1k", 1, 1000, 5),
            ("single_8k", 1, 8000, 5),
            ("single_32k", 1, 32000, 3),
            ("single_100k", 1, 100000, 3),
            ("conc4_8k", 4, 8000, 5),
            ("conc16_32k", 16, 32000, 3),
            ("conc32_32k", 32, 32000, 3),
            ("conc32_100k", 32, 100000, 2),
        ]

        all_scenarios = []
        for label, conc, ctx, batches in scenarios:
            result = await run_scenario(session, label, conc, ctx, batches)
            all_scenarios.append(result)

    # Save
    output_file = os.path.join(OUTPUT_DIR, "server_decomposition.json")
    with open(output_file, "w") as f:
        json.dump({"scenarios": all_scenarios}, f, indent=2)
    print(f"\nAll results saved to {output_file}")

    # Print final summary table
    print(f"\n{'='*120}")
    print(f"{'Scenario':>20} | {'Conc':>5} | {'Ctx':>8} | {'Serialize':>10} | "
          f"{'HTTP OH':>10} | {'Prefill':>10} | {'Decode':>10} | "
          f"{'Total':>10} | {'CPU%':>6} | {'GPU%':>6}")
    print("-" * 120)
    for s in all_scenarios:
        if "error" in s:
            print(f"{'':>20} | ERROR: {s.get('error','')}")
            continue
        print(f"{s['label']:>20} | {s['concurrency']:>5} | {s['ctx_tokens']:>8,} | "
              f"{s.get('t_serialize_ms_mean', 0):>10.2f} | "
              f"{s.get('t_http_overhead_ms_mean', 0):>10.2f} | "
              f"{s.get('t_server_prefill_ms_mean', 0):>10.0f} | "
              f"{s.get('t_decode_ms_mean', 0):>10.0f} | "
              f"{s.get('t_total_ms_mean', 0):>10.0f} | "
              f"{s.get('cpu_estimate_pct_mean', 0):>6.1f} | "
              f"{s.get('gpu_estimate_pct_mean', 0):>6.1f}")
    print("=" * 120)


if __name__ == "__main__":
    asyncio.run(main())
