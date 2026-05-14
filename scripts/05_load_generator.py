#!/usr/bin/env python3
"""
Phase 3: Load generator for GPU utilization profiling.
Generates sustained load at various concurrency levels while
py-spy and rocm-smi capture CPU and GPU utilization.
"""
import asyncio
import aiohttp
import json
import time
import os
import sys

MODEL = os.environ.get("MODEL_NAME", "/work/models/MiniMax-M2.5")
URL = os.environ.get("API_URL", "http://127.0.0.1:8000/v1/chat/completions")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/work/results/cpu-gpu-codesign/phase3")


async def send_request(session, ctx_size):
    text = "Analyze this code for performance issues. " * (ctx_size // 6 + 1)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are helpful. " * 200},
            {"role": "user", "content": text[:ctx_size * 4]},
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    token_count = 0
    first_token_time = None
    try:
        async with session.post(URL, json=payload) as resp:
            async for line in resp.content:
                line_str = line.decode().strip()
                if line_str.startswith("data: ") and line_str[6:] != "[DONE]":
                    try:
                        c = json.loads(line_str[6:])
                        delta = c.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content") or delta.get("reasoning"):
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            token_count += 1
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        return {"error": str(e), "ctx": ctx_size}

    total_time = time.perf_counter() - t0
    ttft = (first_token_time - t0) if first_token_time else total_time
    return {
        "ctx": ctx_size,
        "tokens": token_count,
        "time_s": total_time,
        "ttft_s": ttft,
    }


async def run_phase(concurrency, ctx_size, duration_s=30):
    print(f"  Running conc={concurrency}, ctx={ctx_size} for {duration_s}s...")
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = []
        start = time.perf_counter()
        while time.perf_counter() - start < duration_s:
            tasks = [send_request(session, ctx_size) for _ in range(concurrency)]
            batch = await asyncio.gather(*tasks, return_exceptions=True)
            for r in batch:
                if isinstance(r, dict) and "error" not in r:
                    results.append(r)
        return results


async def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    phases = [
        (1, 1000, 20),
        (4, 8000, 30),
        (16, 32000, 40),
        (32, 32000, 40),
        (32, 100000, 40),
    ]

    for conc, ctx, dur in phases:
        label = f"conc{conc}_ctx{ctx}"
        print(f"Phase 3 load: {label}")
        results = await run_phase(conc, ctx, dur)

        if results:
            times = [r["time_s"] for r in results]
            ttfts = [r["ttft_s"] for r in results]
            all_results[label] = {
                "concurrency": conc,
                "context_tokens": ctx,
                "duration_s": dur,
                "count": len(results),
                "mean_time_s": sum(times) / len(times),
                "p50_time_s": sorted(times)[len(times) // 2],
                "mean_ttft_s": sum(ttfts) / len(ttfts),
                "total_tokens": sum(r["tokens"] for r in results),
                "throughput_req_per_s": len(results) / dur,
            }
            mt = all_results[label]["mean_time_s"]
            print(f"  -> {len(results)} requests, {mt:.1f}s mean, "
                  f"{all_results[label]['throughput_req_per_s']:.1f} req/s")
        else:
            all_results[label] = {"count": 0, "error": "no successful requests"}
            print(f"  -> 0 successful requests")

    output_file = os.path.join(OUTPUT_DIR, "load_test_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
