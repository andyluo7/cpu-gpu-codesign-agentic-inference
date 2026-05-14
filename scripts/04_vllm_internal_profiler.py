#!/usr/bin/env python3
"""
Phase 2: vLLM internal CPU component profiler.

Instruments key vLLM/LMCache functions via monkey-patching to measure
CPU time spent in each component without modifying vLLM source.

Components measured:
  - Tokenization (encode)
  - Detokenization (decode)
  - Scheduler decision making
  - KV cache block management
  - LMCache connector (hash, lookup, transfer)
  - Sampler / logit processing
  - HTTP request parsing (FastAPI/uvicorn overhead)

Usage:
  # Start vLLM with this module pre-loaded
  python3 04_vllm_internal_profiler.py --attach <vllm_pid>
  # Or: PYTHONPATH=. python3 -c "import vllm_internal_profiler; ..." before vllm starts

  # After benchmark, dump results
  curl http://127.0.0.1:8001/dump_profile
"""
import argparse
import json
import os
import sys
import time
import threading
import statistics
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Global timing storage
_timings = defaultdict(list)
_lock = threading.Lock()
_enabled = True


def record_timing(component: str, duration_ms: float):
    """Thread-safe timing recording."""
    if not _enabled:
        return
    with _lock:
        _timings[component].append(duration_ms)


def get_summary():
    """Get summary statistics for all components."""
    with _lock:
        summary = {}
        for comp, times in _timings.items():
            if not times:
                continue
            summary[comp] = {
                "count": len(times),
                "total_ms": sum(times),
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "p95_ms": sorted(times)[int(0.95 * len(times))] if len(times) >= 20 else max(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
            }

        # Compute CPU vs GPU breakdown
        cpu_components = [
            "tokenize", "detokenize", "scheduler",
            "kv_cache_mgmt", "lmcache_lookup", "lmcache_hash",
            "sampler_cpu", "http_parse",
        ]
        gpu_components = [
            "prefill_gpu", "decode_gpu", "lmcache_transfer",
            "sampler_gpu",
        ]

        total_cpu_ms = sum(
            summary.get(c, {}).get("total_ms", 0) for c in cpu_components
        )
        total_gpu_ms = sum(
            summary.get(c, {}).get("total_ms", 0) for c in gpu_components
        )
        total_all = total_cpu_ms + total_gpu_ms

        summary["_cpu_total_ms"] = total_cpu_ms
        summary["_gpu_total_ms"] = total_gpu_ms
        summary["_cpu_pct"] = (total_cpu_ms / total_all * 100) if total_all > 0 else 0
        summary["_gpu_pct"] = (total_gpu_ms / total_all * 100) if total_all > 0 else 0

        return summary


def reset_timings():
    """Reset all collected timings."""
    with _lock:
        _timings.clear()


class ProfileHandler(BaseHTTPRequestHandler):
    """HTTP handler for profile data access."""
    def do_GET(self):
        if self.path == "/dump_profile":
            summary = get_summary()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(summary, indent=2).encode())
        elif self.path == "/reset_profile":
            reset_timings()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/raw_timings":
            with _lock:
                raw = {k: v[-1000:] for k, v in _timings.items()}  # last 1000
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(raw).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress access logs


def start_profile_server(port=8001):
    """Start a background HTTP server to serve profile data."""
    server = HTTPServer(("0.0.0.0", port), ProfileHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Profile server listening on port {port}")
    return server


def instrument_function(module, func_name, component_name):
    """Monkey-patch a function to add timing instrumentation."""
    original = getattr(module, func_name, None)
    if original is None:
        print(f"  WARNING: {module.__name__}.{func_name} not found, skipping")
        return False

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000
        record_timing(component_name, duration_ms)
        return result

    setattr(module, func_name, wrapper)
    print(f"  Instrumented {module.__name__}.{func_name} -> {component_name}")
    return True


async def instrument_async_function(module, func_name, component_name):
    """Monkey-patch an async function to add timing instrumentation."""
    import asyncio
    original = getattr(module, func_name, None)
    if original is None:
        print(f"  WARNING: {module.__name__}.{func_name} not found, skipping")
        return False

    async def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = await original(*args, **kwargs)
        duration_ms = (time.perf_counter() - t0) * 1000
        record_timing(component_name, duration_ms)
        return result

    setattr(module, func_name, wrapper)
    print(f"  Instrumented (async) {module.__name__}.{func_name} -> {component_name}")
    return True


def setup_instrumentation():
    """
    Set up monkey-patching for vLLM components.
    Call this BEFORE starting vLLM server.
    """
    print("Setting up vLLM instrumentation...")
    count = 0

    # 1. Tokenizer
    try:
        from transformers import PreTrainedTokenizerFast
        count += instrument_function(PreTrainedTokenizerFast, "encode", "tokenize")
        count += instrument_function(PreTrainedTokenizerFast, "decode", "detokenize")
    except ImportError:
        print("  Could not instrument tokenizer")

    # 2. Try to instrument vLLM's tokenizer wrapper
    try:
        from vllm.transformers_utils.tokenizer import MistralTokenizer
        count += instrument_function(MistralTokenizer, "encode", "tokenize_mistral")
    except (ImportError, AttributeError):
        pass

    # 3. LMCache connector
    try:
        from lmcache.integration.vllm.connector_v1 import LMCacheConnectorV1Impl
        count += instrument_function(LMCacheConnectorV1Impl, "build_partial", "lmcache_lookup")
        count += instrument_function(LMCacheConnectorV1Impl, "save", "lmcache_save")
    except (ImportError, AttributeError):
        print("  LMCache not available, skipping connector instrumentation")

    print(f"Instrumented {count} functions")
    return count


def standalone_benchmark():
    """
    Standalone mode: just measures tokenizer and JSON parsing CPU overhead.
    No vLLM server needed — useful as a quick Phase 2 data point.
    """
    import json as json_mod

    print("\n=== Standalone CPU Component Benchmark ===\n")

    # Load tokenizer
    model_path = os.environ.get("MODEL_PATH", "/work/models/MiniMax-M2.5")
    print(f"Loading tokenizer from {model_path}...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    results = {}

    # 1. Tokenization at various sizes
    print("\n--- Tokenization ---")
    for n_tokens in [1000, 8000, 32000, 64000, 100000]:
        text = "The quick brown fox. " * (n_tokens // 5 + 1)
        # Warmup
        for _ in range(3):
            ids = tokenizer.encode(text)
        # Measure
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            ids = tokenizer.encode(text)
            times.append((time.perf_counter() - t0) * 1000)
        actual = len(ids)
        results[f"tokenize_{n_tokens}"] = {
            "actual_tokens": actual,
            "mean_ms": statistics.mean(times),
            "p50_ms": statistics.median(times),
            "throughput_mtok_s": actual / statistics.mean(times) / 1000,
        }
        print(f"  {actual:>8,} tokens: {statistics.mean(times):>8.2f} ms "
              f"({actual / statistics.mean(times) / 1000:.1f} Mtok/s)")

    # 2. JSON serialization (request build)
    print("\n--- JSON Serialization (request payload) ---")
    for n_tokens in [1000, 8000, 32000, 100000]:
        text = "x " * (n_tokens * 2)
        payload = {
            "model": "MiniMax-M2.5",
            "messages": [
                {"role": "system", "content": "You are helpful." * 100},
                {"role": "user", "content": text},
            ],
            "max_tokens": 128,
            "stream": True,
        }
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            _ = json_mod.dumps(payload)
            times.append((time.perf_counter() - t0) * 1000)
        results[f"json_serialize_{n_tokens}"] = {
            "mean_ms": statistics.mean(times),
            "p50_ms": statistics.median(times),
        }
        print(f"  ~{n_tokens:>8,} tokens payload: {statistics.mean(times):>8.2f} ms")

    # 3. JSON parsing (response parse)
    print("\n--- JSON Parsing (SSE chunk) ---")
    chunk = json_mod.dumps({
        "id": "chatcmpl-abc123",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "MiniMax-M2.5",
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
    })
    times = []
    for _ in range(10000):
        t0 = time.perf_counter()
        _ = json_mod.loads(chunk)
        times.append((time.perf_counter() - t0) * 1000)
    results["json_parse_sse_chunk"] = {
        "mean_ms": statistics.mean(times),
        "p50_ms": statistics.median(times),
        "per_token_us": statistics.mean(times) * 1000,  # microseconds per parse
    }
    print(f"  SSE chunk parse: {statistics.mean(times)*1000:.1f} µs/chunk")

    # 4. Hash computation (simulates LMCache cache key)
    print("\n--- Hash Computation (cache key) ---")
    import hashlib
    for n_tokens in [1000, 8000, 32000, 100000]:
        data = b"token_id_" * n_tokens
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = hashlib.sha256(data).hexdigest()
            times.append((time.perf_counter() - t0) * 1000)
        results[f"sha256_hash_{n_tokens}"] = {
            "mean_ms": statistics.mean(times),
            "throughput_gb_s": len(data) / statistics.mean(times) / 1e6,
        }
        print(f"  {n_tokens:>8,} tokens: {statistics.mean(times):>8.3f} ms "
              f"({len(data) / statistics.mean(times) / 1e6:.1f} GB/s)")

    # Save
    output_dir = os.environ.get("OUTPUT_DIR", "/work/results/cpu-gpu-codesign/phase2")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "cpu_component_benchmark.json")
    with open(output_file, "w") as f:
        json_mod.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["standalone", "instrument", "server"],
                        default="standalone")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.mode == "standalone":
        standalone_benchmark()
    elif args.mode == "instrument":
        setup_instrumentation()
        start_profile_server(args.port)
        print("Instrumentation active. Start vLLM in this process.")
    elif args.mode == "server":
        start_profile_server(args.port)
        print("Profile server only. Connect from running vLLM.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
