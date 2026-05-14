#!/usr/bin/env python3
"""
Phase 2.1: Tokenizer CPU benchmark
Measures tokenization and detokenization time at various context lengths
for MiniMax-M2.5 tokenizer.
"""
import json
import os
import sys
import time
import statistics
from pathlib import Path

# Try to import transformers tokenizer
try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed")
    sys.exit(1)

MODEL_PATH = os.environ.get("MODEL_PATH", "/work/models/MiniMax-M2.5")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/work/results/cpu-gpu-codesign/phase2")
NUM_WARMUP = 3
NUM_ITERS = 20

# Context lengths to test (in characters, roughly 1 char ≈ 0.25 tokens for English)
CONTEXT_LENGTHS_TOKENS = [1_000, 4_000, 8_000, 16_000, 32_000, 64_000, 100_000, 150_000]


def generate_text(num_chars: int) -> str:
    """Generate deterministic text of approximately num_chars characters."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "In the realm of artificial intelligence, large language models have "
        "demonstrated remarkable capabilities across a wide range of tasks. "
        "These models, trained on vast corpora of text data, can generate "
        "coherent and contextually relevant responses to complex queries. "
        "The architecture of modern LLMs typically involves transformer-based "
        "neural networks with attention mechanisms that allow the model to "
        "capture long-range dependencies in the input text. "
    )
    repeats = (num_chars // len(base)) + 1
    return (base * repeats)[:num_chars]


def benchmark_tokenizer():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    load_time = time.perf_counter() - t0
    print(f"Tokenizer loaded in {load_time:.3f}s")
    print(f"Vocab size: {tokenizer.vocab_size}")

    results = []

    for target_tokens in CONTEXT_LENGTHS_TOKENS:
        # Generate text — estimate ~4 chars per token
        text = generate_text(target_tokens * 4)

        # Warmup
        for _ in range(NUM_WARMUP):
            ids = tokenizer.encode(text)
            _ = tokenizer.decode(ids)

        # Measure tokenization
        encode_times = []
        actual_token_count = 0
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            ids = tokenizer.encode(text)
            t1 = time.perf_counter()
            encode_times.append(t1 - t0)
            actual_token_count = len(ids)

        # Measure detokenization
        decode_times = []
        for _ in range(NUM_ITERS):
            t0 = time.perf_counter()
            _ = tokenizer.decode(ids)
            t1 = time.perf_counter()
            decode_times.append(t1 - t0)

        # Measure incremental detokenization (simulates streaming decode)
        # Process tokens one at a time (like vLLM streaming output)
        incr_decode_times = []
        chunk_sizes = [1, 8, 32]
        incr_results = {}
        for chunk_size in chunk_sizes:
            times_for_chunk = []
            for _ in range(min(NUM_ITERS, 5)):  # fewer iters for incremental
                total_t = 0
                # Only decode the last 128 tokens incrementally (simulates output)
                output_ids = ids[-128:]
                for i in range(0, len(output_ids), chunk_size):
                    chunk = output_ids[i:i+chunk_size]
                    t0 = time.perf_counter()
                    _ = tokenizer.decode(chunk)
                    t1 = time.perf_counter()
                    total_t += (t1 - t0)
                times_for_chunk.append(total_t)
            incr_results[chunk_size] = {
                "mean_ms": statistics.mean(times_for_chunk) * 1000,
                "p50_ms": statistics.median(times_for_chunk) * 1000,
            }

        row = {
            "target_tokens": target_tokens,
            "actual_tokens": actual_token_count,
            "text_chars": len(text),
            "encode_mean_ms": statistics.mean(encode_times) * 1000,
            "encode_p50_ms": statistics.median(encode_times) * 1000,
            "encode_p95_ms": sorted(encode_times)[int(0.95 * NUM_ITERS)] * 1000,
            "encode_min_ms": min(encode_times) * 1000,
            "encode_max_ms": max(encode_times) * 1000,
            "encode_throughput_tok_per_s": actual_token_count / statistics.mean(encode_times),
            "decode_mean_ms": statistics.mean(decode_times) * 1000,
            "decode_p50_ms": statistics.median(decode_times) * 1000,
            "decode_p95_ms": sorted(decode_times)[int(0.95 * NUM_ITERS)] * 1000,
            "decode_throughput_tok_per_s": actual_token_count / statistics.mean(decode_times),
            "incremental_decode": incr_results,
        }
        results.append(row)

        print(f"\n--- {actual_token_count:,} tokens ({len(text):,} chars) ---")
        print(f"  Encode: {row['encode_mean_ms']:.2f} ms mean "
              f"({row['encode_throughput_tok_per_s']:,.0f} tok/s)")
        print(f"  Decode: {row['decode_mean_ms']:.2f} ms mean "
              f"({row['decode_throughput_tok_per_s']:,.0f} tok/s)")
        for cs, ir in incr_results.items():
            print(f"  Incr decode (chunk={cs}): {ir['mean_ms']:.2f} ms for 128 output tokens")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "tokenizer_benchmark.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_PATH,
            "tokenizer_load_time_s": load_time,
            "vocab_size": tokenizer.vocab_size,
            "num_warmup": NUM_WARMUP,
            "num_iters": NUM_ITERS,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Tokens':>10} | {'Encode (ms)':>12} | {'Encode tok/s':>14} | "
          f"{'Decode (ms)':>12} | {'Decode tok/s':>14}")
    print("-" * 90)
    for r in results:
        print(f"{r['actual_tokens']:>10,} | {r['encode_mean_ms']:>12.2f} | "
              f"{r['encode_throughput_tok_per_s']:>14,.0f} | "
              f"{r['decode_mean_ms']:>12.2f} | "
              f"{r['decode_throughput_tok_per_s']:>14,.0f}")
    print("=" * 90)


if __name__ == "__main__":
    benchmark_tokenizer()
