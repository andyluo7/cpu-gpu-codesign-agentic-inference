# CPU-GPU Co-Design Analysis — Raw Results Summary

**Date:** 2026-05-14
**Hardware:** 2× AMD MI300X (192 GB HBM each), ENC1-CLS01-SVR08
**Model:** MiniMax-M2.5 FP8, TP=2
**Framework:** vLLM 0.19.0 + LMCache (BUILD_WITH_HIP=1)

---

## Phase 2.1: Tokenizer CPU Benchmark

| Tokens | Encode (ms) | Encode tok/s | Decode (ms) | Decode tok/s |
|--------|-------------|-------------|-------------|-------------|
| 679 | 1.18 | 576,506 | 0.15 | 4,530,614 |
| 2,711 | 5.09 | 532,379 | 0.60 | 4,549,240 |
| 5,423 | 10.35 | 523,861 | 1.22 | 4,458,404 |
| 10,840 | 20.46 | 529,718 | 2.54 | 4,259,818 |
| 21,679 | 42.72 | 507,414 | 5.17 | 4,189,299 |
| 43,359 | 87.85 | 493,582 | 10.59 | 4,092,758 |
| 67,745 | 134.90 | 502,188 | 16.98 | 3,989,994 |
| 101,615 | 220.38 | 461,085 | 25.84 | 3,932,452 |

- Tokenization throughput: ~500k tok/s (linear scaling)
- Detokenization: ~4M tok/s (8× faster than encode)
- Incremental detokenize (streaming, 128 output tokens): 0.27ms total

## Phase 2.2: CPU Component Benchmark

| Component | Size | Time | Notes |
|-----------|------|------|-------|
| JSON serialize (request payload) | 1k tokens | 0.05 ms | |
| JSON serialize | 100k tokens | 0.82 ms | |
| SSE chunk parse | per chunk | 1.9 µs | |
| SHA256 hash (cache key) | 1k tokens | 0.007 ms | 1.3 GB/s |
| SHA256 hash | 100k tokens | 0.620 ms | 1.5 GB/s |

## Phase 2 Deep-Dive: Server Decomposition — HBM Prefix Cache

| Scenario | Conc | Ctx | Serialize | HTTP OH | Prefill | Decode | Total | CPU% | GPU% |
|----------|------|-----|-----------|---------|---------|--------|-------|------|------|
| single_1k | 1 | 1K | 0.04ms | 7ms | 41ms | 1,780ms | 1,828ms | 0.4% | 99.6% |
| single_8k | 1 | 8K | 0.10ms | 15ms | 124ms | 3,142ms | 3,282ms | 0.5% | 99.5% |
| single_32k | 1 | 32K | 0.39ms | 47ms | 682ms | 7,736ms | 8,465ms | 0.6% | 99.4% |
| single_100k | 1 | 100K | 1.18ms | 131ms | 3,555ms | 20,792ms | 24,479ms | 0.6% | 99.4% |
| conc4_8k | 4 | 8K | 0.10ms | 53ms | 137ms | 3,101ms | 3,291ms | 1.6% | 98.4% |
| conc16_32k | 16 | 32K | 0.34ms | 555ms | 498ms | 7,832ms | 8,885ms | 6.2% | 93.8% |
| conc32_32k | 32 | 32K | 0.35ms | 1,130ms | 636ms | 7,873ms | 9,639ms | 11.6% | 88.4% |
| conc32_100k | 32 | 100K | 1.34ms | 3,885ms | 2,479ms | 19,591ms | 25,957ms | 14.9% | 85.1% |

## Phase 2 Deep-Dive: Server Decomposition — LMCache DRAM (blog config, gpu-mem=0.78)

| Scenario | Conc | Ctx | Serialize | HTTP OH | Prefill | Decode | Total | CPU% | GPU% |
|----------|------|-----|-----------|---------|---------|--------|-------|------|------|
| single_1k | 1 | 1K | 0.04ms | 7ms | 44ms | 2,653ms | 2,704ms | 0.3% | 99.7% |
| single_8k | 1 | 8K | 0.10ms | 15ms | 178ms | 3,376ms | 3,569ms | 0.4% | 99.6% |
| conc4_8k | 4 | 8K | 0.10ms | 50ms | 121ms | 3,455ms | 3,627ms | 1.4% | 98.6% |
| conc16_32k | 16 | 32K | 0.34ms | 515ms | 1,655ms | 8,063ms | 10,233ms | 5.1% | 94.9% |
| conc32_32k | 32 | 32K | 0.35ms | 1,135ms | 722ms | 8,386ms | 10,243ms | 11.0% | 89.0% |
| conc32_100k | 32 | 100K | 1.34ms | 3,937ms | 28,537ms | 20,769ms | 53,244ms | 9.8% | 90.2% |

## Phase 3: Load Test Results

| Scenario | Count | Mean Time(s) | Mean TTFT(s) | Throughput |
|----------|-------|-------------|-------------|------------|
| conc1_ctx1k | 1 | 94.2 | 92.6 | 0.05 req/s (cold start) |
| conc4_ctx8k | 36 | 3.6 | 0.2 | 1.2 req/s |
| conc16_ctx32k | 64 | 10.5 | 1.7 | 1.6 req/s |
| conc32_ctx32k | 128 | 10.9 | 2.0 | 3.2 req/s |
| conc32_ctx100k | 32 | 74.6 | 51.8 | 0.8 req/s |
