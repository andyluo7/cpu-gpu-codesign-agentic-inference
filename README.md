# CPU-GPU Co-Design for Agentic LLM Inference

Quantifying where time goes in end-to-end agentic LLM inference — and why your CPU might be stealing 15% and more of your GPU throughput.

**📝 Blog post:** [andyluo7.github.io](https://andyluo7.github.io/llm/amd/mi300x/vllm/lmcache/performance/2026/05/14/cpu-gpu-codesign-agentic-inference-mi300x/)

## Key Findings

- **At low concurrency, CPU overhead is negligible** — 0.4–0.6% of E2E time
- **At high concurrency (32 users), CPU overhead reaches 11–15%** of E2E time
- **Scheduling + queue wait = 94% of CPU overhead**, not tokenization
- **Tokenization at 100k tokens costs only 220ms** (~500k tok/s), tiny vs GPU prefill
- **LMCache adds zero measurable CPU overhead** vs HBM prefix cache

## Hardware & Setup

- 2× AMD Instinct MI300X (192 GB HBM3 each)
- MiniMax-M2.5 FP8 MoE, TP=2
- vLLM 0.19.0 (ROCm) + LMCache (source-built for HIP)
- 739 anonymized Claude Code agentic conversation traces

## Structure

```
├── BLOG.md                          # Full blog post (markdown)
├── PLAN.md                          # Original experiment plan
├── RESULTS_SUMMARY.md               # Raw results summary
├── scripts/
│   ├── 01_tokenizer_benchmark.py    # Tokenizer CPU micro-benchmark
│   ├── 02_request_profiler.py       # E2E request time decomposition
│   ├── 03_gpu_monitor.sh            # GPU utilization sampling
│   ├── 04_vllm_internal_profiler.py # vLLM internal component profiler
│   ├── 05_load_generator.py         # Concurrent load generator
│   ├── 06_server_decomposition.py   # HBM-PC server-side decomposition
│   └── 07_lmcache_decomposition.py  # LMCache server-side decomposition
└── results/
    ├── phase1_request_profiler.json
    ├── phase2_tokenizer_benchmark.json
    ├── phase2_cpu_component_benchmark.json
    ├── phase2-deepdive/
    │   └── server_decomposition.json     # HBM prefix cache arm
    ├── phase2-deepdive-lmcache/
    │   └── server_decomposition.json     # LMCache DRAM arm
    └── phase3/
        ├── load_test_results.json
        └── cpu_flame_worker_tp0.svg
```

## Related Work

This analysis accompanies our LMCache multi-turn agentic benchmark on MI300X, which compared KV-cache strategies (no cache, HBM prefix cache, LMCache CPU DRAM) using the same hardware and workload traces.

## License

MIT
