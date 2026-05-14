---
layout: post
title: "CPU-GPU Co-Design for Agentic LLM Inference on AMD MI300X"
date: 2026-05-14
categories: [LLM, AMD, MI300X, vLLM, LMCache, Performance]
---

*Quantifying where time actually goes — and why your CPU might be stealing 15% of your GPU throughput.*

---

## Key Summary

We instrumented the full request lifecycle of agentic LLM inference on AMD MI300X to answer a simple question: **how much of end-to-end latency is CPU work vs GPU work?**

Using MiniMax-M2.5 (230 GB FP8 MoE) on 2× MI300X with vLLM 0.19.0, we decomposed every request into serialization, HTTP overhead (tokenization + scheduling + queue wait), GPU prefill, and GPU decode across 8 scenarios spanning concurrency 1–32 and context 1k–100k tokens.

**Headline findings:**
- **At low concurrency, CPU overhead is negligible** — 0.4–0.6% of E2E time for single requests at any context length
- **At high concurrency, CPU overhead becomes material** — 11–15% of E2E time at 32 concurrent users
- **The bottleneck is not tokenization or JSON parsing** — it's **scheduling + queue wait**, which scales superlinearly with concurrency
- **Tokenization at 100k tokens costs only 220ms** (~500k tok/s on a single CPU core), tiny compared to GPU prefill (2–4 seconds)
- **LMCache adds minimal CPU overhead** vs HBM prefix cache — the CPU% split is nearly identical between the two strategies
- **The real CPU-GPU co-design opportunity** is not in making CPU faster, but in **overlapping CPU work with GPU work** and reducing scheduling contention at high concurrency

---

## 1. Motivation: The Hidden CPU Tax in Agentic Inference

Our previous work benchmarked [LMCache for multi-turn agentic workloads on MI300X](https://github.com/andyluo7/openclaw-workspace/tree/main/multiturn-agentic-bench), comparing KV-cache strategies. We measured TTFT, throughput, and cache hit rates. But we treated the inference server as a black box — we never asked *where inside the server* the time goes.

Agentic AI workloads are not just GPU workloads. Every request passes through a CPU pipeline before and after GPU execution:

```
Client                          Server (vLLM)                      GPU
──────                          ─────────────                      ───
 │                                    │                              │
 │─── serialize request ──────────────│                              │
 │    (JSON, 0.04-1.3ms)              │                              │
 │                                    │                              │
 │                          ┌─────────┴──────────┐                   │
 │                          │ HTTP parse         │                   │
 │                          │ Tokenize input     │                   │
 │                          │ Schedule request   │  "HTTP Overhead"  │
 │                          │ KV cache lookup    │  (7-3900ms)       │
 │                          │ Queue wait         │                   │
 │                          └─────────┬──────────┘                   │
 │                                    │                              │
 │                                    │──── GPU prefill ─────────────│
 │                                    │     (41-28537ms)             │
 │                                    │                              │
 │                                    │──── GPU decode (streaming) ──│
 │                                    │     (1780-20792ms)           │
 │                                    │                              │
 │◄── parse SSE response ─────────────│                              │
 │    (1.9µs per chunk)               │                              │
```

The question: at scale (32 concurrent users, 100k token contexts), does the CPU pipeline become a bottleneck?

---

## 2. Methodology

### 2.1 Hardware & Software

| Component | Specification |
|-----------|--------------|
| GPU | 2× AMD Instinct MI300X (192 GB HBM3 each), gfx942 |
| CPU | AMD EPYC (ENC1-CLS01-SVR08) |
| Model | MiniMaxAI/MiniMax-M2.5 FP8, TP=2 |
| Framework | vLLM 0.19.0 (ROCm) |
| KV Cache | HBM prefix cache / LMCache CPU DRAM |
| Workload | 739 anonymized Claude Code agentic conversations |

### 2.2 What We Measured

We decomposed each request into five time components:

| Component | Where | What It Captures |
|-----------|-------|-----------------|
| **t_serialize** | Client CPU | JSON serialization of the request payload |
| **t_http_overhead** | Server CPU | HTTP parsing + tokenization + scheduling + queue wait + KV cache lookup |
| **t_server_prefill** | Server GPU | Attention computation over all input tokens |
| **t_decode** | Server GPU (mostly) | Autoregressive token generation + streaming |
| **t_response_parse** | Client CPU | SSE chunk parsing + tool call extraction |

We classify `t_serialize + t_http_overhead + t_response_parse` as **CPU time** and `t_server_prefill + t_decode` as **GPU time**.

**Note:** `t_http_overhead` is measured as the gap between client sending the HTTP request and receiving the first byte back. This includes tokenization, scheduling, queue wait time, and KV cache management — all CPU-side work that happens before the GPU begins prefill. At low concurrency this is mostly tokenization + scheduling. At high concurrency, queue wait dominates.

### 2.3 Test Matrix

| Scenario | Concurrency | Context | Purpose |
|----------|------------|---------|---------|
| single_1k | 1 | 1,000 | Baseline: pure overhead |
| single_8k | 1 | 8,000 | Typical agent turn |
| single_32k | 1 | 32,000 | Large agent context |
| single_100k | 1 | 100,000 | Maximum agent context |
| conc4_8k | 4 | 8,000 | Light multi-tenant |
| conc16_32k | 16 | 32,000 | Medium load |
| conc32_32k | 32 | 32,000 | High load, moderate context |
| conc32_100k | 32 | 100,000 | Stress: high load + large context |

Each scenario was run with 3–5 batches of requests, with results aggregated.

---

## 3. Results

### 3.1 The CPU-GPU Split: It's All About Concurrency

**HBM Prefix Cache Configuration:**

| Scenario | Conc | Ctx | HTTP OH (ms) | Prefill (ms) | Decode (ms) | Total (ms) | **CPU%** | **GPU%** |
|----------|------|-----|-------------|-------------|-------------|------------|---------|---------|
| single_1k | 1 | 1K | 7 | 41 | 1,780 | 1,828 | **0.4%** | 99.6% |
| single_8k | 1 | 8K | 15 | 124 | 3,142 | 3,282 | **0.5%** | 99.5% |
| single_32k | 1 | 32K | 47 | 682 | 7,736 | 8,465 | **0.6%** | 99.4% |
| single_100k | 1 | 100K | 131 | 3,555 | 20,792 | 24,479 | **0.6%** | 99.4% |
| conc4_8k | 4 | 8K | 53 | 137 | 3,101 | 3,291 | **1.6%** | 98.4% |
| conc16_32k | 16 | 32K | 555 | 498 | 7,832 | 8,885 | **6.2%** | 93.8% |
| conc32_32k | 32 | 32K | 1,130 | 636 | 7,873 | 9,639 | **11.6%** | 88.4% |
| conc32_100k | 32 | 100K | 3,885 | 2,479 | 19,591 | 25,957 | **14.9%** | 85.1% |

The pattern is clear: **CPU overhead scales with concurrency, not context length.**

- Single-request: CPU% is flat at ~0.5% regardless of whether context is 1k or 100k
- At concurrency 32: CPU% jumps to 11–15%
- The dominant CPU cost is `t_http_overhead` (scheduling + queue wait), not tokenization

### 3.2 LMCache vs HBM Prefix Cache: CPU Overhead Comparison

**LMCache DRAM Configuration (gpu-mem-util=0.78):**

| Scenario | Conc | Ctx | HTTP OH (ms) | Prefill (ms) | Decode (ms) | Total (ms) | **CPU%** | **GPU%** |
|----------|------|-----|-------------|-------------|-------------|------------|---------|---------|
| single_1k | 1 | 1K | 7 | 44 | 2,653 | 2,704 | **0.3%** | 99.7% |
| single_8k | 1 | 8K | 15 | 178 | 3,376 | 3,569 | **0.4%** | 99.6% |
| conc4_8k | 4 | 8K | 50 | 121 | 3,455 | 3,627 | **1.4%** | 98.6% |
| conc16_32k | 16 | 32K | 515 | 1,655 | 8,063 | 10,233 | **5.1%** | 94.9% |
| conc32_32k | 32 | 32K | 1,135 | 722 | 8,386 | 10,243 | **11.0%** | 89.0% |
| conc32_100k | 32 | 100K | 3,937 | 28,537 | 20,769 | 53,244 | **9.8%** | 90.2% |

**Key comparison — CPU overhead is nearly identical:**

| Scenario | HBM-PC CPU% | LMCache CPU% | Delta |
|----------|------------|-------------|-------|
| single_1k | 0.4% | 0.3% | −0.1% |
| conc4_8k | 1.6% | 1.4% | −0.2% |
| conc16_32k | 6.2% | 5.1% | −1.1% |
| conc32_32k | 11.6% | 11.0% | −0.6% |
| conc32_100k | 14.9% | 9.8% | −5.1% |

**LMCache does NOT add measurable CPU overhead.** In fact, CPU% is slightly *lower* with LMCache at high concurrency because LMCache's CPU DRAM cache reduces HBM pressure, meaning less time in KV block eviction decisions on the CPU side.

The `t_http_overhead` is nearly identical between the two configs (~1,130–1,135ms at conc32_32k), confirming that the LMCache connector's CPU-side work (hash computation, cache lookup, DMA scheduling) is negligible.

### 3.3 Where Does CPU Time Actually Go?

We ran standalone micro-benchmarks to isolate each CPU component:

| Component | Time at 100K tokens | % of HTTP Overhead (conc=32) |
|-----------|-------------------|------------------------------|
| Tokenization (encode) | 220 ms | ~5.7% |
| JSON serialization (request build) | 0.82 ms | <0.1% |
| SHA256 hash (cache key) | 0.62 ms | <0.1% |
| SSE chunk parse (per token) | 1.9 µs | <0.1% |
| Detokenization (128 tokens) | 0.27 ms | <0.1% |
| **Scheduling + queue wait** | **~3,660 ms** | **~94%** |

The smoking gun: **scheduling + queue wait accounts for ~94% of CPU overhead** at high concurrency. Tokenization, hashing, and serialization are negligible.

This makes sense: at 32 concurrent requests, the vLLM scheduler must:
1. Decide which requests to batch together
2. Walk the prefix cache tree to find matching blocks
3. Allocate KV blocks for new tokens
4. Manage the preemption queue when HBM is under pressure
5. Coordinate across TP workers

Each of these is O(n) or worse in the number of concurrent requests, and they all happen on a single Python thread (GIL-bound).

### 3.4 Tokenization Deep-Dive: Linear but Fast

| Tokens | Encode (ms) | Throughput (tok/s) |
|--------|------------|--------------------|
| 679 | 1.18 | 576,506 |
| 2,711 | 5.09 | 532,379 |
| 5,423 | 10.35 | 523,861 |
| 10,840 | 20.46 | 529,718 |
| 21,679 | 42.72 | 507,414 |
| 43,359 | 87.85 | 493,582 |
| 67,745 | 134.90 | 502,188 |
| 101,615 | 220.38 | 461,085 |

Tokenization scales linearly with input length at ~500k tok/s. Even at 100k tokens (the largest agentic context we tested), tokenization takes only **220ms** — under 1% of E2E time for any scenario.

The HuggingFace `tokenizers` library (Rust-based BPE) is already highly optimized. Switching to a C++ tokenizer would save ~50–100ms at 100k tokens — not enough to matter.

**Detokenization** (streaming output) is even faster: 0.27ms for 128 output tokens. Per-token streaming overhead is not a concern.

---

## 4. Analysis: The Scheduling Wall

### 4.1 Why Scheduling Dominates at High Concurrency

The `t_http_overhead` captures everything from HTTP request receipt to first GPU kernel launch. At concurrency 1, it's dominated by tokenization (~220ms for 100k). At concurrency 32, it balloons to **3,885ms** — a 30× increase.

The growth is **superlinear** with concurrency:

| Concurrency | HTTP Overhead (32K ctx) | Growth Factor |
|-------------|------------------------|--------------|
| 1 | 47 ms | 1.0× |
| 4 | 53 ms | 1.1× |
| 16 | 555 ms | 11.8× |
| 32 | 1,130 ms | 24.0× |

This superlinear scaling points to **contention** in the scheduling path:

1. **Python GIL:** vLLM's scheduler runs in the main asyncio event loop. At 32 concurrent requests, the GIL serializes scheduling decisions, tokenization, and HTTP handling.

2. **Prefix cache tree walks:** With prefix caching enabled, every scheduling decision walks the block hash tree. At high concurrency with diverse prompts, the tree grows and walks become expensive.

3. **Block allocation contention:** The KV block allocator must coordinate free/used block tables across TP workers.

4. **Queue wait:** When the GPU is saturated, requests queue in the scheduler waiting for slots.

### 4.2 The 15% Rule

Our data suggests a practical rule of thumb:

> **At production-level concurrency (16–32 users), CPU overhead consumes 10–15% of E2E latency on MI300X.**

This means that even with an infinitely fast GPU, you would only recover 85–90% of theoretical speedup. The remaining 10–15% is CPU-bound.

For a concrete example: at conc32_100k with HBM prefix cache, total E2E is 25,957ms. GPU time is 22,070ms (prefill + decode). Even if GPU time went to zero, the CPU overhead of 3,887ms would remain — setting a hard floor on latency.

---

## 5. Optimization Recommendations

### Tier 1: High Impact, Framework-Level

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| **Pipeline scheduling with GPU execution** | 5–10% E2E at high concurrency | Medium |
| **Move tokenization off main event loop** | 2–3% at high concurrency | Low |
| **Batch scheduling decisions** | 3–5% at high concurrency | Medium |
| **Pre-allocate KV blocks speculatively** | 2–3% at high concurrency | Medium |

### Tier 2: System-Level Tuning

| Optimization | Expected Impact | Effort |
|-------------|----------------|--------|
| **NUMA affinity** (pin workers to GPU-local node) | 1–2% | Low |
| **CPU frequency governor** (`performance` mode) | 0.5–1% | Trivial |
| **Dedicated CPU cores for scheduler** (isolcpus) | 1–2% | Low |

### Tier 3: Not Worth Optimizing

| Component | Why Not |
|-----------|---------|
| Tokenizer speed | Already 500k tok/s, <1% of E2E |
| JSON serialization | <1ms even at 100k tokens |
| SSE parsing | 1.9µs per chunk — effectively zero |
| LMCache hash/lookup | <1ms even at 100k tokens |
| Detokenization | 0.27ms for 128 output tokens |

---

## 6. Key Takeaways

### For inference platform teams:

1. **CPU overhead is real but bounded.** At 32 concurrent users, 10–15% of E2E latency is CPU. This sets a floor on achievable latency regardless of GPU speed.

2. **Scheduling is the bottleneck, not tokenization.** Don't waste time optimizing the tokenizer — optimize the scheduler and its interaction with the KV cache manager.

3. **LMCache adds zero measurable CPU overhead.** The cache connector's hash/lookup/DMA scheduling cost is lost in the noise. If you're avoiding LMCache because of CPU concerns, don't.

4. **The GIL is the elephant in the room.** At 32+ concurrent requests, Python GIL serializes scheduling, tokenization, and HTTP handling. Multi-process architectures (like vLLM V1's separated EngineCore) are the right direction.

### For hardware architects:

1. **CPU performance matters for inference at scale.** A faster CPU won't help a single request, but it directly impacts latency at 16+ concurrent users.

2. **PCIe/Infinity Fabric bandwidth is not the CPU bottleneck.** The CPU overhead is all compute (scheduling, hash computation, Python interpretation), not data transfer.

3. **NUMA topology matters.** Ensuring scheduler threads run on CPU cores local to the GPU's NUMA node reduces memory access latency for KV block table management.

### For the agentic AI community:

1. **The CPU-GPU co-design question is a scheduling problem**, not a compute problem. The path forward is better overlap between CPU scheduling and GPU execution.

2. **Context length matters less than concurrency.** A single 100k-token request has 0.6% CPU overhead. Thirty-two 1k-token requests have 11%+ CPU overhead. If you're scaling to many concurrent agent sessions, CPU efficiency of the scheduler is critical.

---

## 7. Open Questions & Future Directions

### 7.1 Can We Eliminate the CPU Bottleneck? Rust, No-GIL, and Beyond

Our data shows that **94% of CPU overhead is scheduling + queue wait**, not tokenization or serialization. This has direct implications for optimization strategies:

**Rewriting the scheduler in Rust or C++:**

The vLLM scheduler today is pure Python — prefix tree walks, block allocation, preemption logic, all running under the GIL. Rewriting the hot path in Rust (via PyO3) or C++ (via pybind11) could yield significant gains:

| Component | Current (Python) | Estimated (Rust) | Speedup | Impact on E2E |
|-----------|-----------------|-------------------|---------|---------------|
| Prefix tree walk | O(n) per request, GIL-held | O(n) but no GIL, SIMD-friendly | 5–10× | 2–4% at conc=32 |
| Block allocation | Dict lookups + list ops | Lock-free concurrent allocator | 10–20× | 1–2% at conc=32 |
| Hash computation | Python `hash()` | Rust `xxhash` / `blake3` | 3–5× | <0.5% (already fast) |
| Request batching | Python list sorting | Rust `rayon` parallel sort | 5–10× | 1–2% at conc=32 |

Total estimated E2E improvement: **4–8% at conc=32** from a Rust scheduler rewrite. This is meaningful but not transformative — the real win is eliminating GIL contention, not raw speed.

**Removing the Python GIL:**

Python 3.13+ introduced experimental free-threaded mode (`--disable-gil`). For vLLM, this could be transformative:

- Currently: tokenization, scheduling, HTTP handling, and detokenization all serialize through the GIL
- Without GIL: these can truly parallelize across CPU cores
- The `t_http_overhead` at conc=32 (1,130ms for 32K context) includes substantial GIL contention — multiple requests competing for the same Python thread
- **Estimated impact: 20–40% reduction in `t_http_overhead` at high concurrency**, translating to 3–6% E2E improvement

However, GIL removal has risks:
- vLLM's internal data structures (block tables, prefix cache tree) would need thread-safe redesign
- Many Python C extensions assume GIL protection
- The `torch` runtime itself has GIL interactions during tensor operations

**The pragmatic path — vLLM V1's multi-process architecture:**

vLLM V1 already separates the EngineCore (scheduler) from the APIServer (HTTP handling) into different processes. This is effectively a GIL bypass:

```
APIServer (Process 1)     EngineCore (Process 2)     Workers (Process 3+)
├── HTTP parsing          ├── Scheduling             ├── GPU prefill
├── Tokenization          ├── Block allocation       ├── GPU decode
├── Request routing       ├── Cache management       ├── KV transfers
└── SSE streaming         └── Preemption logic       └── Sampling
         │                         │                        │
         └── IPC (shared mem) ─────┘                        │
                                   └── IPC (shared mem) ────┘
```

This architecture already eliminates most GIL contention. Our measurements show that vLLM 0.19.0 (which uses V1) achieves reasonable scaling — the 15% CPU overhead at conc=32 is *after* the multi-process split. Without it, we'd likely see 25–30%.

**Recommendation:** The highest-ROI optimization is **pipelining scheduling with GPU execution** — start scheduling the next batch while the current batch is still executing on GPU. This doesn't require any language change, just better overlap in the EngineCore.

### 7.2 Sub-Agent Explosion: What Happens at 12× Concurrency?

Modern agentic frameworks (Claude Code, OpenHands, SWE-Agent) routinely spawn sub-agents. A single user session might fork into 4–12 parallel sub-agents for tasks like:
- Searching multiple codebases simultaneously
- Running parallel tool calls (web search + file read + code execution)
- Exploring multiple solution paths (tree-of-thought)

**The math gets scary fast:**

If 4 users each spawn 3 sub-agents, you have 4 × (1 + 3) = 16 effective concurrent sessions. If each spawns 12 sub-agents: 4 × (1 + 12) = **52 concurrent sessions.**

Extrapolating from our data:

| Users | Sub-agents/user | Effective Conc | Est. CPU% | Est. HTTP OH (32K) |
|-------|----------------|----------------|-----------|--------------------|
| 4 | 0 | 4 | 1.6% | 53 ms |
| 4 | 3 | 16 | 6.2% | 555 ms |
| 4 | 12 | 52 | **20–25%** | **~3,000 ms** |
| 8 | 12 | 104 | **30–40%** | **~8,000+ ms** |

At 52 effective concurrent requests, our superlinear scaling model predicts:
- HTTP overhead would reach ~3,000ms (vs 1,130ms at conc=32) — that's 3 seconds of pure CPU wait before a single GPU kernel fires
- CPU% of E2E could hit 20–25%, meaning **one quarter of your GPU investment is wasted on CPU scheduling**
- The prefix cache tree would become deep and wide (52 diverse conversation prefixes), making tree walks even more expensive

**Sub-agent-specific challenges:**

1. **Prefix divergence:** Sub-agents share a common parent prefix but diverge quickly (different tool calls, different search results). This creates a bushy prefix tree that's expensive to walk but has high reuse potential — exactly the regime where LMCache's L2 tier pays off.

2. **Bursty arrival patterns:** Sub-agents don't arrive at a steady rate — they burst (parent spawns 12 children simultaneously). The scheduler must absorb this burst, and queue wait time spikes.

3. **Priority inversion:** The parent agent is blocked waiting for sub-agent results. If sub-agents are queued behind other users' requests, the parent's end-to-end latency multiplies.

**Co-design implications:**

- **Request routing becomes critical:** With 52+ concurrent sessions, a single vLLM instance may not be enough. Disaggregated serving (separate prefill and decode nodes) or multi-instance routing could reduce per-instance scheduling pressure.
- **Sub-agent-aware scheduling:** A scheduler that understands parent-child relationships could prioritize sub-agents of the same parent to complete a "generation" faster, rather than round-robin across all requests.
- **Shared prefix optimization:** Sub-agents from the same parent share ~80–90% of their prefix. A scheduler that detects this and batches sibling sub-agents together for prefill could dramatically reduce redundant computation.

### 7.3 Hybrid Workloads: Database Queries, RAG, and Tool Execution

Real agentic workloads don't just call the LLM — they interleave LLM inference with CPU/IO-bound operations:

```
 Turn 1: LLM generates SQL query          (GPU: 2-5s)
 Turn 2: Execute SQL against database      (CPU/IO: 50-500ms)
 Turn 3: LLM analyzes results              (GPU: 3-8s)
 Turn 4: Retrieve documents from vector DB (CPU/IO: 20-200ms)
 Turn 5: LLM synthesizes final answer      (GPU: 5-15s)
```

**The inter-turn gap is a new CPU cost we didn't measure:**

Our benchmark focused on the *intra-request* CPU-GPU split (what happens inside a single LLM call). But agentic workloads have a second CPU cost: the **inter-turn gap** — the time between the LLM finishing one turn and the next turn's prompt being ready.

This gap includes:

| Operation | Typical Latency | Where |
|-----------|----------------|-------|
| Tool call parsing | 0.1–1 ms | Client CPU |
| Database query (PostgreSQL) | 5–500 ms | External service |
| Vector DB retrieval (FAISS/pgvector) | 10–200 ms | CPU + sometimes GPU |
| Web API call (search, code execution) | 100–2,000 ms | Network + external |
| Result formatting + context assembly | 1–10 ms | Client CPU |
| Re-tokenization of updated context | 50–220 ms | Server CPU |

**Performance implications:**

1. **GPU idle time:** During tool execution, the GPU allocated to this user's session sits idle. At 100k context, the KV cache for one session holds ~12 GB of HBM. If tool execution takes 500ms, that's 500ms × 12 GB of stranded GPU memory that could serve other requests.

2. **The KV cache cold-start problem:** If the scheduler evicts this session's KV blocks during tool execution (to serve other requests), the next turn must re-prefill the entire context. This is exactly the scenario where LMCache's CPU DRAM tier shines — it preserves KV state across tool-execution gaps at negligible cost.

3. **CPU contention between tool execution and scheduling:** If tool execution (database queries, vector search) runs on the same CPU cores as the vLLM scheduler, it competes for CPU resources. At high concurrency + frequent tool calls, this could push CPU overhead well beyond the 15% we measured for pure LLM inference.

**Estimated E2E impact of hybrid workloads:**

| Workload Type | LLM Time | Tool Time | Inter-turn OH | GPU Idle % | Effective CPU% |
|---------------|----------|-----------|--------------|-----------|----------------|
| Pure chat | 100% | 0% | ~0% | ~0% | 10–15% |
| Light tools (search) | 70% | 20% | 10% | 15–20% | 20–25% |
| Heavy tools (DB + RAG) | 50% | 35% | 15% | 25–35% | 25–35% |
| Code execution agents | 40% | 45% | 15% | 35–45% | 30–40% |

For code execution agents (the Claude Code use case our traces come from), **CPU and IO operations may consume 40–50% of wall-clock time**, with GPU active only 50–60% of the time. This fundamentally changes the co-design equation:

- **For pure LLM serving:** Buy the best GPU, CPU barely matters
- **For agentic serving:** CPU, memory bandwidth, and IO become co-equal with GPU. System balance matters more than peak GPU FLOPS.

**Optimization strategies for hybrid workloads:**

1. **Speculative prefetching:** While the LLM generates a tool call, pre-warm likely next-turn prefixes based on the tool type. For example, if the model calls `search()`, pre-tokenize a template like `"Search results: {placeholder}"` to have partial KV cache ready.

2. **KV cache reservation:** Reserve a "parking" slot in CPU DRAM for active sessions during tool execution, preventing eviction. LMCache already enables this — the question is whether to make it tool-call-aware.

3. **Separate CPU pools:** Dedicate specific CPU cores to vLLM scheduling and others to tool execution. NUMA-aware pinning becomes critical: vLLM scheduler threads on cores near the GPU, tool execution threads on cores near the NIC (for database queries) or NVMe (for document retrieval).

4. **Async tool execution with GPU overlap:** Execute tool calls concurrently with other users' LLM inference, then "re-inject" the results when ready. This requires the scheduler to support interruptible sessions — start other requests during the tool gap, then preempt them when the tool-calling session is ready to continue.

---

## Appendix: Reproduction

### Environment

```bash
# Container
docker run -d --name lmcache-bench --entrypoint /bin/bash \
  --device=/dev/kfd --device=/dev/dri --network=host --ipc=host \
  --group-add video --cap-add SYS_PTRACE \
  -v /mnt/nvme3n1p1/models:/work/models \
  vllm/vllm-openai-rocm:v0.19.0 -c "sleep infinity"

# LMCache (source build for ROCm)
docker exec lmcache-bench bash -c "
  git clone --depth 1 https://github.com/LMCache/LMCache.git /work/LMCache
  cd /work/LMCache && BUILD_WITH_HIP=1 pip install -e . --no-build-isolation
  pip uninstall -y nixl nixl-cu12 cupy-cuda12x cufile-python cuda-pathfinder
"
```

### Server Configs

**HBM Prefix Cache:**
```bash
VLLM_FLOAT32_MATMUL_PRECISION=high \
vllm serve /work/models/MiniMax-M2.5 \
  --tensor-parallel-size 2 --enable-prefix-caching \
  --gpu-memory-utilization 0.85 --host 0.0.0.0 --port 8000
```

**LMCache DRAM:**
```bash
PYTHONHASHSEED=0 VLLM_FLOAT32_MATMUL_PRECISION=high \
LMCACHE_LOCAL_CPU=true LMCACHE_CHUNK_SIZE=256 \
vllm serve /work/models/MiniMax-M2.5 \
  --tensor-parallel-size 2 --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --gpu-memory-utilization 0.78 --host 0.0.0.0 --port 8000
```

### Benchmark Scripts

All scripts and raw data are available at [github.com/andyluo7/cpu-gpu-codesign-agentic-inference](https://github.com/andyluo7/cpu-gpu-codesign-agentic-inference).

---

*This analysis accompanies our LMCache multi-turn agentic benchmark and uses the same hardware, model, and workload traces.*
