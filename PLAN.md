# CPU-GPU Co-Design Analysis Plan
## Quantifying CPU vs GPU Time in Agentic LLM Inference on MI300X

**Goal:** Measure, decompose, and optimize the CPU vs GPU time split in end-to-end agentic inference workloads. Understand where CPU becomes the bottleneck and how CPU-GPU co-design can improve E2E performance.

**Hardware:** 2× AMD MI300X (192 GB HBM each), ENC1-CLS01-SVR08
**Model:** MiniMax-M2.5 FP8, TP=2
**Framework:** vLLM 0.19.0 + LMCache (ROCm)
**Workload:** 739 Claude Code agentic traces (kv-cache-tester)

---

## Motivation

Our LMCache blog focused exclusively on KV-cache strategy impact on TTFT and throughput. But agentic workloads involve substantial **CPU-side work** that we never measured:

```
                    E2E Request Latency
    ┌──────────────────────────────────────────────┐
    │                                              │
    │  ┌─────────┐  ┌──────┐  ┌──────┐  ┌──────┐  │
    │  │Tokenize │→ │Sched │→ │Prefill│→ │Decode│  │  ← GPU-heavy
    │  │(CPU)    │  │(CPU) │  │(GPU)  │  │(GPU) │  │
    │  └─────────┘  └──────┘  └──────┘  └──────┘  │
    │                                              │
    │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
    │  │Tool-call│→ │JSON parse│→ │HTTP/SSE    │  │  ← CPU-heavy
    │  │ parse   │  │/validate │  │serialization│  │
    │  │(CPU)    │  │(CPU)     │  │(CPU)       │  │
    │  └─────────┘  └──────────┘  └────────────┘  │
    └──────────────────────────────────────────────┘
```

**Hypothesis:** At high concurrency with large agentic contexts (50-150k tokens), CPU-side operations (tokenization, scheduling, KV-cache management, tool-call parsing, HTTP serialization) may consume 20-40% of E2E latency — and optimizing them is a free throughput win that doesn't require GPU changes.

---

## Phase 1: Baseline E2E Time Decomposition (Macro)

**Objective:** Break down where time goes at a coarse level across the full request lifecycle.

### 1.1 Client-Side Instrumentation

Instrument `trace_replay_tester.py` to measure fine-grained timing at the HTTP client level:

| Timer | What it measures | How |
|-------|-----------------|-----|
| `t_request_build` | Time to construct the OpenAI API request JSON (conversation history serialization) | `time.perf_counter()` around request construction |
| `t_network_to_first_byte` | HTTP RTT + server processing until first SSE byte | Time from `aiohttp.post()` to first chunk |
| `t_ttft` | Time to first token (already measured) | Existing metric |
| `t_streaming` | Token streaming duration | First token to last token |
| `t_response_parse` | Time to parse the SSE response, extract tool calls, validate JSON | After last byte to result object ready |
| `t_inter_turn` | Time between turns in a multi-turn conversation | End of turn N to start of turn N+1 |

**Key insight:** `t_inter_turn` captures the client-side "thinking" time in an agentic loop — tool execution, context assembly, decision-making. In production (Claude Code, Cursor), this is where the agent framework runs.

### 1.2 Server-Side Instrumentation (vLLM internals)

Use vLLM's built-in torch profiler + custom timing hooks:

| Timer | Where in vLLM | What it measures |
|-------|--------------|-----------------|
| `t_tokenize` | `AsyncLLM.generate()` → tokenizer call | Tokenization of input prompt |
| `t_detokenize` | Detokenizer process | Output token → text conversion |
| `t_schedule` | `Scheduler.schedule()` | Scheduling decision (which requests to batch) |
| `t_prefill_gpu` | GPU kernel time during prefill step | Actual GPU computation |
| `t_decode_gpu` | GPU kernel time during decode step | Per-step GPU decode |
| `t_kv_cache_ops` | KV cache block management | Allocate/free/copy/swap KV blocks |
| `t_lmcache_lookup` | LMCacheConnectorV1 lookup | Hash + search CPU DRAM cache |
| `t_lmcache_transfer` | LMCacheConnectorV1 load/store | DMA between CPU DRAM ↔ GPU HBM |
| `t_sampling` | Sampler | Logit processing + sampling |
| `t_tool_parse` | Tool call parser (minimax_m2) | Parsing structured output into tool calls |

### 1.3 Profiling Method

**Approach A — Torch Profiler (primary):**
```bash
vllm serve ... \
  --profiler-config.profiler=torch \
  --profiler-config.torch_profiler_dir=/work/profiles/phase1 \
  --profiler-config.torch_profiler_with_stack=true \
  --profiler-config.delay_iterations=5 \
  --profiler-config.active_iterations=20
```
Then trigger profiling via `/start_profile` API during a trace replay run. Produces Chrome trace JSON with CPU+GPU timeline.

**Approach B — py-spy (CPU flame graph):**
```bash
py-spy record -o /work/profiles/cpu_flame.svg --pid <vllm_pid> --duration 120 --rate 100
```
Captures where CPU time goes across all Python threads (tokenizer, scheduler, HTTP handler, LMCache connector).

**Approach C — rocprofv3 (GPU kernel-level):**
```bash
rocprofv3 --hip-trace --hsa-trace -o /work/profiles/gpu_trace -- python3 -m vllm.entrypoints.openai.api_server ...
```
Maps exact GPU kernel durations, queue times, and memory transfers. **Note:** requires exclusive GPU access (no other workloads).

**Approach D — Custom Python timing hooks:**
Instrument key vLLM functions with `time.perf_counter()` and `torch.cuda.Event` pairs for precise CPU vs GPU decomposition without profiler overhead.

### 1.4 Test Matrix

| Scenario | Concurrency | Context | Purpose |
|----------|------------|---------|---------|
| S1: Single-request | 1 | 1k/8k/32k/100k | Isolate per-request CPU overhead |
| S2: Low concurrency | 4 | 32k | Match Phase 3 base load |
| S3: Medium concurrency | 16 | 64k | Transition regime |
| S4: High concurrency | 32 | 100k | Match Phase 3 stress |

For each scenario, measure all configs: vanilla, HBM-PC, LMCache.

---

## Phase 2: CPU Time Deep-Dive

**Objective:** Precisely quantify each CPU-side component and identify bottlenecks.

### 2.1 Tokenization

- **What:** BPE tokenization of input prompts (including system prompt, conversation history, tool outputs)
- **Tool:** Benchmark tokenizer standalone (`transformers` tokenizer on CPU)
- **Measurement:**
  ```python
  for ctx_len in [1k, 8k, 32k, 64k, 100k, 150k]:
      t0 = time.perf_counter()
      tokens = tokenizer.encode(prompt[:ctx_len])
      t1 = time.perf_counter()
      # Also measure batch tokenization if applicable
  ```
- **Scaling question:** Does tokenization time scale linearly with input length? At 100k tokens, is it 1ms or 100ms?
- **CPU affinity test:** Pin tokenizer to specific NUMA node vs letting it float

### 2.2 Detokenization

- **What:** Converting output token IDs back to text, including incremental streaming
- **Measurement:** Instrument `Detokenizer.process()` 
- **Scaling:** Detokenization runs on every decode step — at 1000+ concurrent decode steps, does it become a bottleneck?

### 2.3 Scheduling Overhead

- **What:** `Scheduler.schedule()` decides which requests to prefill/decode in each step
- **Measurement:** Time per scheduling decision, and how it scales with:
  - Number of waiting requests
  - Number of running requests  
  - KV block table complexity (with prefix caching, the block allocation tree grows)
- **Hypothesis:** Prefix cache eviction decisions may be expensive at high block counts

### 2.4 KV Cache Management (CPU-side)

- **What:** Block allocation, hash computation for prefix matching, eviction decisions
- **LMCache-specific:** Cache key computation (PYTHONHASHSEED-dependent), CPU DRAM lookup, DMA scheduling
- **Measurement:**
  - Time per hash computation (per request)
  - Time per cache lookup (LMCache hit vs miss)
  - Time per DMA transfer (CPU DRAM → GPU HBM)
  - Total CPU time in KV connector per step

### 2.5 Tool Call Parsing

- **What:** `minimax_m2` tool-call parser extracts structured function calls from model output
- **Measurement:** Time to parse tool-call JSON from streaming output
- **At scale:** With 32 concurrent users each generating tool calls, does parsing serialize?

### 2.6 HTTP/SSE Serialization

- **What:** FastAPI/uvicorn request parsing, SSE event formation, response streaming
- **Measurement:** Time in `aiohttp` + `uvicorn` overhead vs actual model work
- **At scale:** uvicorn worker count, asyncio event loop saturation

---

## Phase 3: GPU Utilization Analysis

**Objective:** Measure GPU idle time and identify CPU-induced GPU starvation.

### 3.1 GPU Activity Ratio

- **Tool:** `rocm-smi --showuse -d 0,1` sampled at 100ms intervals during benchmark
- **Metric:** % time GPU is actively executing kernels vs idle/waiting
- **Expected:** At high concurrency, GPU utilization should be 90%+. If it drops below 80%, CPU is the bottleneck.

### 3.2 GPU Kernel Gap Analysis

- **Tool:** `rocprofv3 --hip-trace` or torch profiler GPU timeline
- **Metric:** Time gaps between consecutive GPU kernel launches
- **These gaps = CPU overhead** (scheduling, data prep, kernel launch overhead)
- Categorize gaps:
  - < 10μs: kernel launch overhead (unavoidable)
  - 10-100μs: CPU dispatch latency (optimizable)
  - 100μs-1ms: scheduling/batching decisions
  - > 1ms: CPU bottleneck (tokenization, cache ops, etc.)

### 3.3 PCIe/Infinity Fabric Transfer Time

- **What:** CPU↔GPU data transfers (input IDs, KV cache DMA for LMCache, sampling results)
- **Tool:** `rocprofv3 --hsa-trace` for memory copy events
- **Metric:** Transfer time as % of total step time

---

## Phase 4: CPU-GPU Co-Design Optimization Experiments

**Objective:** Test specific optimizations and measure their E2E impact.

### 4.1 Tokenizer Optimization

| Experiment | What | Expected Impact |
|-----------|------|----------------|
| Rust tokenizer (tokenizers lib) vs Python | Replace Python BPE with Rust-based tokenizers | 2-5× faster tokenization |
| Pre-tokenized cache | Cache tokenized prefixes (system prompt + tool defs) | Eliminate redundant tokenization of ~12k shared tokens |
| Batch tokenization | Tokenize multiple requests in parallel | Better CPU utilization |
| NUMA-pinned tokenizer | Pin tokenizer threads to CPU cores near GPU's NUMA node | Reduce memory access latency |

### 4.2 Scheduling Optimization

| Experiment | What | Expected Impact |
|-----------|------|----------------|
| Chunked scheduling | Amortize scheduling cost across multiple steps | Reduce per-step overhead |
| Async block allocation | Pre-allocate KV blocks before scheduling | Remove allocation from critical path |
| Simplified prefix tree | Reduce O(n) tree walks for prefix matching | Lower scheduling time at high block counts |

### 4.3 LMCache CPU-Side Optimization

| Experiment | What | Expected Impact |
|-----------|------|----------------|
| Async DMA overlap | Overlap CPU DRAM→HBM transfer with GPU compute | Hide transfer latency |
| Parallel hash computation | Multi-threaded cache key hashing | Faster cache lookups |
| Larger chunk size (512/1024) | Reduce number of cache lookups per request | Less CPU overhead per lookup |
| `LMCACHE_CHUNK_SIZE` sweep | 64/128/256/512/1024 | Find optimal CPU-overhead vs cache-granularity trade-off |

### 4.4 System-Level Optimizations

| Experiment | What | Expected Impact |
|-----------|------|----------------|
| CPU frequency pinning | `cpupower frequency-set -g performance` | Eliminate frequency scaling overhead |
| NUMA affinity | Pin vLLM workers to NUMA node 0 (near GPU 0-3) | Reduce cross-NUMA traffic |
| uvicorn worker scaling | 1 vs 2 vs 4 uvicorn workers | Parallelize HTTP handling |
| Python GIL alternatives | Test with `nogil` build or multiprocess detokenizer | Remove GIL contention |

### 4.5 E2E Agentic Loop Optimization

| Experiment | What | Expected Impact |
|-----------|------|----------------|
| Parallel tool execution | Execute tool calls concurrently (client-side) | Reduce inter-turn latency |
| Incremental context | Send only delta tokens (not full history) each turn | Reduce network + tokenization time |
| Predictive prefetching | Pre-warm likely next prompts based on tool call patterns | Hide prefill latency |

---

## Phase 5: Comprehensive Report

### Deliverables

1. **Time decomposition waterfall chart** — stacked bar per scenario showing:
   - Tokenization (CPU)
   - Scheduling (CPU)
   - KV cache ops (CPU + DMA)
   - Prefill (GPU)
   - Decode (GPU)
   - Detokenization (CPU)
   - Tool-call parsing (CPU)
   - HTTP serialization (CPU)
   - Inter-turn gap (client CPU)

2. **CPU vs GPU time ratio table** — by scenario (concurrency × context length)

3. **GPU utilization heat map** — GPU activity % over time, annotated with CPU bottleneck events

4. **Optimization impact matrix** — each Phase 4 experiment's measured impact on E2E latency

5. **Co-design recommendations** — ranked list of optimizations with effort/impact estimates

### Report Format

Blog-style markdown (consistent with LMCache blog), with:
- Charts (matplotlib/plotly, saved as PNG)
- Tables with raw numbers
- Reproduction scripts
- Specific AMD MI300X / ROCm tuning recommendations

---

## Execution Timeline

| Week | Phase | Work |
|------|-------|------|
| 1 | Phase 1 | Instrument client + server, run 4 scenarios × 3 configs = 12 runs |
| 1-2 | Phase 2 | Deep-dive CPU components (standalone micro-benchmarks) |
| 2 | Phase 3 | GPU utilization analysis (rocprofv3 + torch profiler traces) |
| 2-3 | Phase 4 | Optimization experiments (top 4-6 from Phase 2/3 findings) |
| 3 | Phase 5 | Analysis, charts, report write-up |

---

## Success Criteria

- **Quantitative:** CPU vs GPU time split measured to ±5% accuracy across all scenarios
- **Actionable:** ≥3 concrete optimizations identified with measured E2E improvement
- **Publishable:** Blog-quality report with reproduction steps

---

## Open Questions

1. Does vLLM's V1 engine (async) change the CPU bottleneck profile vs V0?
2. How does the CPU overhead scale with TP degree? (TP=2 vs TP=4 vs TP=8)
3. Is the Python GIL a measurable bottleneck at 32+ concurrent users?
4. Should we test SGLang's scheduling path for comparison (different CPU design)?
5. What's the CPU overhead of Anthropic-style `cache_control` markers vs automatic prefix detection?
