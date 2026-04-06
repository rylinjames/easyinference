# InferScope MCP — Production Target

## Scope

InferScope is a **production-grade MCP server** for NVIDIA Dynamo inference deployments. The only supported configuration:

- **Models:** Kimi K2.5 (400B MoE, 128 experts, GQA), GLM-5 (70B dense, GQA)
- **Engine:** NVIDIA Dynamo 1.0 (vLLM/SGLang as worker backends, not standalone)
- **GPUs:** Hopper H100 SXM, H200 SXM; Blackwell B200, B300
- **Workload:** Long-context coding (131K context default)
- **KV cache:** LMCache + KVBM (KV Block Manager) hierarchical tiering (GPU HBM -> CPU DRAM -> NVMe -> object storage). Note: "Grove" is NVIDIA Dynamo's Kubernetes gang-scheduling component, *not* KV tiering — earlier revisions of this doc conflated the two.
- **Topology:** Aggregated (single worker) or disaggregated (prefill/decode split with NIXL KV transfer)

Target users: Cursor, Anthropic, and neo-cloud providers benchmarking and monitoring Dynamo deployments.

## Project Layout

```
products/
  inferscope/          <- MCP server + CLI (MIT license)
    src/inferscope/
      server.py                  <- FastMCP entry point, tool registration
      server_dynamo.py           <- Dynamo production operations tools
      server_profiling.py        <- Runtime profiling tools
      server_benchmarks.py       <- Benchmark/evaluation tools
      production_target.py       <- Supported model/GPU/engine contract
      errors.py                  <- Structured error responses
      engines/
        dynamo.py                <- DynamoCompiler + DynamoAdapter (PRIMARY)
        vllm.py                  <- vLLM adapter (Dynamo backend only)
        sglang.py                <- SGLang adapter (Dynamo backend only)
        trtllm.py                <- TRT-LLM adapter (kept, not promoted)
        atom.py                  <- ATOM adapter (AMD, kept, not promoted)
        base.py                  <- EngineAdapter/ConfigCompiler ABCs
      tools/
        dynamo_production.py     <- cluster_health, failure_analysis, kv_pipeline, slo_compliance, disagg_topology
        recommend.py             <- Config/engine/parallelism recommendations
        model_intel.py           <- Model profiles, validation, capacity
        hardware_intel.py        <- GPU specs, comparison
        kv_cache.py              <- KV budget, tiering, disaggregation, quantization
        diagnose.py              <- Live deployment diagnostics
        audit.py                 <- Live deployment audit
        live_tuner.py            <- Runtime auto-tuning
        profiling.py             <- Runtime profiling wrapper
      optimization/
        serving_profile.py       <- ServingProfile (central normalized object)
        recommender.py           <- DAG-based recommendation pipeline (6 nodes)
        platform_policy.py       <- Engine support tiers, platform traits
        memory_planner.py        <- Memory breakdown calculation
        validator.py             <- Pre-flight config validation
        checks.py                <- 31 ISA-grounded audit checks
      telemetry/
        prometheus.py            <- Prometheus scraper (vLLM, SGLang, ATOM, Dynamo)
        normalizer.py            <- Cross-engine metric normalization
        capture.py               <- Metric capture helpers
        failure_taxonomy.py      <- Dynamo failure mode classification
        models.py                <- Telemetry data models
      hardware/
        gpu_profiles.py          <- GPU ISA-level specs (all profiles kept, gated by production_target)
      models/
        registry.py              <- Model profiles (all models kept, gated by production_target)
      benchmarks/
        workloads/               <- YAML workload packs (coding-long-context, tool-agent, etc.)
        experiment_specs/        <- YAML experiment configs (dynamo-aggregated-*, dynamo-disagg-*)
      profiling/                 <- Runtime profiling models and tuning
      security.py                <- SSRF protection, input validation
      endpoint_auth.py           <- HTTP auth helpers
      logging.py                 <- Structured logging (structlog, JSON prod, colored dev)
      config.py                  <- Global settings
    tests/                       <- pytest test suite
    docs/
      DEPLOYMENT-GUIDE.md        <- Operator deployment patterns
  isb1/                <- Inference Serving Benchmark Standard 1 (Apache-2.0)
    harness/           <- vLLM server lifecycle, replay execution
    workloads/         <- 4 canonical families: chat, agent, rag, coding
    analysis/          <- Metrics, aggregation, statistics
    quality/           <- HumanEval, MMLU-Pro, ROUGE
    configs/           <- GPU, model, workload YAML configs
```

## Key Architecture Decisions

1. **Dynamo is the only top-level engine.** The recommender DAG auto-selects Dynamo for coding/agent/RAG workloads on Hopper/Blackwell. vLLM and SGLang are kept as Dynamo worker backends.

2. **Scope gating via `production_target.py`, not file deletion.** All GPU profiles, models, and engine adapters remain in the codebase for CLI use. The MCP surface is narrowed by the production target contract.

3. **ServingProfile is the central object.** All optimization pipelines target this normalized profile. Engine-specific compilers translate it to launch configs.

4. **Telemetry is Prometheus-first.** All engines expose `/metrics`. The normalizer converts engine-specific metrics to `NormalizedMetrics` for cross-engine audit checks.

## MCP Tool Surface (~22 tools)

### Dynamo Production Operations (5 tools)
- `tool_dynamo_cluster_health` — Disaggregated cluster health (frontend + workers + Grove)
- `tool_dynamo_failure_analysis` — What fell over: root cause, blast radius, recommended action
- `tool_dynamo_kv_pipeline` — NIXL transfer, Grove tiers, LMCache hit rates
- `tool_dynamo_slo_compliance` — TTFT/ITL/throughput vs SLO targets
- `tool_dynamo_disagg_topology` — Prefill/decode balance and routing efficiency

### Hardware & Model Intelligence (5 tools)
- `tool_get_gpu_specs`, `tool_compare_gpus`
- `tool_get_model_profile`, `tool_validate_serving_config`, `tool_estimate_capacity`

### Recommendations (2 tools)
- `tool_recommend_config` — Dynamo config for model+GPU+workload (default: coding)
- `tool_suggest_parallelism` — TP/PP/DP/EP strategy

### KV Cache Management (4 tools)
- `tool_calculate_kv_budget`, `tool_recommend_kv_strategy`
- `tool_recommend_disaggregation`, `tool_compare_quantization`

### Runtime Profiling (6 tools)
- `tool_profile_runtime`, `tool_audit_deployment`, `tool_check_deployment`
- `tool_check_memory_pressure`, `tool_get_cache_effectiveness`, `tool_auto_tune_deployment`

### Evaluation (~6 tools)
- Benchmark planning, execution, comparison, artifact management

## Dynamo Observability Contract

### Prometheus Metrics (scraped from Dynamo frontend + workers)
```
Frontend:  dynamo_frontend_{inflight,queued}_requests, dynamo_frontend_{ttft,itl,request_duration}_seconds
           dynamo_frontend_model_migration_total, dynamo_frontend_disconnected_clients
Workers:   dynamo_component_{inflight_requests,request_duration_seconds,requests_total}
KV:        dynamo_component_kvstats_{gpu_cache_usage_percent,gpu_prefix_cache_hit_rate,active_blocks,total_blocks}
NIXL:      dynamo_nixl_transfer_{latency_seconds,bytes_total,failures_total}
Grove:     dynamo_grove_tier_{gpu,cpu,ssd}_usage_percent, dynamo_grove_evictions_total
LMCache:   dynamo_lmcache_{hit,miss}_rate
SLO:       dynamo_slo_{ttft,itl}_violations_total
```

### Failure Taxonomy
| Mode | Severity | Detection |
|------|----------|-----------|
| `prefill_starvation` | critical | Prefill workers overloaded, decode workers idle |
| `decode_queue_backup` | critical | Decode queue depth exceeding threshold |
| `kv_transfer_timeout` | critical | NIXL transfer latency exceeding SLO |
| `nixl_failure` | critical | NIXL transfer failure counter increasing |
| `grove_tier_exhaustion` | critical | All KV tiers full |
| `worker_crash` | critical | Worker endpoint unreachable |
| `router_overload` | warning | Frontend queue depth exceeding threshold |
| `lmcache_miss_storm` | warning | Cache hit rate below threshold |

### Reliability Thresholds
- Minimum success rate: 99%
- Warning queue depth: 10 requests
- Warning KV usage: 90%
- Warning migrations: any (>0 during a run)

## DO NOTs

- Do NOT delete engine adapters (vllm.py, sglang.py, atom.py, trtllm.py) — they are used as Dynamo backends or by the CLI
- Do NOT delete GPU profiles or model entries — gating is in `production_target.py`
- Do NOT swallow exceptions with `except Exception: pass` — use specific types + structured logging
- Do NOT auto-select vLLM or SGLang as top-level engines for coding workloads — Dynamo is the answer
- Do NOT change default workload from `coding` to anything else in MCP tools

## Commands

```bash
# Run tests
cd products/inferscope && uv run pytest tests/ -v

# Run MCP server (stdio)
cd products/inferscope && uv run inferscope

# Lint
cd products/inferscope && uv run ruff check src/

# Type check
cd products/inferscope && uv run mypy src/inferscope/
```
