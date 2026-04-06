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
      server.py                  <- FastMCP entry point, calls register_*_tools(mcp)
      server_profiling.py        <- Runtime profiling MCP tools (6)
      server_benchmarks.py       <- Benchmark MCP tools (6)
      production_target.py       <- Supported model/GPU/engine contract
      security.py                <- SSRF protection, input validation
      endpoint_auth.py           <- HTTP auth helpers
      logging.py                 <- Structured logging (structlog, JSON prod, colored dev)
      config.py                  <- Global settings
      cli.py                     <- Typer root CLI (16 commands)
      cli_profiling.py           <- Profiling CLI commands (5)
      cli_benchmarks.py          <- Benchmark CLI commands (4)
      cli_experiments.py         <- Lightning Experiments wrapper command (1)
      engines/
        base.py                  <- EngineAdapter/ConfigCompiler ABCs
        registry.py              <- Engine adapter lookup
        dynamo.py                <- DynamoCompiler + DynamoAdapter (PRIMARY)
        vllm.py                  <- vLLM adapter (Dynamo backend; standalone CLI use)
        sglang.py                <- SGLang adapter (Dynamo backend; standalone CLI use)
        trtllm.py                <- TRT-LLM adapter (kept, not promoted)
        atom.py                  <- ATOM adapter (AMD, kept, not promoted)
      tools/
        recommend.py             <- Config/engine/parallelism recommendations (CLI-only surface)
        model_intel.py           <- Model profiles, validation, capacity (CLI-only surface)
        hardware_intel.py        <- GPU specs, comparison (CLI-only surface)
        kv_cache.py              <- KV budget, tiering, disaggregation, quantization (CLI-only surface)
        diagnose.py              <- Live deployment diagnostics (used by 3 MCP tools)
        audit.py                 <- Live deployment audit (used by tool_audit_deployment)
        live_tuner.py            <- Runtime auto-tuning (used by tool_auto_tune_deployment)
        profiling.py             <- Runtime profiling wrapper (used by tool_profile_runtime)
      optimization/
        serving_profile.py       <- ServingProfile (central normalized object)
        recommender.py           <- DAG-based recommendation pipeline (6 nodes)
        platform_policy.py       <- Engine support tiers, platform traits
        memory_planner.py        <- Memory breakdown calculation
        validator.py             <- Pre-flight config validation
        checks.py                <- 31 ISA-grounded audit checks
        workload_classifier.py   <- Workload mode classification
      telemetry/
        prometheus.py            <- Prometheus scraper (vLLM, SGLang, ATOM, Dynamo)
        normalizer.py            <- Cross-engine metric normalization (NormalizedMetrics)
        capture.py               <- Metric capture helpers
        failure_taxonomy.py      <- Failure mode classification (7 FailureMode values)
        models.py                <- Telemetry data models
      hardware/
        gpu_profiles.py          <- GPU ISA-level specs (all profiles kept, gated by production_target)
        detector.py              <- GPU SKU detection from text
      models/
        registry.py              <- Model profiles (all models kept, gated by production_target)
      benchmarks/
        models.py                <- WorkloadPack, BenchmarkArtifact, BenchmarkSummary
        runtime.py               <- run_openai_replay execution engine
        openai_replay.py         <- Thin wrapper around runtime
        probe_resolution.py      <- resolve_probe_plan central entry point
        preflight.py             <- Workload pack preflight validation
        support.py               <- assess_benchmark_support
        catalog.py               <- Resource resolution + compare_benchmark_artifacts
        procedural.py            <- materialize_procedural_workload
        experiments.py           <- BenchmarkExperimentSpec, BenchmarkRunPlan
        kv_capacity_probe.py     <- Phase 2 KV capacity sweep
        kv_pressure_ramp.py      <- Phase 3 KV pressure ramp
        kv_cache_behavior.py     <- Phase 4 KV cache behavior
        kv_disagg_transfer.py    <- Phase 5 disagg transfer
        kv_report.py             <- Combined KV phase report
        prometheus_capture.py    <- Wrapper around telemetry/capture.py
        _resources/workloads/    <- 16 packaged YAML workload packs
        _resources/experiments/  <- 19 packaged YAML experiment specs
      profiling/
        runtime.py               <- analyze_runtime, build_runtime_profile, derive_bottlenecks
        models.py                <- RuntimeContextHints, RuntimeIdentity, RuntimeProfileReport
        intents.py               <- Profiling intent catalog
        tuning.py                <- Tuning preview generation
    tests/                       <- pytest test suite (25 files, 4433 LOC, 218 test functions)
    docs/
      QUICKSTART.md              <- Operator onboarding
      MCP_QUICKSTART.md          <- MCP client setup
      DEPLOYMENT-GUIDE.md        <- Operator deployment patterns
      EXAMPLE_RESULTS.md         <- Reference benchmark output
  isb1/                <- Inference Serving Benchmark Standard 1 (Apache-2.0)
    workloads/         <- 4 canonical families: chat, agent, rag, coding (+ extensions: coderforge, swebench, deep_research_agent)
    harness/           <- vLLM server lifecycle, replay execution, sweep, manifest, lockfile
    analysis/          <- Metrics, aggregation, statistics, leaderboard, plots
    quality/           <- HumanEval, MMLU-Pro, ROUGE, RULER quality evaluators
    configs/           <- GPU, model, workload, mode, sweep YAML configs (~65 files)
```

## Key Architecture Decisions

1. **Dynamo is the only top-level engine.** The recommender DAG auto-selects Dynamo for coding/agent/RAG workloads on Hopper/Blackwell. vLLM and SGLang are kept as Dynamo worker backends.

2. **Scope gating via `production_target.py`, not file deletion.** All GPU profiles, models, and engine adapters remain in the codebase for CLI use. The MCP surface is narrowed by the production target contract.

3. **ServingProfile is the central object.** All optimization pipelines target this normalized profile. Engine-specific compilers translate it to launch configs.

4. **Telemetry is Prometheus-first.** All engines expose `/metrics`. The normalizer converts engine-specific metrics to `NormalizedMetrics` for cross-engine audit checks.

5. **MCP surface is intentionally narrow.** The MCP server registers exactly 12 tools — observation, measurement, and the supported probe surface. Recommendations and design-time intelligence (KV budget, GPU comparison, model profiles) live in the **CLI**, not the MCP, by design.

## MCP Tool Surface — exactly 12 tools

**Two families, 6 tools each.** Registered in `server.py:10` via `register_profiling_tools(mcp)` and `register_benchmark_tools(mcp)`.

### Profiling family (6 tools, registered in `server_profiling.py`)
- `tool_profile_runtime` — unified live runtime profile (22 params, the largest tool surface)
- `tool_audit_deployment` — run all 31 audit checks against a live endpoint
- `tool_check_deployment` — health snapshot
- `tool_check_memory_pressure` — KV cache utilization + preemption
- `tool_get_cache_effectiveness` — prefix cache hit rate
- `tool_auto_tune_deployment` — recommend config adjustments

### Benchmark family (6 tools, registered in `server_benchmarks.py`)
- `tool_get_production_contract` — return the supported production contract
- `tool_resolve_benchmark_plan` — resolve a probe into a concrete `BenchmarkRunPlan`
- `tool_run_benchmark` — execute a probe against an OpenAI-compatible endpoint (27 params)
- `tool_compare_benchmarks` — compare two saved artifacts
- `tool_get_benchmark_artifact` — load a saved artifact by filename
- `tool_validate_production_lane` — validate that an artifact belongs to the production lane

## CLI-only intelligence surface — NOT registered as MCP tools

The following functions exist in `tools/` and are exposed via the `inferscope` CLI but **NOT** via `@mcp.tool()`. The asymmetry is intentional per the FastMCP `instructions` block — the MCP surface is for observation/measurement; design-time intelligence is for human operators.

| CLI command | Function | Module |
|---|---|---|
| `inferscope gpu` | `get_gpu_specs` | `tools/hardware_intel.py` |
| `inferscope compare` | `compare_gpus` | `tools/hardware_intel.py` |
| `inferscope profile` | `get_model_profile` | `tools/model_intel.py` |
| `inferscope validate` | `validate_serving_config` | `tools/model_intel.py` |
| `inferscope capacity` | `estimate_capacity` | `tools/model_intel.py` |
| `inferscope recommend` | `recommend_config` | `tools/recommend.py` |
| `inferscope engine` | `recommend_engine` | `tools/recommend.py` |
| `inferscope parallelism` | `suggest_parallelism` | `tools/recommend.py` |
| `inferscope kv-budget` | `calculate_kv_budget` | `tools/kv_cache.py` |
| `inferscope kv-strategy` | `recommend_kv_strategy` | `tools/kv_cache.py` |
| `inferscope disagg` | `recommend_disaggregation` | `tools/kv_cache.py` |
| `inferscope quantization` | `compare_quantization` | `tools/kv_cache.py` |

That's **12 functions exposed via CLI but not via MCP**, mirroring the 12 MCP tools.

## ISB-1 Quality Track

`products/isb1/quality/` contains 4 quality evaluators that run alongside performance benchmarks (via `isb1 quality` CLI):

- `HumanEvalRunner` (`humaneval_runner.py`) — code completion
- `MMLUProEvaluator` (`mmlu_pro.py`) — multiple-choice reasoning
- `ROUGEEvaluator` (`rouge_eval.py`) — summarization quality
- `RULEREvaluator` (`ruler.py`) — long-context evaluation (largest of the 4)

Quality data lives in `products/isb1/quality/reference_outputs/` for golden-baseline comparison. **InferScope itself does not have a quality track** — all quality evaluation lives in ISB-1.

## Dynamo Observability Contract

### Prometheus Metrics (scraped from Dynamo frontend + workers)

The authoritative list lives in `telemetry/prometheus.py::DYNAMO_METRICS`. Notable: LMCache uses the `lmcache:` Prometheus prefix (not `dynamo_lmcache_*` as some older design notes claimed — see the self-documenting comment at `telemetry/normalizer.py:90-94`). Dynamo does not expose server-side SLO violation counters — that concept belongs to the client/harness side (see `telemetry/normalizer.py:270-275`).

```
Frontend:  dynamo_frontend_{inflight,queued}_requests, dynamo_frontend_{ttft,itl,request_duration}_seconds
           dynamo_frontend_model_migration_total, dynamo_frontend_disconnected_clients
Workers:   dynamo_component_{inflight_requests,request_duration_seconds,requests_total}
KV:        dynamo_component_kvstats_{gpu_cache_usage_percent,gpu_prefix_cache_hit_rate,active_blocks,total_blocks}
NIXL:      dynamo_nixl_transfer_{latency_seconds,bytes_total,failures_total}
KVBM:      kvbm_offload_blocks_d2h, kvbm_onboard_blocks_h2d, kvbm_host_cache_hit_rate
LMCache:   lmcache:num_hit_tokens_total, lmcache:num_requested_tokens_total, lmcache:retrieve_speed_*
```

### Failure Taxonomy

Defined as `FailureMode` enum in `telemetry/failure_taxonomy.py:11-22`. Current source has **7 modes**:

| Mode | Severity | Detection |
|------|----------|-----------|
| `PREFILL_STARVATION` | critical | Prefill workers overloaded, decode workers idle |
| `DECODE_QUEUE_BACKUP` | critical | Decode queue depth exceeding threshold |
| `KV_TRANSFER_TIMEOUT` | critical | NIXL transfer latency exceeding SLO |
| `NIXL_FAILURE` | critical | NIXL transfer failure counter increasing |
| `WORKER_CRASH` | critical | Worker endpoint unreachable |
| `ROUTER_OVERLOAD` | warning | Frontend queue depth exceeding threshold |
| `LMCACHE_MISS_STORM` | warning | LMCache hit rate below threshold |

(Earlier drafts of this file listed `grove_tier_exhaustion` as an 8th mode. It was removed in the Grove → KVBM correction.)

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
- Do NOT add new MCP tools without updating the §"MCP Tool Surface — exactly 12 tools" section above
- Do NOT reference fictional metric prefixes (e.g., `dynamo_grove_*`, `dynamo_slo_*`, `dynamo_lmcache_*`) — use the real names per `telemetry/prometheus.py::DYNAMO_METRICS`

## Commands

```bash
# Run tests
cd products/inferscope && uv run pytest tests/ -v

# Run MCP server (stdio)
cd products/inferscope && uv run inferscope serve

# Generate MCP config for Cursor / Claude Desktop
cd products/inferscope && uv run inferscope connect

# Lint
cd products/inferscope && uv run ruff check src/

# Type check
cd products/inferscope && uv run mypy src/inferscope/
```
