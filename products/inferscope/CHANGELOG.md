# Changelog

All notable changes to InferScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Validation: [03-runtime-profiling-v1](validations/03-2026-03-25-runtime-profiling-v1.md), [04-hopper-blackwell-hardening](validations/04-2026-03-25-hopper-blackwell-hardening.md)

### Added
- Shared Hopper/Blackwell platform policy used by the recommendation path and support validation
- Shared `probe_resolution.py` to keep CLI and MCP benchmark-plan logic on one narrowed contract
- New product docs that explicitly position InferScope as a runtime profiling + narrow probe product
- Audit/target-architecture document at `docs/AUDIT-TARGET-ARCHITECTURE.md`

### Changed
- `production_target.py` is now the single supported-contract authority
- Benchmark CLI is narrowed to `benchmark-plan`, `benchmark`, and `benchmark-compare`
- Benchmark MCP is narrowed to production contract, probe resolution, probe execution, artifact load, and artifact comparison
- The top-level MCP server no longer presents InferScope as a generic hardware/model/recommendation toolbox
- Default benchmark-plan resolution now flows through the aggregated Dynamo Kimi probe lane
- Benchmark docs and root docs now describe InferScope as a deployment-diagnostics product rather than a benchmark framework

### Removed
- Benchmark matrix public surfaces
- Benchmark strategy public surfaces
- Benchmark stack-plan and stack materialization public surfaces
- Dead benchmark launcher and strategy modules
- Duplicate `optimization/target_profile.py` scope definition

## [0.1.0] - 2026-03-23

### Added
- Initial release with Phase 1 functionality
- **15 MCP tools** across 5 groups: hardware, model intel, recommendations, KV cache, live diagnostics
- **16 CLI commands**: profile, validate, recommend, gpu, compare, capacity, engine, parallelism, kv-budget, kv-strategy, disagg, quantization, check, memory, cache, serve
- **GPU knowledge base**: 9 variants across 5 architectures (Ampere, Hopper, Blackwell, CDNA3, CDNA4)
- **Model registry**: 12+ models across 5 classes (Dense-GQA, Qwen3.5-Hybrid, Frontier-MLA-MoE, Compact-Agentic-MoE, Classical-MoE)
- **3 engine compilers**: vLLM, SGLang, ATOM (TRT-LLM and Dynamo as stubs)
- Normalized ServingProfile → ConfigCompiler → EngineConfig pipeline
- Memory planner with per-layer KV cache math
- Pre-flight config validation (TP divisibility, memory fit, format compatibility)
- KV cache tiering strategy recommendations (GPU/CPU/SSD)
- Prefill/decode disaggregation decision tool
- Quantization comparison tool with GPU format awareness
- Prometheus metric scraping for vLLM/SGLang/ATOM with auto-detection
- Live diagnostics: check, memory pressure, cache effectiveness
- Input validation and SSRF protection for HTTP endpoints
- 129+ unit tests across 9 test files

### Security
- SSRF protection: private IP blocking, URL scheme validation
- MCP tools block private IPs by default; CLI allows private for local operator use
- Localhost-only binding for HTTP transport by default
- Input validators wired into tool entrypoints (model names, GPU names, numeric bounds)
