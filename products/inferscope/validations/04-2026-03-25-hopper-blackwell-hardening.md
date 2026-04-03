# Hopper / Blackwell hardening validation

Date: **March 25, 2026**

## Scope

This validation pass hardened InferScope's MCP/CLI optimization path for NVIDIA Hopper and Blackwell and checked the product position against the current public InferenceX reference.

Official external references used:

- InferenceX dashboard: <https://inferencex.semianalysis.com/>
- InferenceX repository: <https://github.com/SemiAnalysisAI/InferenceX>

Those public references confirm that InferenceX currently tracks hardware families including **H100**, **H200**, **B200**, **GB200 NVL72**, **GB300 NVL72**, and corresponding modern inference frameworks. InferScope is not trying to duplicate that public benchmark role. It is using the same platform reality to drive operator workflows.

## What changed

### Recommendation path

- added a shared platform-policy layer for Hopper/Blackwell
- preserved explicit platform metadata from GPU profile → recommender → compiler
- aligned `recommend_engine()` with the actual DAG-backed `recommend_config()` result
- stopped auto-promoting preview engines

### Compiler path

- fixed **B200 vs GB200** differentiation in the vLLM compiler
- fixed TRT-LLM scheduler field selection
- marked TRT-LLM and Dynamo as preview planning targets in public outputs

### Memory / KV path

- surfaced **Grace coherent overflow** as an advisory tier
- kept HBM fit semantics unchanged
- exposed the overflow tier through capacity and KV strategy surfaces

### Benchmark bridge

- made benchmark launcher workload mapping recognize `long_context_rag`
- added benchmark-plan regression coverage so benchmark stack plans inherit the same NVIDIA policy as the MCP
- added realistic long-context benchmark lanes for `OffloadingConnector` and `LMCache + Grace`

## Validated scenarios

- **H100 / DeepSeek-V3 / chat / 8 GPUs** → `vllm`, memory-valid quantized fallback, `TP=8`
- **H200 / DeepSeek-V3 / chat / 8 GPUs** → `vllm`, `fp8`, memory-valid `TP=8`
- **B200 / DeepSeek-V3 / chat / 4 GPUs** → Blackwell FP4 path, no Grace notes
- **GB200 / DeepSeek-V3 / long_context_rag / 4 GPUs** → Grace overflow advisory present
- coding workloads on **H100 / H200 / B200 / GB200** keep `recommend_engine()` aligned with `recommend_config()`
- benchmark stack planning on Hopper inherits the same TP decision as the optimizer

## Commands run

```bash
cd products/inferscope
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest tests -q
```

## Result

Pass.
