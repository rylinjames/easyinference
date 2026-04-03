# InferScope AI-First Validation Report

**Report version:** 01  
**Date:** March 23, 2026  
**Package version reviewed:** 0.1.0

## Executive summary

InferScope is a real and useful **hardware-aware inference recommendation and validation layer** for open-model serving, but it is **not yet a full enterprise inference control plane**.

What is solid today:

- GPU/model/workload reasoning
- ServingProfile -> compiler -> engine-config flow
- Config generation for **vLLM**, **SGLang**, and **ATOM**
- KV-cache sizing, quantization comparison, and static preflight validation
- Prometheus parsing/normalization for vLLM, SGLang, and ATOM metrics text

What is not complete yet:

- no implemented **21 live audit checks**
- no benchmark integration or optimization sweep
- no real **TRT-LLM** compiler
- no real **Dynamo** compiler
- no meaningful integration test coverage for live engines

## Scope of this validation

This report was built from:

- repository docs review
- source-code inspection
- local command execution
- test/lint/security runs
- targeted scenario validation
- current external market/product research for enterprise coding-agent workloads

This report intentionally distinguishes between:

- **verified repo behavior**
- **directionally correct but overstated claims**
- **future roadmap items**

## What was run

### Commands run

```bash
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
uv run mypy src/inferscope/
uv build
```

### Actual results

#### 1. Fresh-clone pytest path

This currently **fails** as documented:

```bash
uv sync --dev
uv run pytest tests/ -v
```

Observed failure:

```text
ModuleNotFoundError: No module named 'inferscope'
```

#### 2. Working unit-test path

This **passes**:

```bash
PYTHONPATH=src uv run pytest tests/ -v
```

Observed result:

```text
129 passed
```

A local editable install also made imports work.

#### 3. Ruff

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Observed result:

- passed

#### 4. Bandit

```bash
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

Observed result:

- **0 medium/high findings**

#### 5. Mypy

```bash
uv run mypy src/inferscope/
```

Observed result:

- **fails** locally with multiple typing errors
- local run produced **78 errors**

#### 6. Build

```bash
uv build
```

Observed result:

- wheel and source distribution built successfully

## Verified implementation status

### Verified now

- **15 MCP tools** are registered in `src/inferscope/server.py`
- **16 CLI commands** are registered in `src/inferscope/cli.py`
- **12 GPU profiles** are present in `src/inferscope/hardware/gpu_profiles.py`
- **14 model variants** are present in `src/inferscope/models/registry.py`
- `recommend_config()` returns:
  - normalized serving profile
  - engine-specific config
  - memory plan
  - launch command when compiler supports it
- static validation covers:
  - TP divisibility
  - memory fit
  - quantization compatibility
  - known AMD/NVIDIA caveats
- telemetry layer includes:
  - Prometheus scraping
  - engine detection
  - metrics parsing
  - normalization
  - three diagnostic tools: `check`, `memory`, `cache`

### Implemented engines

These compilers are materially implemented:

- **vLLM**
- **SGLang**
- **ATOM**

These are present but still stubs:

- **TRT-LLM**
- **Dynamo**

## Claims review

### PASS

- InferScope is a hardware-aware recommendation/validation layer.
- vLLM, SGLang, and ATOM configs are generated from a shared normalized profile.
- KV-cache math is grounded in model attributes.
- TP validation catches invalid values.
- Quantization guidance respects GPU format support.
- Prometheus text parsing and normalization exist.
- HTTP MCP mode binds to `127.0.0.1`.

### PARTIAL

- "Supports 5 engines"
  - structurally true
  - operationally only **3 of 5** today
- "Live deployment auditing"
  - basic scrape/normalize/diagnose exists
  - the claimed **21 audit checks** do not
- "Security validation on endpoints"
  - validation helpers exist and are tested
  - they are not consistently enforced across all user-facing entrypoints

### FAIL / overstated

- The documented fresh-clone pytest command is currently not reproducible as written.
- `mypy` does not pass in the current local run.
- `tests/integration/` does not yet provide real integration coverage.
- It is inaccurate to say that **all engines** produce real executable configs.
- It is inaccurate to describe the scenario validations as formal automated integration tests.

## Scenario validation

The following scenarios were executed directly against the current code paths.

### Scenario 1: DeepSeek-R1 on 8x H100 for coding

Input:

```python
recommend_config("DeepSeek-R1", "h100", workload="coding", num_gpus=8)
```

Observed output:

- engine: `sglang`
- TP=8, DP=1, EP=1
- precision: `fp8`
- result: **does not fit** at current conservative memory settings

Observed summary:

```text
Recommended: DeepSeek-R1 on 8× H100 SXM | Engine: sglang | TP=8 DP=1 EP=1 | Precision: fp8 | Workload: coding | ❌ does not fit
```

Assessment:

- this is a valid and useful conservative result
- the tool is not hiding the memory constraint

### Scenario 2: Qwen3.5-32B on 1x H200 for chat

Input:

```python
recommend_config("Qwen3.5-32B", "h200", workload="chat", num_gpus=1)
```

Observed output:

- engine: `vllm`
- TP=1
- precision: `fp8`
- result: **fits**

Assessment:

- this is a clean and credible default deployment recommendation

### Scenario 3: DeepSeek-R1 on 8x MI355X for agent workload

Input:

```python
recommend_config("DeepSeek-R1", "mi355x", workload="agent", num_gpus=8)
```

Observed output:

- engine: `atom`
- TP=4, EP=2
- precision: `fp8`
- result: **fits**
- AMD env vars set, including AITER and MLA-related settings

Assessment:

- this is one of the clearest differentiated paths in the repo

### Scenario 4: Invalid TP rejection

Input:

```python
validate_serving_config("Llama-3-70B", "h100", tp=3)
```

Observed output:

```text
TP=3 does not evenly divide num_kv_heads=8. Valid TP values: [1, 2, 4, 8]
```

Assessment:

- high-value preflight validation

### Scenario 5: KV-cache budget at 128K context, batch 10

Inputs:

```python
calculate_kv_budget("DeepSeek-R1", 131072, batch_size=10, kv_dtype="fp8")
calculate_kv_budget("Llama-3-70B", 131072, batch_size=10, kv_dtype="fp8")
```

Observed outputs:

- DeepSeek-R1: **76.25 GB**
- Llama-3-70B: **200.00 GB**

Assessment:

- useful for real capacity planning
- MLA advantage is visible in output

### Scenario 6: Disaggregation decision

Input A:

```python
recommend_disaggregation(
    "DeepSeek-R1",
    "h100",
    avg_prompt_tokens=32768,
    request_rate_per_sec=50,
    has_rdma=True,
    num_gpus=8,
)
```

Observed output:

- **recommended**
- connector: `NixlConnector (UCX/libfabric/EFA)`

Input B:

```python
recommend_disaggregation(
    "DeepSeek-R1",
    "h100",
    avg_prompt_tokens=256,
    request_rate_per_sec=5,
    has_rdma=False,
    num_gpus=8,
)
```

Observed output:

- **not recommended**

Assessment:

- heuristics are useful
- still static rules, not telemetry-backed decisions yet

### Scenario 7: Quantization comparison

Input:

```python
compare_quantization("Llama-3-70B", "h100")
```

Observed top options:

1. `fp8`
2. `bf16`
3. `int8`

Assessment:

- reasonable ranking for H100

### Scenario 8: Live scrape negative-path check

Input:

```python
check_deployment("http://127.0.0.1:9999")
```

Observed output:

```text
Connection refused: http://127.0.0.1:9999/metrics — is the engine running?
```

Assessment:

- graceful failure path works
- this does **not** count as live-engine integration validation

## Engine-specific expectations

### vLLM

Expected current behavior:

- produces non-empty launch command
- supports quantization flags
- supports KV cache dtype
- supports AMD AITER env vars on ROCm

### SGLang

Expected current behavior:

- produces non-empty launch command
- sets `lpm` scheduling for coding/agent workloads
- enables metrics flag

### ATOM

Expected current behavior:

- produces non-empty launch command
- AMD-only path
- sets AITER-related env vars
- sets MLA backend for frontier MLA models

### TRT-LLM

Expected current behavior today:

- returns stub config
- no real launch command
- should be documented as future work

### Dynamo

Expected current behavior today:

- returns stub config
- no real launch command
- should be documented as future work

## Security findings

### Verified

- `validate_endpoint()` blocks private IPs and localhost by default
- `file://` is blocked
- MCP HTTP mode binds localhost only

### Important nuance

The live diagnostics path intentionally calls endpoint validation with `allow_private=True`, so local endpoints like `http://127.0.0.1:9999` are allowed for diagnostics.

That means the claim should be:

- **default endpoint validation blocks localhost/private IPs**
- **diagnostic tools explicitly allow local/private endpoints for operator use**

Not:

- "all endpoint paths block localhost/private IPs"

## Important defects / credibility gaps

1. **Broken documented pytest path**
   - biggest immediate docs credibility issue

2. **Typecheck mismatch**
   - strict mypy config exists
   - current local run does not pass

3. **No real integration tests**
   - unit coverage is good
   - integration confidence is limited

4. **Stub engines exposed alongside implemented engines**
   - needs explicit product-positioning language

5. **Unused API knobs**
   - `target_latency_ms` is exposed but unused
   - `target_ttft_ms` is exposed but unused

6. **Engine-ranking ambiguity**
   - multi-node NVIDIA MLA chat ranking can produce tied rank-1 options while summary still defaults to first item

7. **Validation helper usage is inconsistent**
   - helper functions exist
   - wiring is incomplete across CLI/MCP boundaries

## Recommended wording for public docs

Use wording like this:

> InferScope is a hardware-aware recommendation and validation toolkit for LLM inference deployments. It currently generates executable configs for vLLM, SGLang, and ATOM, and includes early telemetry diagnostics. TRT-LLM and Dynamo support are planned but not fully implemented yet.

Avoid wording like this:

> InferScope fully supports five inference engines and live enterprise deployment auditing.

## Immediate next steps

### Must do before external sharing

1. fix fresh-clone package/test reproducibility
2. fix or scope down mypy/CI claims
3. rewrite validation docs to distinguish unit validation from integration validation
4. clearly label TRT-LLM and Dynamo as stubs

### Product work next

1. implement the 21 live audit checks
2. add integration tests for telemetry and CLI smoke paths
3. add at least one live-engine fixture path for vLLM/SGLang metrics
4. decide whether InferScope is positioned as:
   - inference planning advisor, or
   - inference control plane

## Sources

### Repo evidence

- `src/inferscope/cli.py`
- `src/inferscope/server.py`
- `src/inferscope/tools/recommend.py`
- `src/inferscope/tools/model_intel.py`
- `src/inferscope/tools/kv_cache.py`
- `src/inferscope/tools/diagnose.py`
- `src/inferscope/optimization/recommender.py`
- `src/inferscope/optimization/validator.py`
- `src/inferscope/optimization/memory_planner.py`
- `src/inferscope/engines/vllm.py`
- `src/inferscope/engines/sglang.py`
- `src/inferscope/engines/atom.py`
- `src/inferscope/engines/trtllm.py`
- `src/inferscope/engines/dynamo.py`
- `.github/workflows/ci.yml`

### Official external docs used for market/product context

- OpenAI Codex: https://openai.com/index/introducing-codex/
- OpenAI Codex product page: https://openai.com/codex/
- GitHub Copilot supported models: https://docs.github.com/en/copilot/reference/ai-models/supported-models
- GitHub Copilot coding agent docs: https://docs.github.com/en/copilot/concepts/agents/coding-agent/about-coding-agent
- Anthropic Claude Code Enterprise: https://claude.com/product/claude-code/enterprise
- Google Gemini Code Assist agent mode: https://docs.cloud.google.com/gemini/docs/codeassist/use-agentic-chat-pair-programmer
- Google Gemini Code Assist customization and use cases: https://docs.cloud.google.com/gemini/docs/codeassist/use-code-customization
- Amazon Q Developer overview: https://aws.amazon.com/documentation-overview/q-developer/
- AIDev paper: https://arxiv.org/abs/2602.09185
