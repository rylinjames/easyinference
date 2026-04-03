{# ISB-1 Benchmark Whitepaper Template #}
{# Render with Jinja2: provide a context dict with the variables referenced below. #}

# ISB-1: Inference Serving Benchmark Standard 1

**Version {{ benchmark_version | default("1.0.0") }}**
**Date: {{ publication_date | default("YYYY-MM-DD") }}**

---

## Abstract

This report presents the results of the ISB-1 (Inference Serving Benchmark Standard 1) benchmark evaluation. ISB-1 measures large language model inference serving performance across {{ gpus | length }} GPU configurations, {{ models | length }} models, {{ workloads | length }} production-representative workloads, and {{ modes | length }} execution modes. The benchmark evaluates throughput, latency, goodput, SLO attainment, and output quality using a statistically rigorous methodology with a minimum of {{ min_trials | default(3) }} trials per cell and {{ measurement_duration | default(600) }}-second steady-state measurement windows.

{% if key_findings is defined %}
Key findings include:
{% for finding in key_findings %}
- {{ finding }}
{% endfor %}
{% endif %}

---

## 1. Introduction

Large language model inference serving has become a critical infrastructure workload. The diversity of model architectures (dense, MoE, MLA), quantization formats (bf16, fp8, nvfp4), and hardware platforms (H100, H200, B200, B300) demands a standardized benchmark that enables fair, reproducible comparison.

ISB-1 addresses this need with four workloads designed to exercise distinct serving patterns:

| Workload | Description |
|----------|-------------|
{% for wl in workloads %}
| {{ wl.name }} ({{ wl.id }}) | {{ wl.description }} |
{% endfor %}

---

## 2. Methodology

### 2.1 Benchmark Matrix

The evaluation covers the following matrix:

**GPUs:** {{ gpus | join(", ") }}

**Models:**
{% for model in models %}
- {{ model.name }} ({{ model.architecture }})
{% endfor %}

**Modes:**
{% for mode in modes %}
- **{{ mode.name }}:** {{ mode.description }}
{% endfor %}

**Quantizations:** {{ quantizations | join(", ") }}

### 2.2 Measurement Protocol

Each benchmark cell follows the ISB-1 measurement protocol:

1. **Warmup:** {{ warmup_requests | default(100) }} requests over a minimum of {{ warmup_seconds | default(60) }} seconds. Steady-state is validated by monitoring throughput CV across sliding windows (threshold: {{ steady_state_threshold | default("20%") }}).
2. **Measurement:** {{ measurement_duration | default(600) }} seconds of steady-state measurement per rate point.
3. **Trials:** {{ min_trials | default(3) }} trials per cell, extended to {{ max_trials | default(5) }} if CV exceeds {{ cv_threshold | default("10%") }}.
4. **Statistical tests:** Paired t-tests for mode comparisons; BCa bootstrap 95% confidence intervals for metric uncertainty.

### 2.3 Metric Definitions

| Metric | Definition |
|--------|-----------|
| TTFT | Time from request submission to first output token |
| TPOT | (E2E latency - TTFT) / (output tokens - 1). **TTFT excluded.** |
| ITL | Inter-token gaps starting from token index 2. **TTFT excluded.** |
| Goodput | Rate of requests meeting both TTFT and TPOT SLO thresholds |
| SLO Attainment | Fraction of successful requests meeting all SLOs |

---

## 3. Results

### 3.1 Throughput Overview

{% if throughput_table is defined %}
{{ throughput_table }}
{% else %}
{# Placeholder: Insert throughput summary table #}
| GPU | Model | Workload | Mode A (tok/s) | Mode B (tok/s) | Improvement |
|-----|-------|----------|---------------|---------------|-------------|
{% for row in throughput_rows | default([]) %}
| {{ row.gpu }} | {{ row.model }} | {{ row.workload }} | {{ "%.1f" | format(row.mode_a) }} | {{ "%.1f" | format(row.mode_b) }} | {{ "%.1f%%" | format(row.improvement * 100) }} |
{% endfor %}
{% endif %}

{% if throughput_figure is defined %}
![Throughput comparison across GPUs and models]({{ throughput_figure }})
*Figure 1: Generation throughput (tokens/s) across the benchmark matrix.*
{% else %}
{# Placeholder: throughput figure path #}
{% endif %}

### 3.2 Latency Analysis

{% if latency_table is defined %}
{{ latency_table }}
{% else %}
{# Placeholder: Insert latency summary table #}
| GPU | Model | Workload | TTFT p95 (ms) | TPOT p95 (ms) | ITL p95 (ms) |
|-----|-------|----------|--------------|--------------|-------------|
{% for row in latency_rows | default([]) %}
| {{ row.gpu }} | {{ row.model }} | {{ row.workload }} | {{ "%.1f" | format(row.ttft_p95 * 1000) }} | {{ "%.2f" | format(row.tpot_p95 * 1000) }} | {{ "%.2f" | format(row.itl_p95 * 1000) }} |
{% endfor %}
{% endif %}

{% if latency_figure is defined %}
![Latency distribution across workloads]({{ latency_figure }})
*Figure 2: Latency percentile distributions across workloads.*
{% else %}
{# Placeholder: latency figure path #}
{% endif %}

### 3.3 Goodput and SLO Attainment

{% if goodput_table is defined %}
{{ goodput_table }}
{% else %}
{# Placeholder: Insert goodput summary table #}
| GPU | Model | Workload | Mode | Goodput (req/s) | SLO Attainment |
|-----|-------|----------|------|----------------|----------------|
{% for row in goodput_rows | default([]) %}
| {{ row.gpu }} | {{ row.model }} | {{ row.workload }} | {{ row.mode }} | {{ "%.2f" | format(row.goodput) }} | {{ "%.1f%%" | format(row.slo_attainment * 100) }} |
{% endfor %}
{% endif %}

### 3.4 Rate Sweep Curves

{% if rate_sweep_figure is defined %}
![Throughput-latency tradeoff curves]({{ rate_sweep_figure }})
*Figure 3: Throughput vs. latency tradeoff as request rate increases.*
{% else %}
{# Placeholder: rate sweep figure path #}
{% endif %}

### 3.5 Power Efficiency

{% if power_table is defined %}
{{ power_table }}
{% else %}
{# Placeholder: Insert power efficiency table #}
| GPU | Model | Mode | Avg Power (W) | Watts/Token |
|-----|-------|------|---------------|-------------|
{% for row in power_rows | default([]) %}
| {{ row.gpu }} | {{ row.model }} | {{ row.mode }} | {{ "%.0f" | format(row.avg_power) }} | {{ "%.3f" | format(row.watts_per_token) }} |
{% endfor %}
{% endif %}

### 3.6 Leaderboard

{% if leaderboard_figure is defined %}
![Leaderboard heatmap]({{ leaderboard_figure }})
*Figure 4: Leaderboard heatmap showing relative performance across the matrix.*
{% else %}
{# Placeholder: leaderboard heatmap figure path #}
{% endif %}

{% if leaderboard_table is defined %}
{{ leaderboard_table }}
{% endif %}

---

## 4. Claim Evaluation

{% if claims is defined %}
The following performance claims were evaluated against the measured data:

{% for claim in claims %}
### Claim {{ loop.index }}: {{ claim.description }}

- **Subject:** {{ claim.subject }}
- **Asserted metric:** {{ claim.metric }} {{ claim.direction }} by {{ claim.threshold }}
- **Measured delta:** {{ claim.measured_delta }}
- **p-value:** {{ claim.p_value }}
- **95% CI:** [{{ claim.ci_lower }}, {{ claim.ci_upper }}]
- **Verdict:** **{{ claim.verdict }}**

{% endfor %}
{% else %}
{# Placeholder: claim evaluation results #}
No claims were evaluated in this benchmark run.
{% endif %}

---

## 5. Quality Evaluation

{% if quality_results is defined %}
Output quality was assessed across all optimized configurations:

| Configuration | ROUGE-L vs bf16 | HumanEval pass@1 | MMLU-Pro Accuracy |
|--------------|-----------------|-------------------|-------------------|
{% for row in quality_results %}
| {{ row.config }} | {{ "%.3f" | format(row.rouge_l) }} | {{ "%.1f%%" | format(row.humaneval * 100) }} | {{ "%.1f%%" | format(row.mmlu_pro * 100) }} |
{% endfor %}

{% if quality_pass %}
All configurations passed the quality evaluation criteria.
{% else %}
Some configurations showed quality degradation. See detailed results in the appendix.
{% endif %}
{% else %}
{# Placeholder: quality evaluation results #}
{% endif %}

---

## 6. Conclusion

{% if conclusion is defined %}
{{ conclusion }}
{% else %}
{# Placeholder: Write conclusion summarizing key findings, implications, and future work. #}
This benchmark evaluation provides a comprehensive assessment of LLM inference serving performance across the ISB-1 matrix. All measurements follow the ISB-1 methodology with full reproducibility lockfiles available in the companion data release.
{% endif %}

---

## Appendix A: Reproducibility

All benchmark runs include complete lockfiles capturing:
- vLLM version and git hash
- CUDA and PyTorch versions
- Full nvidia-smi output and NVLink topology
- pip freeze package listing
- SHA-256 hashes of all configuration files
- HuggingFace model revisions
- Random seeds

Lockfiles and raw data are available at: {{ data_url | default("[data repository URL]") }}

## Appendix B: Statistical Details

- Paired t-tests: two-sided, alpha = 0.05
- Bootstrap CIs: BCa method, 10,000 resamples, 95% confidence
- CV threshold: 10% (cells exceeding threshold are flagged)
- Minimum trials: {{ min_trials | default(3) }}, maximum: {{ max_trials | default(5) }}
