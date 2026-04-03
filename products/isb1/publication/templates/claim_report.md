{# ISB-1 Claim Evaluation Report Template #}
{# Render with Jinja2: provide a context dict with the variables referenced below. #}

# ISB-1 Claim Evaluation Report

**Benchmark Version:** {{ benchmark_version | default("1.0.0") }}
**Report Date:** {{ report_date | default("YYYY-MM-DD") }}
**Evaluator:** {{ evaluator | default("ISB-1 Automated Claim Evaluator") }}

---

## Summary

{% if claims is defined %}
This report evaluates **{{ claims | length }}** performance claim{{ "s" if claims | length != 1 else "" }} against data measured under the ISB-1 benchmark standard.

| Verdict | Count |
|---------|-------|
| Supported | {{ claims | selectattr("verdict", "equalto", "Supported") | list | length }} |
| Not Supported | {{ claims | selectattr("verdict", "equalto", "Not Supported") | list | length }} |
| Inconclusive | {{ claims | selectattr("verdict", "equalto", "Inconclusive") | list | length }} |
{% else %}
{# Placeholder: No claims provided. #}
No claims were provided for evaluation.
{% endif %}

---

## Methodology

Claims are evaluated using the following process:

1. **Data retrieval.** Aggregated ISB-1 metrics are loaded for the cells referenced by each claim.
2. **Delta computation.** The actual difference or ratio between the claimed configurations is computed.
3. **Statistical test.** A paired t-test (alpha = 0.05) determines whether the observed difference is statistically significant.
4. **Confidence interval.** A BCa bootstrap 95% CI is computed on the metric difference.
5. **Verdict assignment.**
   - **Supported:** The measured improvement meets or exceeds the claim, and the difference is statistically significant (p < 0.05).
   - **Not Supported:** The measured improvement does not meet the claim, or the difference is not significant.
   - **Inconclusive:** Insufficient data, high variance, or the CI spans the claim threshold.

---

## Claim Evaluations

{% if claims is defined %}
{% for claim in claims %}
### Claim {{ loop.index }}: {{ claim.title | default("Untitled Claim") }}

**Claim statement:** {{ claim.description }}

**Source:** {{ claim.source | default("Not specified") }}

#### Configuration

| Parameter | Baseline | Compared |
|-----------|----------|----------|
| GPU | {{ claim.baseline.gpu }} | {{ claim.compared.gpu }} |
| Model | {{ claim.baseline.model }} | {{ claim.compared.model }} |
| Workload | {{ claim.baseline.workload }} | {{ claim.compared.workload }} |
| Mode | {{ claim.baseline.mode }} | {{ claim.compared.mode }} |
| Quantization | {{ claim.baseline.quantization }} | {{ claim.compared.quantization }} |

#### Claimed Improvement

- **Metric:** {{ claim.metric }}
- **Direction:** {{ claim.direction | default("higher is better") }}
- **Claimed value:** {{ claim.claimed_value }}

#### Measured Results

| Metric | Baseline | Compared | Delta | Ratio |
|--------|----------|----------|-------|-------|
| {{ claim.metric }} | {{ claim.baseline_measured }} | {{ claim.compared_measured }} | {{ claim.measured_delta }} | {{ claim.measured_ratio }} |

{% if claim.additional_metrics is defined %}
**Supporting metrics:**

| Metric | Baseline | Compared |
|--------|----------|----------|
{% for m in claim.additional_metrics %}
| {{ m.name }} | {{ m.baseline }} | {{ m.compared }} |
{% endfor %}
{% endif %}

#### Statistical Analysis

- **Paired t-test:** t = {{ claim.t_statistic }}, p = {{ claim.p_value }}
- **Significant at alpha = 0.05:** {{ "Yes" if claim.significant else "No" }}
- **Bootstrap 95% CI on difference:** [{{ claim.ci_lower }}, {{ claim.ci_upper }}]
- **Number of trials:** {{ claim.num_trials | default("3") }} per configuration
- **CV (baseline):** {{ claim.cv_baseline | default("N/A") }}
- **CV (compared):** {{ claim.cv_compared | default("N/A") }}

{% if claim.high_variance | default(false) %}
**Warning:** One or more configurations exhibited high variance (CV > 10%). The statistical power of this comparison may be limited.
{% endif %}

#### Verdict: **{{ claim.verdict }}**

{{ claim.verdict_rationale | default("") }}

---

{% endfor %}
{% else %}
{# Placeholder: No claims to evaluate. #}
No claims were provided for evaluation in this report.

---
{% endif %}

## Data Provenance

{% if data_provenance is defined %}
- **Sweep config:** {{ data_provenance.sweep_config }}
- **Run date:** {{ data_provenance.run_date }}
- **Lockfile:** {{ data_provenance.lockfile_path }}
- **Raw data:** {{ data_provenance.raw_data_path }}
- **Aggregated data:** {{ data_provenance.aggregated_data_path }}
{% else %}
{# Placeholder: data provenance details #}
- **Sweep config:** [path to sweep YAML]
- **Run date:** [date of benchmark execution]
- **Lockfile:** [path to lockfile]
- **Raw data:** [path to raw results]
- **Aggregated data:** [path to aggregated results]
{% endif %}

## Reproducibility

All claims can be independently verified by:

1. Checking out the ISB-1 benchmark at version {{ benchmark_version | default("1.0.0") }}.
2. Using the lockfile to match the software environment.
3. Running the relevant cells with the same configuration and seeds.
4. Comparing results against the bootstrap 95% CI from this report.

Results are considered reproduced if the measured metrics fall within the reported confidence intervals.

---

*Report generated by the ISB-1 Claim Evaluation Framework. For methodology details, see [METHODOLOGY.md]({{ methodology_url | default("docs/METHODOLOGY.md") }}).*
