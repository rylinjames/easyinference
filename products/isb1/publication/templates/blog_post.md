{# ISB-1 Benchmark Blog Post Template #}
{# Render with Jinja2: provide a context dict with the variables referenced below. #}

# {{ title | default("ISB-1 Benchmark Results: LLM Inference Serving Performance") }}

*Published {{ publication_date | default("YYYY-MM-DD") }}*

---

{% if summary is defined %}
{{ summary }}
{% else %}
We are releasing the results of the ISB-1 benchmark, a comprehensive evaluation of LLM inference serving performance across {{ gpus | length }} GPU configurations, {{ models | length }} models, and {{ workloads | length }} production-representative workloads. This post highlights the key findings and what they mean for production deployments.
{% endif %}

---

## What is ISB-1?

ISB-1 (Inference Serving Benchmark Standard 1) is an open benchmark for evaluating how well inference engines serve large language models under realistic production conditions. Unlike single-model, single-GPU benchmarks, ISB-1 tests a full matrix of hardware, models, workloads, and optimization modes:

- **{{ workloads | length }} workloads:** {{ workload_names | default(["Chat", "Agent", "RAG", "Coding"]) | join(", ") }}
- **{{ gpus | length }} GPUs:** {{ gpus | join(", ") }}
- **{{ models | length }} models** spanning dense, MoE, and MLA architectures
- **{{ modes | length }} modes** from default vLLM to expert-tuned configurations

Every result is backed by at least {{ min_trials | default(3) }} trials with {{ measurement_duration | default(600) }}-second measurement windows and full reproducibility lockfiles.

---

## Key Findings

{% if key_findings is defined %}
{% for finding in key_findings %}
### {{ finding.title }}

{{ finding.body }}

{% if finding.figure is defined %}
![{{ finding.figure_caption | default("") }}]({{ finding.figure }})
{% endif %}

{% endfor %}
{% else %}
{# Placeholder: Insert 3-5 key findings, each with a title, body paragraph, and optional figure. #}

### Finding 1: [Title]

[Description of the finding with specific numbers.]

### Finding 2: [Title]

[Description of the finding with specific numbers.]

### Finding 3: [Title]

[Description of the finding with specific numbers.]
{% endif %}

---

## Throughput Highlights

{% if throughput_figure is defined %}
![Throughput comparison]({{ throughput_figure }})
*Generation throughput (tokens/s) across GPUs and models at the optimal request rate.*
{% endif %}

{% if top_results is defined %}
| Rank | GPU | Model | Workload | Mode | Throughput (tok/s) |
|------|-----|-------|----------|------|-------------------|
{% for row in top_results %}
| {{ loop.index }} | {{ row.gpu }} | {{ row.model }} | {{ row.workload }} | {{ row.mode }} | {{ "%.0f" | format(row.throughput) }} |
{% endfor %}
{% else %}
{# Placeholder: Insert top throughput results table. #}
{% endif %}

---

## Latency and SLO Attainment

{% if latency_figure is defined %}
![Latency distributions]({{ latency_figure }})
*TTFT and TPOT distributions across workloads.*
{% endif %}

{% if slo_summary is defined %}
{{ slo_summary }}
{% else %}
{# Placeholder: Summarize SLO attainment rates and which configurations met production latency targets. #}
{% endif %}

---

## What This Means for Production

{% if production_implications is defined %}
{{ production_implications }}
{% else %}
{# Placeholder: 2-3 paragraphs on practical implications for teams deploying LLM inference. #}

These results provide a data-driven foundation for hardware selection, model choice, and engine tuning decisions. Key takeaways for production teams:

1. **Hardware selection.** [Summary of GPU tradeoffs.]
2. **Model architecture impact.** [Summary of how MoE vs. dense models perform across workloads.]
3. **Optimization value.** [Summary of the gap between default and optimized configurations.]
{% endif %}

---

## Methodology in Brief

- **Measurement:** {{ measurement_duration | default(600) }}s steady-state windows per rate point, with validated warmup phases.
- **Statistics:** Paired t-tests for comparisons, BCa bootstrap 95% CIs, CV < 10% stability requirement.
- **Quality:** ROUGE-L, HumanEval, and MMLU-Pro checks ensure optimizations do not degrade output quality.
- **Reproducibility:** Full lockfiles with software versions, hardware state, config hashes, and random seeds.

For the complete methodology, see the [full whitepaper]({{ whitepaper_url | default("[whitepaper URL]") }}) and the [ISB-1 methodology documentation]({{ methodology_url | default("https://github.com/.../docs/METHODOLOGY.md") }}).

---

## Data Availability

All raw data, aggregated metrics, lockfiles, and configuration files are available at: {{ data_url | default("[data repository URL]") }}

The ISB-1 benchmark harness is open source under the Apache 2.0 license: {{ repo_url | default("[repository URL]") }}

---

{% if call_to_action is defined %}
## {{ call_to_action.title | default("Get Involved") }}

{{ call_to_action.body | default("We welcome contributions from the community. Hardware vendors and operators can submit Mode C configurations for inclusion in future benchmark runs.") }}
{% else %}
## Get Involved

We welcome contributions from the community. Hardware vendors and infrastructure operators can submit Mode C configurations for inclusion in future benchmark runs. See the [contribution guide]({{ contributing_url | default("[contributing URL]") }}) for details.
{% endif %}
