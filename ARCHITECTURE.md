# EasyInference Monorepo Architecture

EasyInference is a two-product monorepo with a hard boundary between benchmark standardization and operator diagnostics.

- **ISB-1** is the reproducible benchmark standard
- **InferScope** is the operator-facing runtime profiling and narrow probe product

## Product boundary

The repo is meant to be complementary to public frontier benchmarking, not a clone of it.

- **InferenceX** owns the broad public frontier benchmark layer
- **ISB-1** owns reproducible benchmark methodology, configs, harnesses, analysis, and publication in this repo
- **InferScope** owns deployment-specific profiling, probe execution, and artifact comparison

If a feature looks like generic benchmark infrastructure, it should default toward **ISB-1**.
If a feature looks like deployment diagnosis or remediation, it should default toward **InferScope**.

## Repository structure

```text
EasyInference/
├── products/
│   ├── isb1/         # benchmark standard, harness, configs, analysis, publication
│   └── inferscope/   # operator diagnostics, runtime profiling, narrow probe tooling
├── demo/
├── .github/workflows/
├── docs/
├── README.md
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── VALIDATION.md
└── Makefile
```

## Product responsibilities

### `products/isb1/`

ISB-1 owns the benchmark system of record:

- workload generators and schemas
- harness execution
- telemetry capture for benchmark runs
- manifests and lockfiles
- analysis, statistics, and reporting
- publication assets and claim evaluation

### `products/inferscope/`

InferScope owns the operator-facing product:

- live runtime profiling
- deployment checks and bottleneck summaries
- narrow benchmark probe resolution and execution
- artifact comparison for deployment changes
- MCP and CLI presentation layers for those workflows

InferScope should not be used to rebuild a second benchmark platform beside ISB-1.

## InferScope internal boundary

InferScope should be understood as four layers:

1. **scope** — `production_target.py` defines the supported product lane
2. **telemetry + profiling** — live runtime evidence
3. **benchmarks** — narrow probe resolution, replay, artifacts, comparison
4. **presentation** — CLI and MCP surfaces

That means benchmark matrix discovery, suite planning, and stack materialization are not core InferScope responsibilities.

## Root ownership

The repository root owns only monorepo-level surfaces:

- landing docs
- architecture and contribution guidance
- validation entrypoints
- CI wiring
- top-level Make targets

It does not own product logic.

## Donor harness rule

`inferscope-bench/` is a donor harness only.

- it is not a product
- it should not be modified for feature work
- ideas may be absorbed into product code, but the donor tree itself is not a supported surface

## Design rule

Keep the separation hard:

- benchmark breadth and benchmark-standard infrastructure belong in **ISB-1**
- deployment-specific diagnostics and remediation workflows belong in **InferScope**

That separation is what keeps the monorepo strategically coherent.
