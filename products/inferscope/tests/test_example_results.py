"""Validation for checked-in example result fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from inferscope.benchmarks.catalog import compare_benchmark_artifacts, load_benchmark_artifact
from inferscope.benchmarks.preflight import BenchmarkArtifactManifest
from inferscope.profiling.models import RuntimeProfileReport


DOCS_EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "examples"


def test_runtime_profile_example_validates_against_runtime_profile_report() -> None:
    payload = json.loads((DOCS_EXAMPLES / "runtime-profile-example.json").read_text())
    report = RuntimeProfileReport.model_validate(payload)

    assert report.endpoint == "http://localhost:8000"
    assert report.identity is not None
    assert report.identity.engine == "dynamo"


def test_benchmark_example_artifacts_validate_and_compare() -> None:
    baseline = load_benchmark_artifact(DOCS_EXAMPLES / "benchmark-artifact-baseline.json")
    candidate = load_benchmark_artifact(DOCS_EXAMPLES / "benchmark-artifact-candidate.json")

    assert baseline.pack_name == "kimi-k2-long-context-coding"
    assert candidate.pack_name == "kimi-k2-long-context-coding"
    assert baseline.provenance is not None
    assert baseline.provenance.lane is not None
    assert baseline.provenance.lane.class_name == "production_validated"
    assert baseline.provenance.lane.production_target_name == "dynamo_long_context_coding"
    assert candidate.provenance is not None
    assert candidate.provenance.lane is not None
    assert (
        candidate.provenance.lane.experiment == "dynamo-aggregated-lmcache-kimi-k2"
    )

    comparison = compare_benchmark_artifacts(baseline, candidate)
    expected = json.loads((DOCS_EXAMPLES / "benchmark-comparison-example.json").read_text())

    assert comparison == expected


def test_artifact_manifest_example_validates_against_manifest_schema() -> None:
    manifest = BenchmarkArtifactManifest.from_file(DOCS_EXAMPLES / "artifact-manifest-example.yaml")

    assert manifest.schema_version == "1"
    assert manifest.artifact_kind == "huggingface_weights"
    assert manifest.model == "Qwen2.5-7B-Instruct"
    assert manifest.engine == "vllm"
    assert manifest.tensor_parallel_size == 1
    assert manifest.gpu_family == "ampere"


def test_live_smoke_exports_and_summaries_are_well_formed() -> None:
    lightning_runtime = json.loads((DOCS_EXAMPLES / "lightning-h100-live-smoke-runtime-profile.json").read_text())
    lightning_artifact = load_benchmark_artifact(
        DOCS_EXAMPLES / "lightning-h100-live-smoke-benchmark-artifact.json"
    )
    lightning_summary = json.loads((DOCS_EXAMPLES / "lightning-h100-live-smoke-summary.json").read_text())
    modal_summary = json.loads((DOCS_EXAMPLES / "modal-a10g-live-smoke-summary.json").read_text())
    production_summary = json.loads((DOCS_EXAMPLES / "kimi-dynamo-production-reference-summary.json").read_text())

    runtime_report = RuntimeProfileReport.model_validate(lightning_runtime)

    assert runtime_report.endpoint == "http://localhost:8000"
    assert lightning_artifact.pack_name == "kimi-k2-long-context-coding"
    assert lightning_artifact.provenance is not None
    assert lightning_artifact.provenance.lane is not None
    assert lightning_artifact.provenance.lane.class_name == "production_validated"
    assert lightning_summary["status"] == "authenticated-export-complete"
    assert lightning_summary["benchmark_summary"]["succeeded_requests"] == 2
    assert production_summary["production_target_name"] == "dynamo_long_context_coding"
    assert production_summary["lane"]["claim_scope"] == "production_comparable"
    assert modal_summary["platform"] == "modal"
    assert modal_summary["inferscope"]["workload_pack"] == "coding-smoke"
    assert modal_summary["isb1"]["simple_quick_bench"]["completed"] == 1
