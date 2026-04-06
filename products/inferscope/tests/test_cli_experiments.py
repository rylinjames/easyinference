from __future__ import annotations

import types
from pathlib import Path

from typer.testing import CliRunner

from inferscope.cli import app
from inferscope import cli_experiments


runner = CliRunner()


class _FakeSeries:
    def __init__(self) -> None:
        self.values: list[tuple[object, int]] = []

    def append(self, value: object, step: int = 0) -> None:
        self.values.append((value, step))


class _FakeExperiment:
    def __init__(self) -> None:
        self.url = "https://lightning.ai/test/exp"
        self.series: dict[str, _FakeSeries] = {}
        self.logged_files: list[str] = []
        self.status: str | None = None

    def __getitem__(self, key: str) -> _FakeSeries:
        return self.series.setdefault(key, _FakeSeries())

    def log_file(self, path: str) -> None:
        self.logged_files.append(path)

    def finalize(self, status: str) -> None:
        self.status = status


class _FakeLitLogger:
    def __init__(self) -> None:
        self.experiment = _FakeExperiment()

    def init(self, **_: object) -> _FakeExperiment:
        return self.experiment


class _FakeModelDump:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return self.payload


class _FakePreflight:
    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": ["Artifact manifest loaded"],
        }


class _FakeRunPlan:
    def __init__(self) -> None:
        self.preflight_validation = _FakePreflight()

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {"concurrency": 2, "preflight_validation": {"valid": True}}


class _FakeSummary:
    total_requests = 10
    succeeded = 10
    failed = 0
    concurrency = 2
    wall_time_ms = 1500.0
    latency_avg_ms = 100.0
    latency_p50_ms = 90.0
    latency_p95_ms = 120.0
    latency_p99_ms = 140.0
    ttft_avg_ms = 50.0
    ttft_p90_ms = 55.0
    ttft_p95_ms = 60.0
    ttft_p99_ms = 65.0
    prompt_tokens = 1000
    completion_tokens = 250
    total_tokens = 1250
    metrics_targets_total = 1
    metrics_targets_with_errors = 0
    metrics_capture_complete = True

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {
            "total_requests": self.total_requests,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "latency_p95_ms": self.latency_p95_ms,
            "metrics_capture_complete": self.metrics_capture_complete,
        }


class _FakeArtifact:
    def __init__(self) -> None:
        self.summary = _FakeSummary()
        self.default_filename = "artifact.json"
        self.run_plan = {"observed_runtime": {"reliability": {}, "observability": {}}}

    def save_json(self, path: str | Path) -> Path:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("{\"ok\": true}")
        return file_path

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {"summary": self.summary.model_dump(mode="json")}


def test_experiment_run_requires_litlogger(monkeypatch) -> None:
    def _raise_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(cli_experiments.importlib, "import_module", _raise_import)

    result = runner.invoke(app, ["experiment-run", "http://localhost:8000"])

    assert result.exit_code == 2
    assert "litlogger is not installed" in result.output


def test_experiment_run_logs_profile_plan_and_benchmark(monkeypatch) -> None:
    fake_litlogger = _FakeLitLogger()

    async def _fake_profile_runtime(endpoint: str, **_: object) -> dict[str, object]:
        assert endpoint == "http://localhost:8000"
        return {"summary": "profiled", "confidence": 0.92}

    def _fake_resolve_probe_plan(*args: object, **kwargs: object) -> object:
        assert args[0] == "kimi-k2-long-context-coding"
        assert kwargs["model_artifact_path"] == "artifacts/model"
        assert kwargs["artifact_manifest"] == "artifact-manifest-example.yaml"
        return types.SimpleNamespace(
            workload_pack="pack",
            workload_reference="kimi-k2-long-context-coding",
            run_plan=_FakeRunPlan(),
            support=_FakeModelDump({"status": "supported"}),
        )

    async def _fake_run_openai_replay(*args: object, **_: object) -> _FakeArtifact:
        assert args[1] == "http://localhost:8000"
        return _FakeArtifact()

    monkeypatch.setattr(cli_experiments.importlib, "import_module", lambda name: fake_litlogger)
    monkeypatch.setattr(cli_experiments, "profile_runtime", _fake_profile_runtime)
    monkeypatch.setattr(cli_experiments, "resolve_probe_plan", _fake_resolve_probe_plan)
    monkeypatch.setattr(cli_experiments, "run_openai_replay", _fake_run_openai_replay)
    monkeypatch.setattr(cli_experiments, "build_default_artifact_path", lambda artifact: Path("artifacts") / artifact.default_filename)

    with runner.isolated_filesystem():
        Path("artifact-manifest-example.yaml").write_text(
            "schema_version: '1'\nmodel: Qwen2.5-7B-Instruct\nartifact_kind: huggingface_weights\n"
        )
        result = runner.invoke(
            app,
            [
                "experiment-run",
                "http://localhost:8000",
                "--teamspace",
                "easyinference-evaluation-project",
                "--benchmark",
                "--model-artifact-path",
                "artifacts/model",
                "--artifact-manifest",
                "artifact-manifest-example.yaml",
            ],
        )

        assert result.exit_code == 0
        assert "Logged InferScope run to Lightning experiment" in result.output
        assert "https://lightning.ai/test/exp" in result.output
        assert Path("lightning_logs").exists()
        assert fake_litlogger.experiment.status == "success"
        assert any(path.endswith("profile-runtime.json") for path in fake_litlogger.experiment.logged_files)
        assert any(path.endswith("benchmark-plan.json") for path in fake_litlogger.experiment.logged_files)
        assert any(path.endswith("artifact.json") for path in fake_litlogger.experiment.logged_files)
        assert any(path.endswith("artifact-manifest-example.yaml") for path in fake_litlogger.experiment.logged_files)
        assert '"preflight_validation"' in result.output
        assert '"valid": true' in result.output
        assert '"production_readiness"' in result.output
        assert '"ready": true' in result.output
