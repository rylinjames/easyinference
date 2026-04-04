"""Regression tests for the engine registry."""

from __future__ import annotations

import pytest

from inferscope.engines.atom import ATOMAdapter, ATOMCompiler
from inferscope.engines.dynamo import DynamoAdapter, DynamoCompiler
from inferscope.engines.registry import get_compiler, get_engine_adapter
from inferscope.engines.sglang import SGLangAdapter, SGLangCompiler
from inferscope.engines.trtllm import TRTLLMAdapter, TRTLLMCompiler
from inferscope.engines.vllm import VLLMAdapter, VLLMCompiler


@pytest.mark.parametrize(
    ("engine", "compiler_type", "adapter_type"),
    [
        ("atom", ATOMCompiler, ATOMAdapter),
        ("dynamo", DynamoCompiler, DynamoAdapter),
        ("sglang", SGLangCompiler, SGLangAdapter),
        ("trtllm", TRTLLMCompiler, TRTLLMAdapter),
        ("vllm", VLLMCompiler, VLLMAdapter),
    ],
)
def test_engine_registry_exposes_all_supported_engines(
    engine: str,
    compiler_type: type,
    adapter_type: type,
) -> None:
    assert isinstance(get_compiler(engine), compiler_type)
    assert isinstance(get_engine_adapter(engine), adapter_type)


def test_engine_registry_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="Unknown engine: madeup"):
        get_compiler("madeup")

    with pytest.raises(ValueError, match="Unknown engine: madeup"):
        get_engine_adapter("madeup")
