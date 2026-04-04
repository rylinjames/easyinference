"""Engine registry — maps engine names to compilers and adapters."""

from __future__ import annotations

from inferscope.engines.atom import ATOMAdapter, ATOMCompiler
from inferscope.engines.base import ConfigCompiler, EngineAdapter
from inferscope.engines.dynamo import DynamoAdapter, DynamoCompiler
from inferscope.engines.sglang import SGLangAdapter, SGLangCompiler
from inferscope.engines.trtllm import TRTLLMAdapter, TRTLLMCompiler
from inferscope.engines.vllm import VLLMAdapter, VLLMCompiler

_COMPILERS: dict[str, type[ConfigCompiler]] = {
    "atom": ATOMCompiler,
    "dynamo": DynamoCompiler,
    "sglang": SGLangCompiler,
    "trtllm": TRTLLMCompiler,
    "vllm": VLLMCompiler,
}

_ADAPTERS: dict[str, type[EngineAdapter]] = {
    "atom": ATOMAdapter,
    "dynamo": DynamoAdapter,
    "sglang": SGLangAdapter,
    "trtllm": TRTLLMAdapter,
    "vllm": VLLMAdapter,
}


def get_compiler(engine: str) -> ConfigCompiler:
    """Get a config compiler instance for the given engine."""
    cls = _COMPILERS.get(engine)
    if cls is None:
        raise ValueError(f"Unknown engine: {engine}. Available: {list(_COMPILERS.keys())}")
    return cls()


def get_engine_adapter(engine: str) -> EngineAdapter:
    """Get an engine adapter instance for the given engine."""
    cls = _ADAPTERS.get(engine)
    if cls is None:
        raise ValueError(f"Unknown engine: {engine}. Available: {list(_ADAPTERS.keys())}")
    return cls()


def all_adapters() -> list[EngineAdapter]:
    """Get instances of all engine adapters for auto-detection."""
    return [cls() for cls in _ADAPTERS.values()]
