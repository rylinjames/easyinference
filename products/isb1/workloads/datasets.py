"""HuggingFace dataset downloaders for ISB-1 real-world workloads.

Downloads and caches:
- SWE-bench_Verified: Real GitHub issue → code change tasks
- CoderForge-Preview: Agent trajectories with tool calls
- CodeSearchNet: Code+doc pairs for context padding

All datasets are cached in ``~/.cache/isb1/`` and loaded lazily on
first use.  Raises ``RuntimeError`` if a dataset cannot be loaded —
no synthetic fallbacks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path.home() / ".cache" / "isb1"


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


# ---------------------------------------------------------------------------
# SWE-bench_Verified
# ---------------------------------------------------------------------------

_SWEBENCH_CACHE = "swebench_verified.json"


def load_swebench_verified() -> list[dict[str, Any]]:
    """Load SWE-bench_Verified dataset, downloading if needed.

    Returns a list of dicts with keys:
        instance_id, repo, problem_statement, patch, base_commit,
        hints_text, created_at, version, FAIL_TO_PASS, PASS_TO_PASS
    """
    cache = _ensure_cache_dir() / _SWEBENCH_CACHE
    if cache.is_file():
        logger.info("Using cached SWE-bench_Verified at %s", cache)
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading SWE-bench_Verified from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
        records = [dict(row) for row in ds]
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, default=str)
        tmp.rename(cache)
        logger.info("Cached %d SWE-bench instances to %s", len(records), cache)
        return records
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SWE-bench_Verified. "
            "Install the datasets extra: pip install 'isb1[datasets]'  "
            "and ensure network access to HuggingFace Hub."
        ) from exc


# ---------------------------------------------------------------------------
# CoderForge-Preview
# ---------------------------------------------------------------------------

_CODERFORGE_CACHE = "coderforge_preview.json"


def load_coderforge_preview() -> list[dict[str, Any]]:
    """Load CoderForge-Preview agent trajectories, downloading if needed.

    Returns a list of dicts with agent trajectory fields.
    """
    cache = _ensure_cache_dir() / _CODERFORGE_CACHE
    if cache.is_file():
        logger.info("Using cached CoderForge-Preview at %s", cache)
        with open(cache, encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading CoderForge-Preview from HuggingFace...")
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]

        ds = load_dataset("togethercomputer/CoderForge-Preview", "trajectories", split="train")
        records = [dict(row) for row in ds]
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, default=str)
        tmp.rename(cache)
        logger.info("Cached %d CoderForge trajectories to %s", len(records), cache)
        return records
    except Exception as exc:
        raise RuntimeError(
            "Failed to load CoderForge-Preview. "
            "Install the datasets extra: pip install 'isb1[datasets]'  "
            "and ensure network access to HuggingFace Hub."
        ) from exc


# ---------------------------------------------------------------------------
# CodeSearchNet (context padding)
# ---------------------------------------------------------------------------

_CSN_CACHE_PREFIX = "codesearchnet"


def load_codesearchnet(
    languages: tuple[str, ...] = ("python", "javascript"),
    max_per_language: int = 50_000,
) -> dict[str, list[dict[str, Any]]]:
    """Load CodeSearchNet functions for context padding.

    Args:
        languages: Languages to load (default: Python + JavaScript).
        max_per_language: Cap per language to avoid huge cache files.

    Returns:
        ``{language: [{func_name, code, docstring, url, ...}]}``
    """
    result: dict[str, list[dict[str, Any]]] = {}
    cache_dir = _ensure_cache_dir()

    for lang in languages:
        cache_file = cache_dir / f"{_CSN_CACHE_PREFIX}_{lang}.json"
        if cache_file.is_file():
            logger.info("Using cached CodeSearchNet/%s at %s", lang, cache_file)
            with open(cache_file, encoding="utf-8") as f:
                result[lang] = json.load(f)
            continue

        logger.info("Downloading CodeSearchNet/%s from HuggingFace...", lang)
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]

            ds = load_dataset("code-search-net/code_search_net", lang, split="train")
            records = []
            for i, row in enumerate(ds):
                if i >= max_per_language:
                    break
                records.append({
                    "func_name": row.get("func_name", ""),
                    "code": row.get("whole_func_string", row.get("func_code_string", "")),
                    "docstring": row.get("func_documentation_string", ""),
                    "url": row.get("func_code_url", ""),
                    "language": lang,
                })
            tmp = cache_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
            tmp.rename(cache_file)
            logger.info("Cached %d CodeSearchNet/%s functions to %s", len(records), lang, cache_file)
            result[lang] = records
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load CodeSearchNet/{lang}. "
                "Install the datasets extra: pip install 'isb1[datasets]'  "
                "and ensure network access to HuggingFace Hub."
            ) from exc

    return result


# ---------------------------------------------------------------------------
# Context builder utility
# ---------------------------------------------------------------------------

_CONTEXT_BUCKET_TOKENS = {
    "8k": 8_000,
    "16k": 16_000,
    "32k": 32_000,
    "64k": 64_000,
    "128k": 128_000,
}

# Rough estimate: 1 token ≈ 4 chars for code
_CHARS_PER_TOKEN = 4


def build_repo_context(
    functions: list[dict[str, Any]],
    target_bucket: str,
    rng: Any,
) -> str:
    """Build a synthetic repo context block from CodeSearchNet functions.

    Samples functions until the target token budget is reached, formatting
    them as file-like blocks.

    Args:
        functions: CodeSearchNet function records.
        target_bucket: One of "8k", "16k", "32k", "64k", "128k".
        rng: numpy random generator for reproducible sampling.

    Returns:
        A string of ``--- path/to/file.py ---\\n<code>`` blocks.
    """
    target_tokens = _CONTEXT_BUCKET_TOKENS.get(target_bucket, 32_000)
    target_chars = target_tokens * _CHARS_PER_TOKEN

    if not functions:
        raise RuntimeError(
            "No CodeSearchNet functions available for context padding. "
            "Ensure load_codesearchnet() succeeded before calling build_repo_context()."
        )

    indices = rng.permutation(len(functions))
    blocks: list[str] = []
    total_chars = 0

    for idx in indices:
        func = functions[int(idx)]
        code = func.get("code", "")
        if not code.strip():
            continue
        name = func.get("func_name", "unknown")
        lang = func.get("language", "python")
        ext = ".py" if lang == "python" else ".js" if lang == "javascript" else f".{lang}"

        # Build a file-like block
        path = f"src/{name.replace('.', '/')}{ext}"
        block = f"--- {path} ---\n{code}\n"
        block_chars = len(block)

        if total_chars + block_chars > target_chars:
            break
        blocks.append(block)
        total_chars += block_chars

    return "\n".join(blocks) if blocks else "# (repository context)\n"
