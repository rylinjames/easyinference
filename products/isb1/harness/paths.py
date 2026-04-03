"""Filesystem paths for the ISB-1 benchmark product."""

from __future__ import annotations

from pathlib import Path

_PRODUCT_ROOT = Path(__file__).resolve().parents[1]
_PRODUCT_LOCAL_PREFIXES = {"configs", "results", "publication", "lockfiles", "traces"}


def product_root() -> Path:
    """Return the filesystem root of the ISB-1 product."""
    return _PRODUCT_ROOT


def default_config_root() -> Path:
    """Return the default configuration directory for benchmark assets."""
    return _PRODUCT_ROOT / "configs"


def default_results_root() -> Path:
    """Return the default results directory for benchmark outputs."""
    return _PRODUCT_ROOT / "results"


def default_traces_root() -> Path:
    """Return the default trace directory for generated benchmark request pools."""
    return _PRODUCT_ROOT / "traces"


def default_publication_root() -> Path:
    """Return the default publication directory for generated artifacts."""
    return _PRODUCT_ROOT / "publication"


def default_lockfiles_root() -> Path:
    """Return the default lockfile directory for reproducibility snapshots."""
    return _PRODUCT_ROOT / "lockfiles"


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path for benchmark CLI usage.

    Resolution order:
    1. Absolute paths are preserved.
    2. Existing paths under the caller's current working directory win.
    3. Known benchmark-local prefixes (for example ``configs/...`` or
       ``results/...``) resolve relative to the ISB-1 product root.
    4. Existing paths under the product root are accepted.
    5. Otherwise the path is resolved relative to the caller's CWD.
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = (Path.cwd() / candidate).resolve()
    product_candidate = (_PRODUCT_ROOT / candidate).resolve()

    if cwd_candidate.exists():
        return cwd_candidate

    if candidate.parts and candidate.parts[0] in _PRODUCT_LOCAL_PREFIXES:
        return product_candidate

    if product_candidate.exists():
        return product_candidate

    return cwd_candidate


def resolve_existing_path(path: str | Path) -> Path:
    """Resolve a path and require that it already exists."""

    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return resolved
