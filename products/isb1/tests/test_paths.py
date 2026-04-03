"""Tests for benchmark product path defaults."""

from pathlib import Path

from harness.config_validator import ConfigValidator
from harness.paths import default_config_root, default_results_root, product_root
from harness.runner import CellConfig


def test_product_root_contains_expected_directories() -> None:
    root = product_root()
    assert root.name == "isb1"
    assert (root / "configs").is_dir()
    assert (root / "results").is_dir()


def test_default_config_root_matches_product_layout() -> None:
    assert default_config_root() == product_root() / "configs"
    assert (default_config_root() / "sweep" / "core.yaml").exists()


def test_default_results_root_matches_product_layout() -> None:
    assert default_results_root() == product_root() / "results"


def test_config_validator_uses_product_default_config_root() -> None:
    validator = ConfigValidator()
    assert validator.config_root == default_config_root().resolve()


def test_cell_config_uses_product_default_paths() -> None:
    cell = CellConfig()
    assert Path(cell.config_root) == default_config_root()
    assert Path(cell.output_dir) == default_results_root()
