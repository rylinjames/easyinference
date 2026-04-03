.PHONY: setup validate run-core run-core-dry run-extended analyze leaderboard report quality clean test \
	isb1-setup isb1-validate isb1-lint isb1-format-check isb1-run-core isb1-run-core-dry isb1-run-extended isb1-analyze isb1-leaderboard isb1-report isb1-quality isb1-clean isb1-test \
	inferscope-lint inferscope-typecheck inferscope-security inferscope-test inferscope-package-smoke all-checks

ISB1_DIR := products/isb1
INFERSCOPE_DIR := products/inferscope

setup: isb1-setup
validate: isb1-validate
test: isb1-test
run-core: isb1-run-core
run-core-dry: isb1-run-core-dry
run-extended: isb1-run-extended
analyze: isb1-analyze
leaderboard: isb1-leaderboard
report: isb1-report
quality: isb1-quality
clean: isb1-clean

isb1-setup:
	$(MAKE) -C $(ISB1_DIR) setup

isb1-validate:
	$(MAKE) -C $(ISB1_DIR) validate

isb1-lint:
	$(MAKE) -C $(ISB1_DIR) lint

isb1-format-check:
	$(MAKE) -C $(ISB1_DIR) format-check

isb1-run-core:
	$(MAKE) -C $(ISB1_DIR) run-core

isb1-run-core-dry:
	$(MAKE) -C $(ISB1_DIR) run-core-dry

isb1-run-extended:
	$(MAKE) -C $(ISB1_DIR) run-extended

isb1-analyze:
	$(MAKE) -C $(ISB1_DIR) analyze

isb1-leaderboard:
	$(MAKE) -C $(ISB1_DIR) leaderboard

isb1-report:
	$(MAKE) -C $(ISB1_DIR) report

isb1-quality:
	$(MAKE) -C $(ISB1_DIR) quality

isb1-clean:
	$(MAKE) -C $(ISB1_DIR) clean

isb1-test:
	$(MAKE) -C $(ISB1_DIR) test

inferscope-lint:
	cd $(INFERSCOPE_DIR) && uv run ruff check src/
	cd $(INFERSCOPE_DIR) && if [ -d tests ]; then uv run ruff check tests/; fi
	cd $(INFERSCOPE_DIR) && uv run ruff format --check src/
	cd $(INFERSCOPE_DIR) && if [ -d tests ]; then uv run ruff format --check tests/; fi

inferscope-typecheck:
	cd $(INFERSCOPE_DIR) && uv run mypy src/inferscope/

inferscope-security:
	cd $(INFERSCOPE_DIR) && uv run bandit -r src/inferscope/ -c pyproject.toml -ll

inferscope-test:
	cd $(INFERSCOPE_DIR) && if [ -d tests ]; then uv run pytest tests/ -v --tb=short; else echo "Skipping pytest — no tests/ directory present"; fi

inferscope-package-smoke:
	cd $(INFERSCOPE_DIR) && rm -rf .venv-smoke dist && uv python install 3.12 && uv build && uv venv .venv-smoke --python 3.12 && uv pip install --python .venv-smoke/bin/python dist/*.whl && .venv-smoke/bin/inferscope benchmark-workloads && .venv-smoke/bin/inferscope benchmark-experiments && .venv-smoke/bin/python -c "from inferscope.benchmarks import load_experiment, load_workload; assert load_workload('coding-long-context').name == 'coding-long-context'; assert load_experiment('vllm-single-endpoint-baseline').name == 'vllm-single-endpoint-baseline'; assert load_workload('benchmarks/workloads/coding-long-context.yaml').name == 'coding-long-context'; assert load_experiment('benchmarks/experiment_specs/vllm-single-endpoint-baseline.yaml').name == 'vllm-single-endpoint-baseline'"

all-checks: isb1-lint isb1-format-check isb1-test inferscope-lint inferscope-typecheck inferscope-security inferscope-package-smoke inferscope-test
