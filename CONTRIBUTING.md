# Contributing to EasyInference

Contributions are routed by product.

## Pick the product first

- For benchmark work, use `products/isb1/`
- For MCP / optimization work, use `products/inferscope/`
- For root docs, workflows, or monorepo plumbing, keep changes explicit about which product contracts are affected

## Product-local guides

- ISB-1: [products/isb1/docs/CONTRIBUTING.md](products/isb1/docs/CONTRIBUTING.md)
- InferScope: [products/inferscope/CONTRIBUTING.md](products/inferscope/CONTRIBUTING.md)

## Common workflow

1. Branch from `main`
2. Keep scope tight
3. Update docs when public behavior changes
4. Run the relevant product validations before opening a PR
5. Call out compatibility, migration, or rollout impact in the PR description

## Root-level commands

### Benchmark

```bash
make validate
make isb1-lint
make isb1-format-check
make test
```

These are legacy aliases that delegate into `products/isb1/`.

### InferScope

```bash
make inferscope-lint
make inferscope-typecheck
make inferscope-security
make inferscope-package-smoke
make inferscope-test
```

### Full pass

```bash
make all-checks
```

## CI ownership

GitHub Actions is split by product:
- `.github/workflows/isb1-ci.yml`
- `.github/workflows/inferscope-ci.yml`

If your change touches only one product, run that product’s checks first.

## Licensing

This repository is multi-license:
- `products/isb1/` is Apache-2.0
- `products/inferscope/` is MIT

Be explicit if a root-level change affects one product more than the other.
