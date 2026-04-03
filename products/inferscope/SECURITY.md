# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ Current release |

## Reporting a Vulnerability

If you discover a security vulnerability in InferScope, please report it responsibly.

**DO NOT** open a public GitHub issue for security vulnerabilities.

Instead, please email: **security@inferscope.dev** (or open a private security advisory on GitHub).

We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Security Model

InferScope operates in two modes with different trust boundaries:

### Read-only mode (default)

- All MCP tools and CLI commands are **read-only** by default
- No config changes are applied automatically
- No shell commands are executed
- File system writes are limited to benchmark/cache output under `~/.inferscope/` by default
- Runtime profiles are not persisted to disk by default in v1

### Network access

- Telemetry scraping and runtime identity enrichment make outbound HTTP requests to user-specified endpoints
- Endpoints are validated and restricted to HTTP/HTTPS schemes
- **MCP** runtime tools block private IP ranges by default to reduce SSRF exposure
- **CLI** runtime tools allow private and localhost endpoints because the operator is expected to run them near the target system
- The MCP server binds to `127.0.0.1` (localhost only) by default
- Runtime profiling uses the same network policy for `/metrics` and best-effort `/v1/models` enrichment

### Input validation

- All user inputs (model names, GPU names, endpoints) are validated against known registries or sanitized
- URL endpoints are validated for scheme and host before any HTTP request
- No user input is passed to shell commands or `eval()`
- MCP benchmark tools restrict artifact paths to the configured benchmark directory

## Known Limitations

- The HTTP transport (`--transport streamable-http`) exposes an HTTP server. Use a reverse proxy with authentication in production.
- GPU telemetry endpoints (NVIDIA DCGM port 9400, AMD DME port 5000) are assumed to be on a trusted network. Both NVIDIA and AMD GPU targets are day-one supported.
- Runtime identity enrichment via `/v1/models` is best-effort and may lag the exact Prometheus snapshot in time.
- A future empirical profile store may contain deployment fingerprints — treat it as sensitive if persistence is added later.
