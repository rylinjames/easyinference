"""Coding session trace generator for ISB-1 benchmarks.

Produces synthetic coding-assistant workload traces.  Each session
simulates a developer working within a repository: the context contains
a system prompt, 5-15 synthetic source files (Python, TypeScript, Rust,
Go), and a series of conversation turns.  High prefix reuse across turns
within a session reflects the pattern of real coding assistants where
the repository context is largely stable.
"""

from __future__ import annotations

from typing import Any


from workloads.base import Request, WorkloadGenerator, _new_request_id

# ---------------------------------------------------------------------------
# Language definitions: filenames, boilerplate, and snippet templates
# ---------------------------------------------------------------------------

_LANGUAGES: dict[str, dict[str, Any]] = {
    "python": {
        "extension": ".py",
        "dirs": ["src", "lib", "utils", "models", "services", "api", "tests"],
        "filenames": [
            "main", "config", "database", "auth", "router", "middleware",
            "schema", "validators", "cache", "worker", "metrics", "logger",
            "handlers", "serializers", "exceptions", "settings", "tasks",
        ],
        "snippets": [
            '''"""Module for handling {purpose}."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class {class_name}:
    """{class_doc}"""

    name: str
    config: dict[str, Any] = field(default_factory=dict)
    _initialized: bool = field(default=False, init=False, repr=False)

    def initialize(self) -> None:
        """Set up internal state and validate configuration."""
        if self._initialized:
            logger.warning("Already initialized, skipping")
            return
        self._validate_config()
        self._initialized = True
        logger.info("Initialized %s with config keys: %s", self.name, list(self.config.keys()))

    def _validate_config(self) -> None:
        required = {{"timeout", "max_retries", "endpoint"}}
        missing = required - set(self.config.keys())
        if missing:
            raise ValueError(f"Missing required config keys: {{missing}}")

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process incoming data and return transformed result."""
        if not self._initialized:
            raise RuntimeError("Must call initialize() before processing")
        result = {{
            "status": "success",
            "input_keys": list(data.keys()),
            "processed_by": self.name,
        }}
        logger.debug("Processed %d fields", len(data))
        return result
''',
            '''"""Utility functions for {purpose}."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable:
    """Decorator that retries a function with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(min(delay, max_delay))
                    delay *= 2.0
            raise RuntimeError("Unreachable")
        return wrapper
    return decorator


def compute_checksum(data: bytes, algorithm: str = "sha256") -> str:
    """Compute a hex digest checksum for the given data."""
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def chunk_list(items: list[T], size: int) -> list[list[T]]:
    """Split a list into chunks of the given size."""
    return [items[i : i + size] for i in range(0, len(items), size)]
''',
            '''"""Database access layer for {purpose}."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)


class ConnectionPool:
    """Manages a pool of database connections."""

    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20) -> None:
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._connections: list[Any] = []
        self._available: list[Any] = []

    @contextmanager
    def acquire(self) -> Generator[Any, None, None]:
        """Acquire a connection from the pool."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            self._release_connection(conn)

    def _get_connection(self) -> Any:
        if self._available:
            return self._available.pop()
        if len(self._connections) < self.max_size:
            conn = self._create_connection()
            self._connections.append(conn)
            return conn
        raise RuntimeError("Connection pool exhausted")

    def _create_connection(self) -> Any:
        logger.info("Creating new connection to %s", self.dsn)
        return {{"dsn": self.dsn, "active": True}}

    def _release_connection(self, conn: Any) -> None:
        self._available.append(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        for conn in self._connections:
            conn["active"] = False
        self._connections.clear()
        self._available.clear()
        logger.info("All connections closed")
''',
        ],
    },
    "typescript": {
        "extension": ".ts",
        "dirs": ["src", "lib", "components", "hooks", "services", "utils", "types"],
        "filenames": [
            "index", "app", "config", "router", "middleware", "auth",
            "api-client", "store", "types", "validators", "helpers",
            "constants", "logger", "cache", "event-bus",
        ],
        "snippets": [
            '''/**
 * {purpose} service implementation.
 */

interface {class_name}Config {{
  endpoint: string;
  apiKey: string;
  timeout: number;
  retryCount: number;
}}

interface RequestOptions {{
  method: "GET" | "POST" | "PUT" | "DELETE";
  path: string;
  body?: Record<string, unknown>;
  headers?: Record<string, string>;
}}

export class {class_name} {{
  private config: {class_name}Config;
  private baseUrl: string;

  constructor(config: {class_name}Config) {{
    this.config = config;
    this.baseUrl = config.endpoint.replace(/\\/$/, "");
  }}

  async request<T>(options: RequestOptions): Promise<T> {{
    const url = `${{this.baseUrl}}${{options.path}}`;
    const headers: Record<string, string> = {{
      "Content-Type": "application/json",
      Authorization: `Bearer ${{this.config.apiKey}}`,
      ...options.headers,
    }};

    const response = await fetch(url, {{
      method: options.method,
      headers,
      body: options.body ? JSON.stringify(options.body) : undefined,
      signal: AbortSignal.timeout(this.config.timeout),
    }});

    if (!response.ok) {{
      throw new Error(`Request failed: ${{response.status}} ${{response.statusText}}`);
    }}

    return response.json() as Promise<T>;
  }}

  async healthCheck(): Promise<boolean> {{
    try {{
      await this.request({{ method: "GET", path: "/health" }});
      return true;
    }} catch {{
      return false;
    }}
  }}
}}
''',
            '''/**
 * Utility functions for {purpose}.
 */

export type Predicate<T> = (item: T) => boolean;

export function groupBy<T, K extends string | number>(
  items: T[],
  keyFn: (item: T) => K,
): Record<K, T[]> {{
  const result = {{}} as Record<K, T[]>;
  for (const item of items) {{
    const key = keyFn(item);
    if (!result[key]) {{
      result[key] = [];
    }}
    result[key].push(item);
  }}
  return result;
}}

export function debounce<T extends (...args: unknown[]) => void>(
  fn: T,
  delayMs: number,
): T {{
  let timeoutId: ReturnType<typeof setTimeout> | null = null;
  return ((...args: unknown[]) => {{
    if (timeoutId !== null) {{
      clearTimeout(timeoutId);
    }}
    timeoutId = setTimeout(() => fn(...args), delayMs);
  }}) as T;
}}

export function deepClone<T>(obj: T): T {{
  return JSON.parse(JSON.stringify(obj));
}}

export function formatBytes(bytes: number, decimals = 2): string {{
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${{parseFloat((bytes / k ** i).toFixed(decimals))}} ${{sizes[i]}}`;
}}
''',
        ],
    },
    "rust": {
        "extension": ".rs",
        "dirs": ["src", "src/handlers", "src/models", "src/services", "src/utils"],
        "filenames": [
            "main", "lib", "config", "error", "router", "middleware",
            "auth", "database", "cache", "worker", "metrics", "schema",
        ],
        "snippets": [
            '''//! {purpose} implementation.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone)]
pub struct {class_name} {{
    name: String,
    config: HashMap<String, String>,
    cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}}

impl {class_name} {{
    pub fn new(name: impl Into<String>, config: HashMap<String, String>) -> Self {{
        Self {{
            name: name.into(),
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }}
    }}

    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {{
        let cache = self.cache.read().await;
        cache.get(key).cloned()
    }}

    pub async fn set(&self, key: String, value: Vec<u8>) {{
        let mut cache = self.cache.write().await;
        cache.insert(key, value);
    }}

    pub async fn invalidate(&self, key: &str) -> bool {{
        let mut cache = self.cache.write().await;
        cache.remove(key).is_some()
    }}

    pub fn config_value(&self, key: &str) -> Option<&str> {{
        self.config.get(key).map(String::as_str)
    }}
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn test_cache_roundtrip() {{
        let svc = {class_name}::new("test", HashMap::new());
        svc.set("key1".into(), b"value1".to_vec()).await;
        assert_eq!(svc.get("key1").await, Some(b"value1".to_vec()));
        assert!(svc.invalidate("key1").await);
        assert_eq!(svc.get("key1").await, None);
    }}
}}
''',
            '''//! Error types for {purpose}.

use std::fmt;

#[derive(Debug)]
pub enum AppError {{
    NotFound(String),
    Unauthorized(String),
    BadRequest(String),
    Internal(String),
    Timeout {{ operation: String, duration_ms: u64 }},
}}

impl fmt::Display for AppError {{
    fn fmt(&self, f: &mut fmt::Formatter<\'_>) -> fmt::Result {{
        match self {{
            Self::NotFound(msg) => write!(f, "not found: {{msg}}"),
            Self::Unauthorized(msg) => write!(f, "unauthorized: {{msg}}"),
            Self::BadRequest(msg) => write!(f, "bad request: {{msg}}"),
            Self::Internal(msg) => write!(f, "internal error: {{msg}}"),
            Self::Timeout {{ operation, duration_ms }} => {{
                write!(f, "timeout after {{duration_ms}}ms in {{operation}}")
            }}
        }}
    }}
}}

impl std::error::Error for AppError {{}}

pub type Result<T> = std::result::Result<T, AppError>;
''',
        ],
    },
    "go": {
        "extension": ".go",
        "dirs": ["cmd", "internal", "pkg", "internal/handlers", "internal/models"],
        "filenames": [
            "main", "config", "server", "router", "middleware", "auth",
            "database", "cache", "worker", "metrics", "logger", "errors",
        ],
        "snippets": [
            '''// Package {pkg} provides {purpose} functionality.
package {pkg}

import (
\t"context"
\t"fmt"
\t"log"
\t"sync"
\t"time"
)

// {class_name} manages {purpose} operations.
type {class_name} struct {{
\tmu      sync.RWMutex
\tname    string
\tconfig  map[string]string
\tstarted bool
}}

// New{class_name} creates a new instance with the given configuration.
func New{class_name}(name string, config map[string]string) *{class_name} {{
\treturn &{class_name}{{
\t\tname:   name,
\t\tconfig: config,
\t}}
}}

// Start initialises the service and begins processing.
func (s *{class_name}) Start(ctx context.Context) error {{
\ts.mu.Lock()
\tdefer s.mu.Unlock()

\tif s.started {{
\t\treturn fmt.Errorf("%s: already started", s.name)
\t}}

\tlog.Printf("[%s] starting with %d config entries", s.name, len(s.config))
\ts.started = true
\treturn nil
}}

// Stop gracefully shuts down the service.
func (s *{class_name}) Stop(ctx context.Context) error {{
\ts.mu.Lock()
\tdefer s.mu.Unlock()

\tif !s.started {{
\t\treturn nil
\t}}

\tlog.Printf("[%s] shutting down", s.name)
\ts.started = false
\treturn nil
}}

// Process handles a single work item with the given timeout.
func (s *{class_name}) Process(ctx context.Context, data map[string]interface{{}}) (map[string]interface{{}}, error) {{
\ts.mu.RLock()
\tdefer s.mu.RUnlock()

\tif !s.started {{
\t\treturn nil, fmt.Errorf("%s: not started", s.name)
\t}}

\tctx, cancel := context.WithTimeout(ctx, 30*time.Second)
\tdefer cancel()

\t_ = ctx // use context for downstream calls
\tresult := map[string]interface{{}}{{
\t\t"status":       "processed",
\t\t"processed_by": s.name,
\t\t"input_keys":   len(data),
\t}}
\treturn result, nil
}}
''',
            '''// Package {pkg} provides {purpose} utilities.
package {pkg}

import (
\t"crypto/sha256"
\t"encoding/hex"
\t"strings"
)

// Chunk splits a slice into groups of at most size elements.
func Chunk[T any](items []T, size int) [][]T {{
\tif size <= 0 {{
\t\treturn nil
\t}}
\tvar chunks [][]T
\tfor i := 0; i < len(items); i += size {{
\t\tend := i + size
\t\tif end > len(items) {{
\t\t\tend = len(items)
\t\t}}
\t\tchunks = append(chunks, items[i:end])
\t}}
\treturn chunks
}}

// HashString returns the SHA-256 hex digest of a string.
func HashString(s string) string {{
\th := sha256.Sum256([]byte(s))
\treturn hex.EncodeToString(h[:])
}}

// TruncateString shortens a string to maxLen, adding an ellipsis if truncated.
func TruncateString(s string, maxLen int) string {{
\tif len(s) <= maxLen {{
\t\treturn s
\t}}
\tif maxLen <= 3 {{
\t\treturn s[:maxLen]
\t}}
\treturn s[:maxLen-3] + "..."
}}

// JoinNonEmpty joins non-empty strings with the given separator.
func JoinNonEmpty(parts []string, sep string) string {{
\tvar filtered []string
\tfor _, p := range parts {{
\t\tif strings.TrimSpace(p) != "" {{
\t\t\tfiltered = append(filtered, p)
\t\t}}
\t}}
\treturn strings.Join(filtered, sep)
}}
''',
        ],
    },
}

# ---------------------------------------------------------------------------
# Purposes and class names for template filling
# ---------------------------------------------------------------------------

_PURPOSES = [
    "request routing and dispatch",
    "user authentication and session management",
    "data validation and sanitisation",
    "background job scheduling",
    "metrics collection and aggregation",
    "cache management and invalidation",
    "configuration loading and validation",
    "event publishing and subscription",
    "rate limiting and throttling",
    "file storage and retrieval",
    "notification delivery",
    "search indexing and querying",
]

_CLASS_NAMES = [
    "RequestHandler", "SessionManager", "DataValidator", "JobScheduler",
    "MetricsCollector", "CacheManager", "ConfigLoader", "EventBus",
    "RateLimiter", "StorageService", "NotificationService", "SearchIndex",
    "ConnectionPool", "WorkerPool", "TaskQueue", "StateManager",
]

_CLASS_DOCS = [
    "Manages the lifecycle and processing of incoming requests.",
    "Handles creation, validation, and expiration of user sessions.",
    "Validates and sanitises input data against defined schemas.",
    "Schedules and manages background processing tasks.",
    "Collects, aggregates, and exports application metrics.",
    "Provides a caching layer with configurable eviction policies.",
]

# ---------------------------------------------------------------------------
# Coding user query templates
# ---------------------------------------------------------------------------

_CODING_QUERIES_INITIAL = [
    "Can you help me add error handling to the {filename} module? It currently doesn't handle timeout errors properly.",
    "I need to refactor the {class_name} class in {filename} to support dependency injection.",
    "There's a bug in {filename} where the connection pool doesn't release connections on error. Can you fix it?",
    "Please add unit tests for the {class_name} class in {filename}.",
    "I want to add a new method to {class_name} that supports batch processing. Here's the signature I'm thinking of.",
    "Can you review {filename} and suggest performance improvements?",
    "I need to add logging throughout {filename}. What's the best approach?",
    "Help me implement a retry mechanism with exponential backoff in {filename}.",
    "The {class_name} in {filename} needs to be made thread-safe. Can you help?",
    "I want to add configuration validation to {filename}. What fields should be required?",
]

_CODING_FOLLOWUPS = [
    "That looks good. Can you also update the imports and make sure the types are correct?",
    "Almost there, but I think we need to handle the edge case where the input is empty.",
    "Can you add documentation comments to the new methods?",
    "Good, now apply the same pattern to the {other_filename} file.",
    "What about error propagation? Should we wrap these errors or pass them through?",
    "Can you show me how to write a test for the new batch processing method?",
    "I think we also need to update the configuration schema. Can you check?",
    "The linter is complaining about unused imports. Can you clean those up?",
    "How would this change if we needed to support concurrent access?",
    "Can you extract the common logic into a shared utility function?",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_CODING_SYSTEM_PROMPT = (
    "You are an expert software engineering assistant. You have access to the "
    "user's repository context shown below. When providing code changes, show "
    "complete, working implementations. Follow the existing code style and "
    "conventions. Explain your reasoning briefly before showing code. When "
    "fixing bugs, identify the root cause. When refactoring, explain the "
    "trade-offs."
)

# ---------------------------------------------------------------------------
# Project name templates
# ---------------------------------------------------------------------------

_PROJECT_NAMES = [
    "acme-platform", "data-pipeline", "api-gateway", "auth-service",
    "event-processor", "metrics-dashboard", "config-manager",
    "search-engine", "notification-hub", "task-scheduler",
]


class CodingTraceGenerator(WorkloadGenerator):
    """Generate synthetic coding-assistant workload traces.

    Each session simulates a developer working within a repository.  The
    context contains a system prompt, 5-15 synthetic source files in
    Python / TypeScript / Rust / Go, and a series of conversation turns.
    Repository context is shared across turns within a session,
    producing high prefix reuse.

    Parameters:
        seed: Random seed for reproducibility.
        min_files: Minimum source files per session (default 5).
        max_files: Maximum source files per session (default 15).
        min_turns: Minimum conversation turns per session (default 2).
        max_turns: Maximum conversation turns per session (default 6).
    """

    def __init__(
        self,
        seed: int = 42,
        min_files: int = 5,
        max_files: int = 15,
        min_turns: int = 2,
        max_turns: int = 6,
    ) -> None:
        super().__init__(seed=seed)
        self.min_files = min_files
        self.max_files = max_files
        self.min_turns = min_turns
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # File generation
    # ------------------------------------------------------------------

    def _generate_file(self, lang_key: str) -> tuple[str, str]:
        """Generate a synthetic source file and return (filepath, content)."""
        lang = _LANGUAGES[lang_key]
        directory = lang["dirs"][int(self.rng.integers(0, len(lang["dirs"])))]
        basename = lang["filenames"][int(self.rng.integers(0, len(lang["filenames"])))]
        filepath = f"{directory}/{basename}{lang['extension']}"

        snippet_template = lang["snippets"][
            int(self.rng.integers(0, len(lang["snippets"])))
        ]
        purpose = _PURPOSES[int(self.rng.integers(0, len(_PURPOSES)))]
        class_name = _CLASS_NAMES[int(self.rng.integers(0, len(_CLASS_NAMES)))]
        class_doc = _CLASS_DOCS[int(self.rng.integers(0, len(_CLASS_DOCS)))]

        # For Go, we need a package name
        pkg = directory.split("/")[-1]
        if pkg in ("cmd", "internal", "pkg"):
            pkg = "main" if pkg == "cmd" else basename

        content = snippet_template.format(
            purpose=purpose,
            class_name=class_name,
            class_doc=class_doc,
            pkg=pkg,
        )
        return filepath, content

    def _generate_repo_context(self, num_files: int) -> tuple[str, list[str], list[str]]:
        """Generate a synthetic repository context.

        Returns:
            A tuple of (context_string, filenames, class_names) where
            context_string is the formatted repository content.
        """
        project = _PROJECT_NAMES[int(self.rng.integers(0, len(_PROJECT_NAMES)))]
        lang_keys = list(_LANGUAGES.keys())

        # Pick a primary language (60 %) with others mixed in
        primary = lang_keys[int(self.rng.integers(0, len(lang_keys)))]

        files: list[tuple[str, str]] = []
        filenames: list[str] = []
        class_names_used: list[str] = []
        seen_paths: set[str] = set()

        for _ in range(num_files):
            # 60 % primary language, 40 % random
            if self.rng.random() < 0.6:
                lang_key = primary
            else:
                lang_key = lang_keys[int(self.rng.integers(0, len(lang_keys)))]

            filepath, content = self._generate_file(lang_key)

            # Ensure unique paths
            while filepath in seen_paths:
                base, ext = filepath.rsplit(".", 1)
                filepath = f"{base}_{int(self.rng.integers(0, 999))}.{ext}"

            seen_paths.add(filepath)
            files.append((filepath, content))
            filenames.append(filepath)

            # Extract class name from content if present
            for cn in _CLASS_NAMES:
                if cn in content:
                    class_names_used.append(cn)
                    break

        # Format as repository context
        parts = [f"Repository: {project}\n"]
        for fp, content in files:
            parts.append(f"--- {fp} ---\n{content}")

        return "\n".join(parts), filenames, class_names_used

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, num_requests: int) -> list[Request]:
        """Generate *num_requests* coding-assistant requests.

        Each session shares a repository context (high prefix reuse).
        Multiple conversation turns build on the same file set.

        Args:
            num_requests: Target number of requests to produce.

        Returns:
            A list of :class:`Request` in session-sequential order.
        """
        requests: list[Request] = []

        while len(requests) < num_requests:
            session_requests = self._generate_session()
            requests.extend(session_requests)

        return requests[:num_requests]

    def _generate_session(self) -> list[Request]:
        """Create a single coding session with multiple turns."""
        num_files = int(self.rng.integers(self.min_files, self.max_files + 1))
        num_turns = int(self.rng.integers(self.min_turns, self.max_turns + 1))
        session_id = self.rng.bytes(6).hex()

        repo_context, filenames, class_names = self._generate_repo_context(num_files)

        # The system prompt + repo context is the shared prefix across all
        # turns in this session.
        system_content = f"{_CODING_SYSTEM_PROMPT}\n\n{repo_context}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]
        requests: list[Request] = []

        for turn in range(num_turns):
            # Pick a filename and class to reference
            filename = filenames[int(self.rng.integers(0, len(filenames)))]
            class_name = (
                class_names[int(self.rng.integers(0, len(class_names)))]
                if class_names
                else "Handler"
            )
            other_filename = filenames[int(self.rng.integers(0, len(filenames)))]

            if turn == 0:
                template = _CODING_QUERIES_INITIAL[
                    int(self.rng.integers(0, len(_CODING_QUERIES_INITIAL)))
                ]
                user_text = template.format(
                    filename=filename, class_name=class_name
                )
            else:
                template = _CODING_FOLLOWUPS[
                    int(self.rng.integers(0, len(_CODING_FOLLOWUPS)))
                ]
                user_text = template.format(
                    filename=filename,
                    class_name=class_name,
                    other_filename=other_filename,
                )

            messages.append({"role": "user", "content": user_text})

            # Coding responses tend to be longer (code blocks)
            expected_tokens = int(self.rng.integers(300, 1200))

            requests.append(
                Request(
                    request_id=_new_request_id(self.rng),
                    messages=list(messages),
                    expected_output_tokens=expected_tokens,
                    session_id=session_id,
                    metadata={
                        "workload": "coding",
                        "turn": turn,
                        "num_context_files": num_files,
                        "referenced_file": filename,
                        "primary_class": class_name,
                    },
                )
            )

            # Synthetic assistant response for context in subsequent turns
            response_parts = [
                f"Looking at `{filename}`, I can see the issue.",
                f"Here's the updated implementation for `{class_name}`:",
                "```",
                f"// Updated {filename} with the requested changes",
                f"// ... (modified {class_name} implementation)",
                "```",
                "The key changes are:",
                "1. Added proper error handling for edge cases.",
                "2. Improved the control flow for clarity.",
                "3. Added documentation for the public API.",
            ]
            messages.append({
                "role": "assistant",
                "content": "\n".join(response_parts),
            })

        return requests
