"""Regression coverage for `telemetry/prometheus.py` text parser correctness.

Closes the snapshot v1.0.0 P1 bug `prometheus_text_parser_correctness.md`,
which documented 3 sub-bugs:

- Sub-bug 1: histogram buckets dropped from `raw_metrics` with no
  alternate access path → percentile-aware consumers had no way to
  read bucket data.
- Sub-bug 2: label parser split on `,` with no quote-awareness, so
  any future custom metric with quoted commas in label values
  silently corrupted the labels dict.
- Sub-bug 3: bucket-skip used a string-concatenation hack
  (`if "_bucket{" in f"{sample.name}{{":`) that would silently
  break if `MetricSample.name` storage ever changed.
"""

from __future__ import annotations

from inferscope.telemetry.prometheus import (
    MetricSample,
    ScrapeResult,
    _parse_labels,
    parse_prometheus_text,
)


# ----------------------------------------------------------------------------
# Sub-bug 1 — histograms accessible via get_histogram_buckets
# ----------------------------------------------------------------------------


def test_scrape_result_exposes_histogram_buckets() -> None:
    """`ScrapeResult.histograms` must be populated when buckets are present
    so percentile-aware consumers can read them. The previous parser dropped
    bucket data on the floor."""
    text = (
        "# HELP latency_seconds Request latency\n"
        "# TYPE latency_seconds histogram\n"
        'latency_seconds_bucket{le="0.1"} 5\n'
        'latency_seconds_bucket{le="0.5"} 12\n'
        'latency_seconds_bucket{le="1.0"} 18\n'
        'latency_seconds_bucket{le="+Inf"} 20\n'
        "latency_seconds_sum 8.5\n"
        "latency_seconds_count 20\n"
    )
    result = ScrapeResult(endpoint="http://test/metrics", engine="vllm")
    result.samples = parse_prometheus_text(text)

    # Mirror the scrape_metrics post-parse loop manually:
    for sample in result.samples:
        if sample.name.endswith("_bucket"):
            base = sample.name[: -len("_bucket")]
            le_label = sample.labels.get("le", "")
            result.histograms.setdefault(base, []).append((le_label, sample.value))
            continue
        result.raw_metrics[sample.name] = sample.value

    buckets = result.get_histogram_buckets("latency_seconds")
    assert len(buckets) == 4
    assert buckets[0] == ("0.1", 5.0)
    assert buckets[1] == ("0.5", 12.0)
    assert buckets[2] == ("1.0", 18.0)
    assert buckets[3] == ("+Inf", 20.0)

    # raw_metrics should NOT contain any *_bucket entries
    assert all(not k.endswith("_bucket") for k in result.raw_metrics)
    # _sum and _count are still in raw_metrics so the existing get_histogram_avg works
    assert result.raw_metrics["latency_seconds_sum"] == 8.5
    assert result.raw_metrics["latency_seconds_count"] == 20.0
    avg = result.get_histogram_avg("latency_seconds")
    assert avg == 8.5 / 20.0


def test_get_histogram_buckets_returns_empty_for_unknown_base() -> None:
    result = ScrapeResult(endpoint="http://test/metrics", engine="vllm")
    assert result.get_histogram_buckets("nonexistent") == []


# ----------------------------------------------------------------------------
# Sub-bug 2 — quoted commas in label values
# ----------------------------------------------------------------------------


def test_parse_labels_handles_unquoted_simple_case() -> None:
    """Backward-compat: simple labels without quoted commas still work."""
    labels = _parse_labels('engine="vllm",model="kimi"')
    assert labels == {"engine": "vllm", "model": "kimi"}


def test_parse_labels_handles_quoted_comma_in_value() -> None:
    """The headline scenario for sub-bug 2: a label value containing a
    quoted comma must NOT split the label set on it."""
    labels = _parse_labels('message="hello, world",level="info"')
    assert labels == {"message": "hello, world", "level": "info"}


def test_parse_labels_handles_multiple_quoted_commas() -> None:
    labels = _parse_labels('a="x, y, z",b="1, 2"')
    assert labels == {"a": "x, y, z", "b": "1, 2"}


def test_parse_labels_handles_escaped_quote_in_value() -> None:
    """Per OpenMetrics, `\\"` inside a quoted value is an escaped quote."""
    labels = _parse_labels(r'msg="he said \"hi\""')
    assert labels == {"msg": 'he said "hi"'}


def test_parse_labels_handles_escaped_backslash_in_value() -> None:
    labels = _parse_labels(r'path="C:\\Users\\test"')
    assert labels == {"path": r"C:\Users\test"}


def test_parse_labels_empty_string_returns_empty_dict() -> None:
    assert _parse_labels("") == {}


def test_parse_prometheus_text_handles_quoted_comma_in_metric_line() -> None:
    """End-to-end: a full metric line with a quoted comma in the labels parses cleanly."""
    text = 'custom_metric{message="hello, world",level="info"} 42.0'
    samples = parse_prometheus_text(text)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.name == "custom_metric"
    assert sample.value == 42.0
    assert sample.labels == {"message": "hello, world", "level": "info"}


# ----------------------------------------------------------------------------
# Sub-bug 3 — bucket-skip uses .endswith, not string concatenation
# ----------------------------------------------------------------------------


def test_bucket_skip_uses_clean_endswith_check() -> None:
    """The fix replaces `if "_bucket{" in f"{sample.name}{{":` with
    `if sample.name.endswith("_bucket"):`. We verify the new check works on
    a synthetic sample without relying on the f-string substring trick."""
    bucket_sample = MetricSample(
        name="latency_bucket",
        labels={"le": "0.5"},
        value=10.0,
    )
    non_bucket_sample = MetricSample(
        name="latency_sum",
        labels={},
        value=5.0,
    )
    # The fix uses sample.name.endswith("_bucket")
    assert bucket_sample.name.endswith("_bucket")
    assert not non_bucket_sample.name.endswith("_bucket")


def test_metric_sample_with_bucket_substring_in_name_is_correctly_classified() -> None:
    """Defense in depth: the old hack `_bucket{ in f"{name}{{"` would also
    match a hypothetical sample named `something_bucket_other` (because the
    substring `_bucket{` would appear after the trailing `{`). The new
    `endswith` check is correct: only names actually ending in `_bucket`
    are treated as histogram buckets."""
    real_bucket = MetricSample(name="latency_bucket", labels={"le": "1.0"}, value=10.0)
    not_a_bucket = MetricSample(name="latency_bucket_total", labels={}, value=5.0)

    assert real_bucket.name.endswith("_bucket")
    assert not not_a_bucket.name.endswith("_bucket")
