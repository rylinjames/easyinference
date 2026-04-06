# Suggested cross-link to add to `docs/observability/metrics.md`

Add a single line under the existing introductory prose of
`docs/observability/metrics.md`, near the top where the doc first
tells the reader what format the `/metrics` endpoint uses. Suggested
insertion:

```markdown
All metrics are exposed in the standard Prometheus text exposition
format. For a complete, realistic example of what the frontend and
KVBM endpoints look like, see
[Sample /metrics Response](./sample-metrics-response.md).
```

Feel free to reword to fit the existing doc's voice. The important
thing is that a reader landing on `metrics.md` can find the example
without having to search for it.
