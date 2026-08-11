#!/usr/bin/env bash
#
# Load-test a running Sirius instance and print a Markdown latency table.
#
# Prerequisites:
#   * hey            (brew install hey)
#   * the service running on the target URL
#
# Usage:
#   ./scripts/loadtest.sh [URL] [BODY_FILE]
#   REQUESTS=20000 CONCURRENCY=50 ./scripts/loadtest.sh
#
set -euo pipefail

URL="${1:-http://127.0.0.1:8888/fraud-score}"
BODY="${2:-tests/mock-request-body.json}"
REQUESTS="${REQUESTS:-20000}"
CONCURRENCY="${CONCURRENCY:-50}"

if ! command -v hey >/dev/null 2>&1; then
  echo "hey not found. Install it with: brew install hey" >&2
  exit 1
fi

# Warm-up: prime the server's threadpool, caches and first-call costs so the
# measured run reflects steady state rather than a one-off cold-start batch
# (the classic "exactly <concurrency> slow requests" spike). Output discarded.
hey -n 300 -c "$CONCURRENCY" -m POST \
  -H "Content-Type: application/json" -D "$BODY" "$URL" >/dev/null 2>&1 || true

# Measured run.
raw="$(hey -n "$REQUESTS" -c "$CONCURRENCY" -m POST \
  -H "Content-Type: application/json" -D "$BODY" "$URL")"

# Full hey report to stderr for reference; the table goes to stdout.
echo "$raw" >&2

# Robustly pull a percentile from hey's "Latency distribution" block. Matching on
# the number + "in ... secs" tolerates single or doubled percent signs across
# hey builds, and "^p%" stops 5 from matching 95.
pct() {
  echo "$raw" | awk -v p="$1" '$2=="in" && $4=="secs" && $1 ~ "^"p"%" {printf "%.1f", $3 * 1000}'
}
rps="$(echo "$raw" | awk '/Requests\/sec:/ {printf "%.0f", $2}')"

cat <<EOF

| Metric | Value |
|---|---|
| Load | ${REQUESTS} requests at concurrency ${CONCURRENCY} (after warm-up) |
| p50 | $(pct 50) ms |
| p90 | $(pct 90) ms |
| p95 | $(pct 95) ms |
| p99 | $(pct 99) ms |
| Throughput | ${rps} req/s |
EOF