#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1b — Parallelism: DP8 attention + Expert Parallel (DP8EP). Base for opt05/06/07.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01b_dp8ep"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
