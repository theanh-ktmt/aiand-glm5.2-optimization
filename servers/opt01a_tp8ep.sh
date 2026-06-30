#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 1a — Parallelism: TP8 + Expert Parallel (TP8EP).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt01a_tp8ep"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --enable-expert-parallel
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
