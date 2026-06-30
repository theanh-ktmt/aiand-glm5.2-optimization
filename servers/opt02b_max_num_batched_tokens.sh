#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2b — Hyperparameter: --max-num-batched-tokens (tunable via MNBT, default 16384).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02b_max_num_batched_tokens"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --max-num-batched-tokens "${MNBT:-16384}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
