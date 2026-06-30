#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2a — Hyperparameter: --max-num-seqs (tunable via MAX_NUM_SEQS, default 256).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02a_max_num_seqs"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --max-num-seqs "${MAX_NUM_SEQS:-256}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
