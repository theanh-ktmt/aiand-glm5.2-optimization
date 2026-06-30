#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2 — Hyperparameters: max-num-batched-tokens + max-num-seqs + gpu-mem-util.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02_hyperparams"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
