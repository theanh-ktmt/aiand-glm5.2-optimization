#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 2c — Hyperparameter: --gpu-memory-utilization (tunable via GMU, default 0.92).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt02c_gpu_memory_utilization"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --gpu-memory-utilization "${GMU:-0.92}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
