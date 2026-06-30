#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4 — MoE backend: --moe-backend cutlass.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04_moe_cutlass"
BENCH_MODE="mtp"

SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --moe-backend cutlass
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
