#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 6 — (DP8EP) EPLB expert load balancing. Tune redundant experts via EPLB_REDUNDANT.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt06_eplb"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-eplb
    --eplb-config "{\"window_size\":1000,\"step_interval\":3000,\"num_redundant_experts\":${EPLB_REDUNDANT:-16}}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
