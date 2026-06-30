#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 5 — (DP8EP) All2All backend: --all2all-backend pplx.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt05_a2a_pplx"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend pplx
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
