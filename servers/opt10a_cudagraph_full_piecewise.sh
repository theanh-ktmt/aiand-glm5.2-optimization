#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 10a — CUDA graph: FULL_AND_PIECEWISE mode.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt10a_cudagraph_full_piecewise"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
