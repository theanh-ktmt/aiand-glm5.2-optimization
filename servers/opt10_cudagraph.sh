#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 10 — CUDA graph: FULL_AND_PIECEWISE + capture sizes tuned to --max-num-seqs.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt10_cudagraph"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --max-num-seqs 256
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128,192,256]}'
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
