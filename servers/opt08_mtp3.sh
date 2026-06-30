#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — MTP: num_speculative_tokens=3.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt08_mtp3"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
)
serve_main
