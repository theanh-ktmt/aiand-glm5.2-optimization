#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 9 — Sampler: VLLM_USE_FLASHINFER_SAMPLER=1.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt09_flashinfer_sampler"
BENCH_MODE="mtp"
export VLLM_USE_FLASHINFER_SAMPLER=1
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
