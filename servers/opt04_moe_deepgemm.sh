#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 4 — MoE backend: --moe-backend deep_gemm.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt04_moe_deepgemm"
BENCH_MODE="mtp"
export VLLM_USE_DEEP_GEMM=1
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --moe-backend deep_gemm
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
