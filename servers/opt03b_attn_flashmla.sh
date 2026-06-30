#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3b — Attention backend: FlashMLA sparse (MLA path).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03b_attn_flashmla"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
