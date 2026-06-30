#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 3a — Attention backend: FlashInfer MLA sparse + TRT-LLM decode kernels (GLM-5.2 DSA).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt03a_attn_flashinfer_trtllm"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
