#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final1 — Proposed config #1 (TP8 base). From MV-4594:
#   FlashMLA sparse attention + hyperparams + MTP(2), tensor-parallel 8.
#   Best on the typical/low-to-mid concurrency range.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final1"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
)
serve_main
