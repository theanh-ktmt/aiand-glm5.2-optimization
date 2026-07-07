#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final2 — Proposed config #2 (DP8EP base, good for high concurrency). From MV-4594:
#   FlashMLA sparse attention + hyperparams + MTP(2),
#   data-parallel 8 + expert parallel + pplx all2all.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final2"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend pplx
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
)
serve_main
