#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final2 — Proposed config #2 (DP8EP base, good for high concurrency). From MV-4594:
#   FlashMLA sparse attention + hyperparams + MTP(2),
#   data-parallel 8 + expert parallel + naive all2all.
#
# NOTE: MV-4594 originally proposed 'pplx', but pplx was REMOVED in vLLM 0.24.0
# (silently falls back to allgather_reducescatter, which crashed the DP8EP run).
# 'naive' matches pplx's throughput in the 0.23 screening (conc128: 3719 vs 3699
# tok/s) and needs no extra features (deepep_ll/ht are slower and require
# additional flags), so it's the pplx replacement here.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final2"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend naive
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
)
serve_main
