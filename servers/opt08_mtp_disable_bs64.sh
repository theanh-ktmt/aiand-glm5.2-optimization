#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 8 — MTP(5) disabled above batch size 64 (num_speculative_tokens_per_batch_size schedule).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt08_mtp_disable_bs64"
BENCH_MODE="mtp"
# Use 5 speculative tokens for batch 1..64, then 0 (MTP off) above 64.
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":5,"num_speculative_tokens_per_batch_size":[[1,64,5],[65,100000,0]]}'
)
serve_main
