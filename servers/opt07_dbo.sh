#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Opt 7 — (DP8EP) DBO: dual-batch overlap (compute/communication overlap).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="opt07_dbo"
BENCH_MODE="mtp"
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --enable-dbo
    --dbo-decode-token-threshold "${DBO_DECODE_THR:-32}"
    --dbo-prefill-token-threshold "${DBO_PREFILL_THR:-512}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
