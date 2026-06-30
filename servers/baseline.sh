#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Baseline (Day 0) — exact recipes.vllm.ai recipe: TP8 + MTP(5).
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="baseline"
BENCH_MODE="mtp"   # Day-0 uses MTP, so the bench client must use --use-chat-template
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)
serve_main
