#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reference — MTP disabled (non-MTP). Lets you quantify MTP's contribution vs baseline.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="ref_nonmtp"
BENCH_MODE="nonmtp"   # no --use-chat-template on the client
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --tensor-parallel-size 8
)
serve_main
