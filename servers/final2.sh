#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final2 — Proposed config #2 (DP8EP base, good for high concurrency). From MV-4594:
#   FlashMLA sparse attention + hyperparams + MTP(2),
#   data-parallel 8 + expert parallel + allgather_reducescatter all2all.
#
# NOTE: MV-4594 proposed 'pplx', but on vLLM 0.24.0 BOTH 'pplx' and 'naive' were
# removed (they silently fall back to 'allgather_reducescatter'). We set that
# fallback explicitly here. It's the only zero-extra-setup DP8EP all2all left on
# 0.24.0 (deepep_low_latency/high_throughput need DeepEP enabled and are slower).
# A prior run on this fallback crashed a DP worker under load — VALIDATE on a
# subset sweep first (`bash run.sh final2 subset`) before the full run.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final2"
BENCH_MODE="mtp"
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3
    --data-parallel-size 8
    --enable-expert-parallel
    --all2all-backend allgather_reducescatter
    --max-num-batched-tokens 8192
    --max-num-seqs 256
    --gpu-memory-utilization 0.95
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
)
serve_main
