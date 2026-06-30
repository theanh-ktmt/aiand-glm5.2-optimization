#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# final.sh — TEMPLATE for the recommended configuration (Confluence Section 6).
#
# Assemble the winning optimizations here after screening, then run the FULL
# sweep and the MMLU-Pro quality check:
#     bash run.sh final full                 # full throughput/latency sweep
#     RUN_EVAL=1 bash servers/final.sh        # MMLU-Pro accuracy
#     bash eval/quality_check.sh baseline final   # baseline-vs-final accuracy table
#
# As shipped, this equals the Day-0 baseline (TP8 + MTP5). Edit SERVE_ARGS below:
# uncomment / replace lines with the flags that won during screening. Each slot
# maps to one optimization group from SERVERS.md.
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "$0")/.." && pwd)/common.sh"
CONFIG="final"
BENCH_MODE="mtp"          # set to nonmtp only if MTP loses outright

SERVE_ARGS=(
    --kv-cache-dtype fp8_e4m3

    # --- 1. Parallelism: pick ONE -----------------------------------------
    --tensor-parallel-size 8
    # --enable-expert-parallel                       # opt01a: TP8 + EP
    # --data-parallel-size 8 --enable-expert-parallel  # opt01b: DP8 + EP (then 5/6/7 apply)

    # --- 2. Hyperparameters (opt02) ---------------------------------------
    # --max-num-batched-tokens 8192
    # --max-num-seqs 256
    # --gpu-memory-utilization 0.95

    # --- 4. MoE backend (opt04): triton | cutlass | deep_gemm | flashinfer_trtllm
    # --moe-backend deep_gemm        # if deep_gemm, also: export VLLM_USE_DEEP_GEMM=1 (above SERVE_ARGS)

    # --- 5. All2All backend (opt05, DP8EP only) ---------------------------
    # --all2all-backend deepep_low_latency

    # --- 6. EPLB (opt06, DP8EP only) --------------------------------------
    # --enable-eplb --eplb-config '{"window_size":1000,"step_interval":3000,"num_redundant_experts":16}'

    # --- 7. DBO (opt07, DP8EP only) ---------------------------------------
    # --enable-dbo --dbo-decode-token-threshold 32 --dbo-prefill-token-threshold 512

    # --- 10. CUDA graph (opt10) -------------------------------------------
    # --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32,48,64,96,128,192,256]}'

    # --- 8. MTP (opt08): set the winning num_speculative_tokens -----------
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
)

# --- 3/9. Env-var optimizations: uncomment the winners --------------------
# export VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE   # opt03a (or FLASHMLA_SPARSE for opt03b)
# export VLLM_USE_FLASHINFER_SAMPLER=1                  # opt09

serve_main
