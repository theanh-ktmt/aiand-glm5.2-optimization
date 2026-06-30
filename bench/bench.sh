#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bench.sh — InferenceX-style serving benchmark sweep for GLM-5.2-FP8
#
# Drives third_party/InferenceX/utils/bench_serving/benchmark_serving.py against
# an ALREADY-RUNNING vLLM server, sweeping concurrency for each ISL/OSL scenario
# with random data and a saturated server (--request-rate inf). One result JSON
# is written per (scenario, concurrency) cell.
#
# MTP vs non-MTP (the ONLY client-side difference):
#   * MTP runs add --use-chat-template (speculative decoding was trained on the
#     chat format; raw prompts silently tank acceptance length).
#   * non-MTP runs omit it.
# Select with: BENCH_MODE=mtp (default) | nonmtp, or --mode mtp|nonmtp.
#
# Usage:
#   bash bench/bench.sh --config baseline --mode mtp
#   SWEEP=subset bash bench/bench.sh --config opt04c_moe_deepgemm --mode mtp
#
# Env knobs:
#   CONFIG        label used in result filenames (required, or --config)
#   BENCH_MODE    mtp | nonmtp                       (default mtp)
#   SWEEP         full | subset                      (default full)
#                   full   -> conc 1 2 4 8 16 32 64 128, scenarios 1k1k + 8k1k
#                   subset -> conc 1 8 32,             scenarios 1k1k + 8k1k
#   CONCS         override concurrency list          (e.g. "1 8 32")
#   SCENARIOS     override scenarios "ISL:OSL ..."   (e.g. "1024:1024 8192:1024")
#   RANDOM_RANGE_RATIO                                (default 0.8)
#   RESULT_DIR    where JSONs land                   (default results/<CONFIG>)
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INFERENCEX_DIR="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
BENCH_PY="$INFERENCEX_DIR/utils/bench_serving/benchmark_serving.py"

MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
PORT="${PORT:-8888}"
BENCH_MODE="${BENCH_MODE:-mtp}"
SWEEP="${SWEEP:-full}"
RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-0.8}"
CONFIG="${CONFIG:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --mode)   BENCH_MODE="$2"; shift 2 ;;
        --sweep)  SWEEP="$2"; shift 2 ;;
        --port)   PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -z "$CONFIG" ]] && { echo "ERROR: --config (or CONFIG env) is required"; exit 1; }
[[ -f "$BENCH_PY" ]] || { echo "ERROR: $BENCH_PY missing. Run: git submodule update --init"; exit 1; }

# --- Scenario / concurrency matrix -----------------------------------------
SCENARIOS="${SCENARIOS:-1024:1024 8192:1024}"
if [[ -z "${CONCS:-}" ]]; then
    case "$SWEEP" in
        full)   CONCS="1 2 4 8 16 32 64 128" ;;
        subset) CONCS="1 8 32" ;;
        *) echo "ERROR: SWEEP must be full|subset"; exit 1 ;;
    esac
fi

# --- MTP toggle: the single client-side difference -------------------------
CHAT_TEMPLATE_ARG=()
if [[ "$BENCH_MODE" == "mtp" ]]; then
    CHAT_TEMPLATE_ARG=(--use-chat-template)
elif [[ "$BENCH_MODE" != "nonmtp" ]]; then
    echo "ERROR: BENCH_MODE must be mtp|nonmtp"; exit 1
fi

RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/$CONFIG}"
mkdir -p "$RESULT_DIR"

# Tee all benchmark console output to a saved log alongside the result JSONs.
BENCH_LOG="$RESULT_DIR/bench.log"
exec > >(tee -a "$BENCH_LOG") 2>&1
echo "# bench run @ $(date -u +%Y-%m-%dT%H:%M:%SZ)  config=$CONFIG mode=$BENCH_MODE sweep=$SWEEP"

echo "=============================================================="
echo "  Benchmark: config=$CONFIG mode=$BENCH_MODE sweep=$SWEEP"
echo "  scenarios=[$SCENARIOS] concs=[$CONCS] rrr=$RANDOM_RANGE_RATIO"
echo "  results -> $RESULT_DIR"
echo "=============================================================="

pip install -q datasets pandas 2>/dev/null || true

for scenario in $SCENARIOS; do
    ISL="${scenario%%:*}"
    OSL="${scenario##*:}"
    for CONC in $CONCS; do
        fname="${CONFIG}__${BENCH_MODE}_isl${ISL}_osl${OSL}_conc${CONC}"
        echo ">>> [$CONFIG/$BENCH_MODE] ISL=$ISL OSL=$OSL CONC=$CONC"
        python3 "$BENCH_PY" \
            --model "$MODEL" \
            --backend vllm \
            --base-url "http://0.0.0.0:$PORT" \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio "$RANDOM_RANGE_RATIO" \
            --num-prompts "$((CONC * 10))" \
            --max-concurrency "$CONC" \
            --request-rate inf \
            --ignore-eos \
            --num-warmups "$((2 * CONC))" \
            --trust-remote-code \
            --save-result \
            --percentile-metrics 'ttft,tpot,itl,e2el' \
            --metric-percentiles '90,99' \
            --result-dir "$RESULT_DIR" \
            --result-filename "${fname}.json" \
            "${CHAT_TEMPLATE_ARG[@]}" \
            || echo "WARN: cell failed (ISL=$ISL OSL=$OSL CONC=$CONC), continuing"
    done
done

echo "Done. Aggregate with:  python3 aggregate.py results/$CONFIG --config $CONFIG"
