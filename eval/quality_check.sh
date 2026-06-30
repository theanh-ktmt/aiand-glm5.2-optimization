#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# quality_check.sh — MMLU-Pro accuracy: baseline vs recommended config.
#
# Confluence Section 6 quality gate: confirm the optimized config did not
# regress quality versus Day-0. (GLM-5.2-FP8 is text-only, so MMMU-Pro is N/A.)
#
# Runs the eval on each config (launch -> lm-eval mmlu_pro -> teardown), then
# prints a comparison table and writes results/quality_check.csv.
#
# Usage:
#   bash eval/quality_check.sh                 # baseline vs final
#   bash eval/quality_check.sh baseline opt04_moe_deepgemm
#   EVAL_CONC=128 bash eval/quality_check.sh   # override eval concurrency
#
# Skip re-running an eval you already have: SKIP_RUN=1 bash eval/quality_check.sh
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

BASE="${1:-baseline}"
CAND="${2:-final}"

PY="$(command -v python3 || command -v python)"

for cfg in "$BASE" "$CAND"; do
    script="$REPO_ROOT/servers/$cfg.sh"
    [[ -f "$script" ]] || { echo "ERROR: no such config: $script"; exit 1; }
    if [[ "${SKIP_RUN:-0}" == "1" && -d "$REPO_ROOT/results/$cfg/mmlu_pro" ]]; then
        echo "### SKIP eval (existing results): $cfg ###"
        continue
    fi
    echo "### MMLU-Pro eval: $cfg ###"
    RUN_EVAL=1 bash "$script" || echo "WARN: eval for $cfg returned non-zero"
done

echo "### COMPARE ###"
"$PY" "$REPO_ROOT/eval/parse_mmlu.py" \
    "$REPO_ROOT/results/$BASE/mmlu_pro" \
    "$REPO_ROOT/results/$CAND/mmlu_pro" \
    --names "$BASE" "$CAND" \
    --out "$REPO_ROOT/results/quality_check.csv"
