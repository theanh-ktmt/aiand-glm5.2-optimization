#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# quality_check.sh — MMLU-Pro accuracy: baseline vs one or more configs.
#
# Confluence Section 6 quality gate: confirm the optimized config(s) did not
# regress quality versus Day-0. (GLM-5.2-FP8 is text-only, so MMMU-Pro is N/A.)
#
# Runs the eval on each config (launch -> lm-eval mmlu_pro -> teardown), then
# prints one comparison table (delta + Pass? vs the FIRST config) and writes
# results/quality_check.csv.
#
# Usage:
#   bash eval/quality_check.sh                          # baseline final1 final2
#   bash eval/quality_check.sh baseline final1          # just one candidate
#   bash eval/quality_check.sh baseline final1 final2 opt04_moe_deepgemm
#   EVAL_CONC=128 bash eval/quality_check.sh            # override eval concurrency
#   SKIP_RUN=1 bash eval/quality_check.sh               # reuse existing eval results
#
# The FIRST config is the baseline reference for the delta/Pass? verdict.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Default set: baseline + both proposed configs.
if [[ $# -gt 0 ]]; then CONFIGS=("$@"); else CONFIGS=(baseline final1 final2); fi

PY="$(command -v python3 || command -v python)"

for cfg in "${CONFIGS[@]}"; do
    script="$REPO_ROOT/servers/$cfg.sh"
    [[ -f "$script" ]] || { echo "ERROR: no such config: $script"; exit 1; }
    if [[ "${SKIP_RUN:-0}" == "1" && -d "$REPO_ROOT/results/$cfg/mmlu_pro" ]]; then
        echo "### SKIP eval (existing results): $cfg ###"
        continue
    fi
    echo "### MMLU-Pro eval: $cfg ###"
    RUN_EVAL=1 bash "$script" || echo "WARN: eval for $cfg returned non-zero"
done

echo "### COMPARE (baseline = ${CONFIGS[0]}) ###"
dirs=(); for cfg in "${CONFIGS[@]}"; do dirs+=("$REPO_ROOT/results/$cfg/mmlu_pro"); done
"$PY" "$REPO_ROOT/eval/parse_mmlu.py" \
    "${dirs[@]}" \
    --names "${CONFIGS[@]}" \
    --out "$REPO_ROOT/results/quality_check.csv"
