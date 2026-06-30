#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh — end-to-end runner for ONE configuration:
#   launch server -> InferenceX sweep -> tear down -> aggregate to CSV.
#
# Usage:
#   bash run.sh baseline                 # full sweep (1k1k + 8k1k, conc 1..128)
#   bash run.sh opt04_moe_deepgemm subset  # subset sweep (conc 1,8,32) for trials
#
# Positional args:
#   $1  config name = servers/<name>.sh   (required)
#   $2  sweep: full | subset              (default: subset)
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

NAME="${1:?usage: run.sh <config> [full|subset]}"
SWEEP="${2:-subset}"
NAME="${NAME%.sh}"; NAME="${NAME#servers/}"
SCRIPT="$REPO_ROOT/servers/$NAME.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: no such config: $SCRIPT"; exit 1; }

echo "### RUN $NAME (sweep=$SWEEP) ###"
RUN_BENCH=1 SWEEP="$SWEEP" bash "$SCRIPT"

echo "### AGGREGATE $NAME ###"
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results/$NAME" \
    --config "$NAME" --out "$REPO_ROOT/results/${NAME}.csv"
echo "CSV: results/${NAME}.csv"
