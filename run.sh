#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run.sh — end-to-end runner for ONE configuration:
#   launch server -> InferenceX sweep -> tear down -> aggregate to CSV
#   -> sync to W&B + git-backup the CSV (durability for on-demand cloud runs).
#
# Usage:
#   bash run.sh baseline                 # full sweep (1k1k + 8k1k, conc 1..128)
#   bash run.sh opt04_moe_deepgemm subset  # subset sweep (conc 1,16,128) for trials
#
# Positional args:
#   $1  config name = servers/<name>.sh   (required)
#   $2  sweep: full | subset              (default: subset)
#
# Durability knobs (both default ON):
#   WANDB=0       disable W&B sync       (else needs WANDB_API_KEY; see .env)
#   GIT_BACKUP=0  disable git CSV commit+push
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

# Load secrets / settings (WANDB_API_KEY, WANDB_PROJECT, ...) if present.
# .env is git-ignored — never commit credentials.
[[ -f "$REPO_ROOT/.env" ]] && set -a && source "$REPO_ROOT/.env" && set +a

NAME="${1:?usage: run.sh <config> [full|subset]}"
SWEEP="${2:-subset}"
NAME="${NAME%.sh}"; NAME="${NAME#servers/}"
SCRIPT="$REPO_ROOT/servers/$NAME.sh"
[[ -f "$SCRIPT" ]] || { echo "ERROR: no such config: $SCRIPT"; exit 1; }

echo "### RUN $NAME (sweep=$SWEEP) ###"
RUN_BENCH=1 SWEEP="$SWEEP" bash "$SCRIPT"

echo "### AGGREGATE $NAME ###"
CSV="$REPO_ROOT/results/$NAME.csv"
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results/$NAME" --config "$NAME" --out "$CSV"
echo "CSV: results/$NAME.csv"

# --- Durability: push results off this (ephemeral) box ---------------------
if [[ "${WANDB:-1}" != "0" ]]; then
    echo "### W&B SYNC $NAME ###"
    python3 -c "import wandb" 2>/dev/null || pip install -q wandb 2>/dev/null || true
    python3 "$REPO_ROOT/wandb_sync.py" --config "$NAME" || echo "WARN: W&B sync failed"
fi

if [[ "${GIT_BACKUP:-1}" != "0" ]]; then
    echo "### GIT BACKUP $NAME ###"
    git -C "$REPO_ROOT" add -f "results/$NAME.csv" 2>/dev/null \
        && git -C "$REPO_ROOT" commit -q -m "results: $NAME ($(date -u +%FT%TZ))" 2>/dev/null \
        && git -C "$REPO_ROOT" push -q 2>/dev/null \
        && echo "git backup pushed: results/$NAME.csv" \
        || echo "WARN: git backup skipped/failed (no creds, no change, or offline)"
fi
