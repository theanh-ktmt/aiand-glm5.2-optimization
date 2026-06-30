#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh — run the whole optimization campaign in the prescribed order:
#   1. baseline      -> FULL sweep   (reference curve)
#   2. each opt      -> SUBSET sweep (conc 1,8,32, both scenarios) for screening
#   3. (later) build final config and re-run FULL manually.
#
# Each config is launched, benchmarked, and torn down before the next one.
# Results land in results/<config>/ and results/<config>.csv; a combined
# results/all.csv is written at the end for pasting into the sheet.
#
# Usage:
#   bash run_all.sh                 # baseline full + all opts subset
#   bash run_all.sh --only opt04_moe_deepgemm opt09_flashinfer_sampler
#   BASELINE_SWEEP=subset bash run_all.sh   # quick smoke of the whole pipeline
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && set -a && source "$REPO_ROOT/.env" && set +a
BASELINE_SWEEP="${BASELINE_SWEEP:-full}"
OPT_SWEEP="${OPT_SWEEP:-subset}"

# Screening order. DP8EP-dependent opts (a2a/eplb/dbo) are grouped after the
# parallelism choice so you can decide TP8EP vs DP8EP first.
ALL_CONFIGS=(
    baseline
    opt01a_tp8ep opt01b_dp8ep
    opt02_hyperparams
    opt03a_attn_flashinfer_trtllm opt03b_attn_flashmla
    opt04_moe_triton opt04_moe_cutlass opt04_moe_deepgemm opt04_moe_flashinfer_trtllm
    opt05_a2a_naive opt05_a2a_pplx opt05_a2a_deepep_low_latency
    opt05_a2a_deepep_high_throughput opt05_a2a_flashinfer_nvlink
    opt06_eplb opt07_dbo
    opt08_mtp1 opt08_mtp2 opt08_mtp3 opt08_mtp4 opt08_mtp_disable_bs64
    opt09_flashinfer_sampler
    opt10_cudagraph
    ref_nonmtp
)

CONFIGS=("${ALL_CONFIGS[@]}")
if [[ "${1:-}" == "--only" ]]; then
    shift; CONFIGS=("$@")
fi

for cfg in "${CONFIGS[@]}"; do
    sweep="$OPT_SWEEP"
    [[ "$cfg" == "baseline" ]] && sweep="$BASELINE_SWEEP"
    echo "==================== $cfg (sweep=$sweep) ===================="
    if ! bash "$REPO_ROOT/run.sh" "$cfg" "$sweep"; then
        echo "WARN: $cfg failed; continuing to next config." >&2
    fi
done

echo "==================== combined CSV ===================="
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results" \
    --baseline baseline --out "$REPO_ROOT/results/all.csv"
echo "Combined: results/all.csv"

# Back up the combined CSV too (per-config sync/backup already ran in run.sh).
if [[ "${GIT_BACKUP:-1}" != "0" ]]; then
    git -C "$REPO_ROOT" add -f "results/all.csv" 2>/dev/null \
        && git -C "$REPO_ROOT" commit -q -m "results: combined all.csv ($(date -u +%FT%TZ))" 2>/dev/null \
        && git -C "$REPO_ROOT" push -q 2>/dev/null \
        && echo "git backup pushed: results/all.csv" \
        || echo "WARN: git backup of all.csv skipped/failed"
fi
