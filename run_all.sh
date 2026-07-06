#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all.sh — run the whole optimization campaign in the prescribed order:
#   1. baseline      -> FULL sweep   (reference curve)
#   2. each opt      -> SUBSET sweep (1k/1k only, conc 1/16/128) for screening
#   3. (later) build final config and re-run FULL manually.
#
# Each config is launched, benchmarked, and torn down before the next one.
# Results land in results/<config>/ and results/<config>.csv; a combined
# results/all.csv is written at the end for pasting into the sheet.
#
# Usage:
#   bash run_all.sh                 # baseline full + all opts subset
#   bash run_all.sh --only opt04_moe_deepgemm opt09_flashinfer_sampler
#   SKIP_EXISTING=1 bash run_all.sh # resume: skip configs already done
#                                   #   (those with results/<cfg>.csv, e.g. a
#                                   #    baseline you ran earlier via run.sh)
#   BASELINE_SWEEP=subset bash run_all.sh   # quick smoke of the whole pipeline
#   CONFIG_TIMEOUT=2400 bash run_all.sh     # tighten the per-config cap for screening
#
# Resilience: every config already fails fast on a stuck *startup*
# (SERVER_STARTUP_TIMEOUT in common.sh) and the campaign continues on any failure.
# Additionally:
#   * CONFIG_TIMEOUT (default 10800s = 3h; set 0 to disable) is a HARD wall-clock
#     cap on each config — covers post-startup hangs (stuck benchmark cell /
#     teardown). On expiry the run is killed (SIGTERM, then SIGKILL after 60s).
#   * After every config (success, failure, or timeout) reap_gpu force-kills any
#     stray vLLM process and VERIFIES the GPU is free before the next config,
#     so an orphaned/killed server never OOMs the next run.
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && { set -a; source <(tr -d '\r' < "$REPO_ROOT/.env"); set +a; }
BASELINE_SWEEP="${BASELINE_SWEEP:-full}"
OPT_SWEEP="${OPT_SWEEP:-subset}"
CONFIG_TIMEOUT="${CONFIG_TIMEOUT:-10800}"   # seconds; 3h. Set 0 to disable.

# Make sure NO vLLM process is left holding the GPU before the next config.
# The launcher's cmdline is 'vllm serve ...', but the memory is held by the
# engine/worker processes whose titles are 'VLLM::EngineCore' / 'VllmWorker'
# (set via setproctitle) — so we match both, escalate to SIGKILL, and poll
# nvidia-smi until memory actually drains.
reap_gpu() {
    echo ">>> reap_gpu: ensuring no vLLM process holds the GPU"
    # graceful first
    pkill -TERM -f 'vllm serve' 2>/dev/null || true
    pkill -TERM VLLM            2>/dev/null || true
    sleep 5
    local waited=0 used
    while [[ $waited -lt 180 ]]; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        if [[ -z "$used" || "$used" -lt 2000 ]]; then
            echo ">>> reap_gpu: GPU clear (max used=${used:-NA} MiB)"
            return 0
        fi
        echo ">>> reap_gpu: GPU still busy (${used} MiB) — pkill -9 vLLM"
        pkill -9 -f 'vllm serve' 2>/dev/null || true
        pkill -9 VLLM            2>/dev/null || true
        pkill -9 -f 'vllm'       2>/dev/null || true
        sleep 5; waited=$((waited + 5))
    done
    echo "WARN: GPU still >2000 MiB used after reap — next config may OOM." >&2
}

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
    # Resume: skip a config that already completed (run.sh writes results/<cfg>.csv
    # only after a successful sweep+aggregate, so it's a reliable completion marker).
    if [[ "${SKIP_EXISTING:-0}" == "1" && -f "$REPO_ROOT/results/$cfg.csv" ]]; then
        echo "==================== SKIP $cfg (results/$cfg.csv exists) ===================="
        continue
    fi
    echo "==================== $cfg (sweep=$sweep) ===================="
    rc=0
    if [[ "$CONFIG_TIMEOUT" -gt 0 ]]; then
        timeout --signal=TERM --kill-after=60 "$CONFIG_TIMEOUT" \
            bash "$REPO_ROOT/run.sh" "$cfg" "$sweep" || rc=$?
        if [[ "$rc" == "124" ]]; then
            echo "WARN: $cfg exceeded CONFIG_TIMEOUT=${CONFIG_TIMEOUT}s — killed; skipping." >&2
        elif [[ "$rc" != "0" ]]; then
            echo "WARN: $cfg failed (rc=$rc); continuing to next config." >&2
        fi
    else
        bash "$REPO_ROOT/run.sh" "$cfg" "$sweep" || { rc=$?; echo "WARN: $cfg failed (rc=$rc); continuing." >&2; }
    fi
    # Always leave a clean GPU for the next config.
    reap_gpu
done

echo "==================== combined CSV ===================="
python3 "$REPO_ROOT/aggregate.py" "$REPO_ROOT/results" \
    --baseline baseline --out "$REPO_ROOT/results/all.csv"
echo "Combined: results/all.csv"
# (Per-config results are already in W&B; each run.sh call synced its config.)
