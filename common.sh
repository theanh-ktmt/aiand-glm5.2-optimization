#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# common.sh — shared config + helpers for GLM-5.2-FP8 optimization benchmarks
#
# Every server script under servers/ sources this file. It centralizes:
#   * the model / hardware defaults (8x B300, vLLM 0.23.x)
#   * the *mandatory* benchmark-harness flags that must be identical across
#     every configuration (most importantly --no-enable-prefix-caching)
#   * server launch + readiness + cleanup helpers (reused from InferenceX)
#
# It does NOT define any optimization-specific flag — those live in the
# individual servers/*.sh scripts so each one is self-documenting.
# ---------------------------------------------------------------------------
set -uo pipefail

# --- Resolve repo root & third-party InferenceX -----------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT
INFERENCEX_DIR="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
export INFERENCEX_DIR

if [[ ! -f "$INFERENCEX_DIR/benchmarks/benchmark_lib.sh" ]]; then
    echo "ERROR: InferenceX not found at $INFERENCEX_DIR"
    echo "       Run: git submodule update --init --recursive"
    exit 1
fi
# Reuse InferenceX helpers: wait_for_server_ready, start/stop_gpu_monitor, etc.
# shellcheck source=/dev/null
source "$INFERENCEX_DIR/benchmarks/benchmark_lib.sh"

# --- Model / hardware defaults ---------------------------------------------
export MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
export MODEL_PATH="${MODEL_PATH:-}"        # local pre-staged dir, optional
export SERVE_MODEL="${MODEL_PATH:-$MODEL}"
export PORT="${PORT:-8888}"
export TP="${TP:-8}"                        # 8x B300 single replica
export NUM_GPUS="${NUM_GPUS:-8}"            # used by aggregator for per-GPU tput

# Benchmark-harness invariants (apply to EVERY config; not optimizations).
# max-model-len must cover the largest scenario (8k ISL + 1k OSL + buffer).
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

SERVER_LOG="${SERVER_LOG:-$REPO_ROOT/results/server.log}"
export SERVER_LOG
mkdir -p "$REPO_ROOT/results"

# --- Mandatory flags shared by all configurations --------------------------
# IMPORTANT: --no-enable-prefix-caching is REQUIRED. With prefix caching on,
# repeated random/warmup prefixes are served from cache, prefill is skipped,
# and throughput is inflated. This must never be removed for any config.
common_serve_args() {
    echo \
        --host 0.0.0.0 --port "$PORT" \
        --trust-remote-code \
        --no-enable-prefix-caching \
        --max-model-len "$MAX_MODEL_LEN" \
        --tool-call-parser glm47 \
        --enable-auto-tool-choice \
        --reasoning-parser glm45
}

# --- Server lifecycle -------------------------------------------------------
SERVER_PID=""

_descendants() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        echo "$child"; _descendants "$child"
    done
}

cleanup_server() {
    [[ -z "$SERVER_PID" ]] && return 0
    local descendants; descendants=$(_descendants "$SERVER_PID")
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    local pid; for pid in $descendants; do kill -9 "$pid" 2>/dev/null || true; done
    # Wait for GPU memory to drain before the next config launches.
    local waited=0
    while [[ $waited -lt 120 ]]; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -rn | head -1)
        [[ -z "$used" || "$used" -lt 2000 ]] && break
        sleep 3; waited=$((waited + 3))
    done
    SERVER_PID=""
}

# Download weights if a local MODEL_PATH was given but is empty.
ensure_model() {
    if [[ -n "$MODEL_PATH" ]]; then
        if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
            echo "=== MODEL_PATH ($MODEL_PATH) empty, downloading $MODEL ==="
            hf download "$MODEL" --local-dir "$MODEL_PATH"
        fi
    elif [[ "$SERVE_MODEL" != /* ]]; then
        hf download "$SERVE_MODEL" || true
    fi
}

# launch_vllm <config-name> <extra serve args...>
# Combines common flags + caller's optimization flags, starts the server,
# waits for /health, and leaves SERVER_PID set for the caller / trap.
launch_vllm() {
    local config_name="$1"; shift
    ensure_model
    nvidia-smi || true

    local -a args
    read -r -a args <<< "$(common_serve_args)"
    args+=("$@")

    echo "=============================================================="
    echo "  Launching config: $config_name"
    echo "  vllm serve $SERVE_MODEL ${args[*]}"
    echo "=============================================================="

    : > "$SERVER_LOG"
    vllm serve "$SERVE_MODEL" "${args[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"; then
        echo "ERROR: server failed to become healthy for config $config_name"
        cleanup_server
        return 1
    fi
    echo "Server ready (config=$config_name pid=$SERVER_PID)"
}

# serve_main — entrypoint every servers/*.sh calls after defining:
#     CONFIG       config label (e.g. opt04c_moe_deepgemm)
#     BENCH_MODE   mtp | nonmtp  (whether the bench client adds --use-chat-template)
#     SERVE_ARGS   bash array of the optimization-specific serve flags
#
# Default: launch the server, wait for health, stay in the foreground so you
# can benchmark it from another shell (matches the manual workflow).
# Set RUN_BENCH=1 (run.sh does this) to launch -> sweep -> tear down end-to-end.
serve_main() {
    trap cleanup_server EXIT INT TERM
    start_gpu_monitor --output "$REPO_ROOT/results/${CONFIG}_gpu.csv" 2>/dev/null || true
    launch_vllm "$CONFIG" "${SERVE_ARGS[@]}" || exit 1

    if [[ "${RUN_BENCH:-0}" == "1" ]]; then
        CONFIG="$CONFIG" BENCH_MODE="$BENCH_MODE" SWEEP="${SWEEP:-full}" PORT="$PORT" \
            bash "$REPO_ROOT/bench/bench.sh" --config "$CONFIG" --mode "$BENCH_MODE"
        stop_gpu_monitor 2>/dev/null || true
        cleanup_server
    else
        echo
        echo "Server is UP on port $PORT  (config=$CONFIG, bench mode=$BENCH_MODE)."
        echo "Benchmark it from another shell with:"
        echo "  CONFIG=$CONFIG bash bench/bench.sh --config $CONFIG --mode $BENCH_MODE --sweep subset"
        echo "Press Ctrl-C to stop the server."
        wait "$SERVER_PID"
    fi
}
