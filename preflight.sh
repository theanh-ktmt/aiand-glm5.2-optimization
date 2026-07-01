#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# preflight.sh — run this ON THE CLOUD BOX before a long campaign.
#
# Validates the environment WITHOUT launching the model:
#   * required tools (python3, curl, nvidia-smi, git, vllm)
#   * vLLM version + GPU count
#   * InferenceX submodule present and benchmark_serving.py importable
#   * EVERY `--flag` used by common.sh + servers/*.sh exists in `vllm serve --help`
#     (catches flags that don't exist in this exact vLLM build — the #1 risk)
#   * attention-backend names referenced by opt03* exist in this build
#   * bash syntax of every script; wandb availability (warn only)
#
# Exit 0 = good to go. Exit 1 = at least one hard failure (see FAIL lines).
# ---------------------------------------------------------------------------
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$REPO_ROOT/.env" ]] && set -a && source "$REPO_ROOT/.env" && set +a

FAIL=0
ok()   { echo "  OK   $*"; }
warn() { echo "  WARN $*"; }
bad()  { echo "  FAIL $*"; FAIL=1; }

echo "== Tools =="
for t in python3 curl nvidia-smi git vllm; do
    if command -v "$t" >/dev/null 2>&1; then ok "$t -> $(command -v "$t")"; else
        if [[ "$t" == "git" ]]; then warn "$t not found (only needed for submodule update)"; else bad "$t not found"; fi
    fi
done

echo "== vLLM =="
if command -v vllm >/dev/null 2>&1; then
    ver="$(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?')"
    echo "  vLLM version: $ver  (recipes/this harness target: 0.23.x)"
    case "$ver" in 0.23.*) ok "version in 0.23.x" ;; *) warn "version is not 0.23.x — flag names may differ" ;; esac
fi

echo "== Model =="
MODEL="${MODEL:-zai-org/GLM-5.2-FP8}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/workspace/models}"
echo "  MODEL=$MODEL  (client --model / --served-model-name)"
if [[ -n "${MODEL_PATH:-}" ]]; then
    echo "  MODEL_PATH=$MODEL_PATH  (the weights the server will actually load)"
    if [[ -d "$MODEL_PATH" && -n "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        ok "MODEL_PATH exists and is non-empty"
    else
        bad "MODEL_PATH is set but missing/empty"
    fi
    case "$MODEL_PATH" in
        *[Gg][Ll][Mm]*) : ;;
        *) warn "MODEL_PATH doesn't look like a GLM model — the harness serves THIS path, so double-check it's GLM-5.2-FP8 (unset MODEL_PATH to pull $MODEL into $DOWNLOAD_DIR)" ;;
    esac
else
    echo "  MODEL_PATH unset -> server pulls '$MODEL' into --download-dir $DOWNLOAD_DIR"
fi

echo "== GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
    echo "  visible GPUs: $n"
    [[ "$n" -eq "${TP:-8}" ]] && ok "GPU count matches TP=${TP:-8}" || warn "GPU count ($n) != TP=${TP:-8}"
fi

echo "== InferenceX submodule =="
IX="${INFERENCEX_DIR:-$REPO_ROOT/third_party/InferenceX}"
if [[ -f "$IX/benchmarks/benchmark_lib.sh" ]]; then ok "benchmark_lib.sh present"; else bad "missing $IX (run: git submodule update --init --recursive)"; fi
BENCH_PY="$IX/utils/bench_serving/benchmark_serving.py"
if [[ -f "$BENCH_PY" ]]; then
    if python3 "$BENCH_PY" --help >/dev/null 2>&1; then ok "benchmark_serving.py --help works"; else warn "benchmark_serving.py --help failed (deps installed at run time)"; fi
else bad "missing $BENCH_PY"; fi

echo "== bash syntax =="
for f in "$REPO_ROOT"/common.sh "$REPO_ROOT"/run.sh "$REPO_ROOT"/run_all.sh \
         "$REPO_ROOT"/bench/bench.sh "$REPO_ROOT"/eval/quality_check.sh "$REPO_ROOT"/servers/*.sh; do
    bash -n "$f" 2>/dev/null && ok "$(basename "$f")" || bad "syntax: $f"
done

echo "== Serve flags vs 'vllm serve --help=all' =="
# NOTE: plain 'vllm serve --help' only prints config-group pointers in 0.23.x;
# the actual flags appear under --help=all. Use that so the check is accurate.
HELP=""
if command -v vllm >/dev/null 2>&1; then
    HELP="$(vllm serve --help=all 2>/dev/null)"
    [[ -z "$HELP" ]] && HELP="$(vllm serve --help 2>/dev/null)"   # fallback
    # Collect serve flags ONLY from lines that begin with '--' (the SERVE_ARGS
    # array elements and the common_serve_args echo block). This excludes comments
    # and flags belonging to other commands (nvidia-smi/curl/hf/git) on other lines.
    mapfile -t FLAGS < <(grep -rhE '^[[:space:]]*--' \
        "$REPO_ROOT/common.sh" "$REPO_ROOT"/servers/*.sh \
        | grep -oE -- '--[a-zA-Z0-9][a-zA-Z0-9-]*' | sort -u)
    for fl in "${FLAGS[@]}"; do
        # --no-enable-* are the negated halves of BooleanOptionalAction flags.
        probe="$fl"; [[ "$fl" == --no-* ]] && probe="--${fl#--no-}"
        if grep -qF -- "$fl" <<< "$HELP" || grep -qF -- "$probe" <<< "$HELP"; then
            ok "$fl"
        else
            bad "$fl not in 'vllm serve --help=all' (this build may not support it)"
        fi
    done
else
    warn "vllm not found; skipped flag check"
fi

echo "== Enum flag values (moe/all2all/kv-cache-dtype) =="
# Verify the specific choice we pass is accepted by THIS build, by matching it
# inside the help's '--flag {a,b,c}' choice list.
check_choice() {  # $1=flag  $2=value
    local line set
    line="$(grep -F -- "$1 {" <<< "$HELP" | head -1)"
    if [[ -z "$line" ]]; then warn "$1: no choice list in help; skipped '$2'"; return; fi
    set="${line#*\{}"; set="${set%%\}*}"
    if grep -qE "(^|,)$2(,|\$)" <<< "$set"; then ok "$1 $2"; else bad "$1 $2 not in {$set}"; fi
}
if [[ -n "$HELP" ]]; then
    for enum in --moe-backend --all2all-backend --kv-cache-dtype; do
        while IFS= read -r val; do
            [[ -n "$val" ]] && check_choice "$enum" "$val"
        done < <(grep -rhE "^[[:space:]]*$enum " "$REPO_ROOT"/servers/*.sh "$REPO_ROOT"/common.sh \
                 | awk -v f="$enum" '{for(i=1;i<NF;i++) if($i==f){print $(i+1)}}' | sort -u)
    done
else
    warn "no help text; skipped enum value check"
fi

echo "== Attention backends (opt03*) =="
if command -v python3 >/dev/null 2>&1; then
    python3 - <<'PY' 2>/dev/null && bk_ok=1 || bk_ok=0
import sys
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
    names = {b.name for b in E}
except Exception as e:
    print("registry import failed:", e); sys.exit(3)
want = ["FLASHINFER_MLA_SPARSE", "FLASHMLA_SPARSE"]
missing = [w for w in want if w not in names]
print("available:", ", ".join(sorted(names)))
if missing:
    print("MISSING:", ", ".join(missing)); sys.exit(4)
PY
    if [[ "${bk_ok:-0}" == "1" ]]; then ok "FLASHINFER_MLA_SPARSE / FLASHMLA_SPARSE present"; else
        warn "could not confirm opt03 attention backends — check names with the snippet in SERVERS.md"; fi
fi

echo "== Optional: W&B =="
if [[ -n "${WANDB_API_KEY:-}" ]]; then ok "WANDB_API_KEY set"; else warn "WANDB_API_KEY unset (set it in .env or run with WANDB=0)"; fi
python3 -c "import wandb" 2>/dev/null && ok "wandb importable" || warn "wandb not installed (run.sh pip-installs it on demand)"

echo
if [[ "$FAIL" -eq 0 ]]; then echo "PREFLIGHT: PASS — good to go."; else echo "PREFLIGHT: FAIL — fix the FAIL lines above before a long run."; fi
exit "$FAIL"
