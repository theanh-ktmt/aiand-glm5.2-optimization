# (AI&) GLM 5.2 FP8 Optimizations

Benchmark harness to find the best **vLLM 0.23.x** server configuration for
**`zai-org/GLM-5.2-FP8`** on **8x NVIDIA B300**, maximizing throughput while
keeping TTFT / TPOT acceptable.

Method follows [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)
(vendored as a git submodule under `third_party/InferenceX`): random dataset,
saturated server (`--request-rate inf`), `--ignore-eos`, prefix caching **off**,
swept across concurrency. See
[the requirements page](https://moreh.atlassian.net/wiki/spaces/MOREH/pages/2432991505)
(source of truth).

## Layout

```
common.sh            shared config + mandatory flags + server lifecycle helpers
servers/*.sh         one launch script per configuration (baseline + 12 opt groups)
bench/bench.sh       InferenceX-style sweep (MTP and non-MTP) against a live server
run.sh               one config end-to-end: launch -> sweep -> teardown -> CSV
run_all.sh           the whole campaign: baseline FULL + every opt SUBSET -> all.csv
aggregate.py         result JSONs -> CSV (sheet-pasteable)
servers/final.sh     TEMPLATE for the recommended config (fill in winners)
eval/quality_check.sh  MMLU-Pro: baseline vs recommended config
eval/parse_mmlu.py   lm-eval results -> accuracy comparison table + Pass?
results/<config>/    per-config outputs: *.json, server.log (with launch command),
                     bench.log, gpu.csv  (git-ignored)
third_party/InferenceX  benchmark engine (submodule)
SERVERS.md           index of every server script and what it changes
```

## Setup

```bash
git clone --recurse-submodules <this-repo>
cd aiand-glm5.2-optimization
# or, if already cloned:
git submodule update --init --recursive
```

Run inside the vLLM 0.23.x container on a B300 node. Optional: pre-stage weights
and `export MODEL_PATH=/path/to/GLM-5.2-FP8` to skip the HF download.

## Benchmark method (InferenceX)

- **Prefix caching is always off** (`--no-enable-prefix-caching`) - otherwise
  repeated random/warmup prefixes are served from cache, prefill is skipped, and
  throughput is inflated.
- Random dataset, `--random-range-ratio 0.8`, `--request-rate inf`,
  `--ignore-eos`, `--num-warmups = 2x concurrency`, `--num-prompts = 10x concurrency`.
- Scenarios: **1k/1k** (`1024:1024`) and **8k/1k** (`8192:1024`).
- Concurrency sweep: `1 2 4 8 16 32 64 128` (FULL) or `1 16 128` (SUBSET, for
  screening — 128 also exercises the batch-size-gated MTP disable).
- **MTP vs non-MTP** differ on the client by exactly one flag: MTP adds
  `--use-chat-template` (spec decoding was trained on chat-formatted input;
  raw prompts silently tank acceptance length). The baseline uses MTP(5), so it
  is benchmarked in MTP mode.

Tracked metrics (simplified): **Output throughput**, **TTFT** (mean/median/P90),
**TPOT** (mean/median/P90).

## Logs

Each config writes everything to `results/<config>/`:

- `server.log` - vLLM server output; the **first lines record the exact
  `vllm serve` command and relevant env vars** used to launch it.
- `bench.log` - full console output of the benchmark sweep.
- `gpu.csv` - per-second GPU power/clocks/util (via InferenceX `start_gpu_monitor`).
- `*.json` - one `benchmark_serving` result per (scenario, concurrency).

## Durability (Weights & Biases)

These runs are long and unattended on on-demand cloud boxes, so each config's
results are pushed to W&B the moment it finishes — if the instance dies later,
nothing is lost. `wandb_sync.py` logs (one run per config):

- per-concurrency curves (throughput / TTFT / TPOT, x-axis = concurrency) so
  configs overlay in the dashboard;
- a summary table of every row;
- the raw JSONs + `server.log` / `bench.log` / `gpu.csv` + the CSV as a run artifact.

Runs automatically at the end of `run.sh` / `run_all.sh`. Setup on the box:

```bash
cp .env.example .env      # then put your WANDB_API_KEY in .env (git-ignored)
```

`.env` is auto-loaded; `wandb` is auto-installed if missing. Disable with `WANDB=0`.
Project defaults to `glm5.2-fp8-opt` (override `WANDB_PROJECT`). Nothing in
`results/` is committed to git — W&B is the store.

## Usage

### One config, end-to-end

```bash
bash run.sh baseline full          # baseline, full sweep
bash run.sh opt04_moe_deepgemm     # an optimization, subset sweep (default)
# -> results/<config>/*.json  and  results/<config>.csv
```

### Manual (launch + benchmark in separate shells)

```bash
# shell A - launch and keep the server up (command is echoed + logged)
bash servers/opt09_flashinfer_sampler.sh

# shell B - benchmark the running server
CONFIG=opt09_flashinfer_sampler bash bench/bench.sh \
    --config opt09_flashinfer_sampler --mode mtp --sweep subset
```

### Whole campaign

```bash
bash run_all.sh                    # baseline FULL + every opt SUBSET
python3 aggregate.py results --baseline baseline --out results/all.csv
```

### Final config + quality check

After screening, edit `servers/final.sh` (a template; each slot maps to an
optimization group) to combine the winning flags, then:

```bash
bash run.sh final full                      # full throughput/latency sweep
bash eval/quality_check.sh baseline final   # MMLU-Pro accuracy, baseline vs final
```

`quality_check.sh` launches each config, runs lm-eval `mmlu_pro` against the chat
endpoint (auto-installing lm-eval-harness via InferenceX), tears down, and prints
a table with the accuracy delta and a Pass? verdict (fails if the candidate
regresses more than 1.0 accuracy point; tune with `--threshold`). Results also go
to `results/quality_check.csv`. GLM-5.2-FP8 is text-only, so MMMU-Pro is skipped.

## Workflow

1. Benchmark **baseline** (FULL sweep) - the reference curve.
2. For each optimization, run a **SUBSET** sweep (`opt*`) and compare to baseline.
3. Collect the winners into `servers/final.sh` and run a **FULL** sweep.
4. Run `eval/quality_check.sh` (MMLU-Pro) on the final config vs baseline.

See `SERVERS.md` for the full list of server scripts and what each one changes.
