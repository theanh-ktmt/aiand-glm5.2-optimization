# Server scripts

Every script under `servers/` launches one vLLM configuration. All of them carry
the mandatory harness flags from `common.sh`
(`--no-enable-prefix-caching`, `--trust-remote-code`, `--max-model-len 16384`,
`--tool-call-parser glm47`, `--enable-auto-tool-choice`, `--reasoning-parser glm45`)
plus their own optimization-specific flags shown below.

Run one directly to launch + hold the server:  `bash servers/<name>.sh`
Run one end-to-end (launch + bench + CSV):       `bash run.sh <name> [full|subset]`

Bench mode is `mtp` (client adds `--use-chat-template`) for everything except
`ref_nonmtp`, because the Day-0 baseline itself uses MTP(5).

| # | Script | Optimization | Key flags added / changed | Bench |
|---|--------|--------------|---------------------------|-------|
| 0 | `baseline.sh` | **Baseline (Day 0)** | `--tensor-parallel-size 8` `--speculative-config '{"method":"mtp","num_speculative_tokens":5}'` | mtp |
| 1a | `opt01a_tp8ep.sh` | Parallelism: TP8 + EP | `--enable-expert-parallel` | mtp |
| 1b | `opt01b_dp8ep.sh` | Parallelism: DP8 + EP | `--data-parallel-size 8 --enable-expert-parallel` (base for 5/6/7) | mtp |
| 2a | `opt02a_max_num_seqs.sh` | Hyperparam | `--max-num-seqs ${MAX_NUM_SEQS:-256}` | mtp |
| 2b | `opt02b_max_num_batched_tokens.sh` | Hyperparam | `--max-num-batched-tokens ${MNBT:-16384}` | mtp |
| 2c | `opt02c_gpu_memory_utilization.sh` | Hyperparam | `--gpu-memory-utilization ${GMU:-0.92}` | mtp |
| 3a | `opt03a_attn_flashinfer_trtllm.sh` | Attention backend | `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA_SPARSE` (FlashInfer + TRT-LLM decode) | mtp |
| 3b | `opt03b_attn_flashmla.sh` | Attention backend | `VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE` | mtp |
| 4a | `opt04_moe_triton.sh` | MoE backend | `--moe-backend triton` | mtp |
| 4b | `opt04_moe_cutlass.sh` | MoE backend | `--moe-backend cutlass` | mtp |
| 4c | `opt04_moe_deepgemm.sh` | MoE backend | `--moe-backend deep_gemm` + `VLLM_USE_DEEP_GEMM=1` | mtp |
| 4d | `opt04_moe_flashinfer_trtllm.sh` | MoE backend | `--moe-backend flashinfer_trtllm` | mtp |
| 5a | `opt05_a2a_naive.sh` | (DP8EP) All2All | `--all2all-backend naive` | mtp |
| 5b | `opt05_a2a_pplx.sh` | (DP8EP) All2All | `--all2all-backend pplx` | mtp |
| 5c | `opt05_a2a_deepep_low_latency.sh` | (DP8EP) All2All | `--all2all-backend deepep_low_latency` | mtp |
| 5d | `opt05_a2a_deepep_high_throughput.sh` | (DP8EP) All2All | `--all2all-backend deepep_high_throughput` | mtp |
| 5e | `opt05_a2a_flashinfer_nvlink.sh` | (DP8EP) All2All | `--all2all-backend flashinfer_nvlink_two_sided` | mtp |
| 6 | `opt06_eplb.sh` | (DP8EP) EPLB | `--enable-eplb --eplb-config '{…,"num_redundant_experts":${EPLB_REDUNDANT:-16}}'` | mtp |
| 7 | `opt07_dbo.sh` | (DP8EP) DBO | `--enable-dbo --dbo-decode-token-threshold … --dbo-prefill-token-threshold …` | mtp |
| 8a | `opt08_mtp1.sh` | MTP | `num_speculative_tokens=1` | mtp |
| 8b | `opt08_mtp2.sh` | MTP | `num_speculative_tokens=2` | mtp |
| 8c | `opt08_mtp3.sh` | MTP | `num_speculative_tokens=3` | mtp |
| 8d | `opt08_mtp4.sh` | MTP | `num_speculative_tokens=4` | mtp |
| 8e | `opt08_mtp_disable_bs64.sh` | MTP | MTP(5), disabled above batch 64 (`num_speculative_tokens_per_batch_size`) | mtp |
| 8f | `opt08_mtp_disable_bs128.sh` | MTP | MTP(5), disabled above batch 128 | mtp |
| 9 | `opt09_flashinfer_sampler.sh` | Sampler | `VLLM_USE_FLASHINFER_SAMPLER=1` | mtp |
| 10a | `opt10a_cudagraph_full_piecewise.sh` | CUDA graph | `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` | mtp |
| 10b | `opt10b_cudagraph_tuned_capture.sh` | CUDA graph | FULL_AND_PIECEWISE + tuned `cudagraph_capture_sizes` | mtp |
| — | `ref_nonmtp.sh` | Reference | MTP disabled — quantifies MTP's contribution | **nonmtp** |

## Notes on flag names (vLLM 0.23.x verified)

- **MoE backend** (`--moe-backend`): valid values include
  `triton`, `cutlass`, `deep_gemm`, `flashinfer_trtllm`, `flashinfer_cutlass`, …
- **All2All** (`--all2all-backend`): `naive`, `pplx`, `deepep_low_latency`,
  `deepep_high_throughput`, `flashinfer_nvlink_two_sided` (a.k.a. flashinfer_nvlink),
  `flashinfer_all2allv`.
- **Attention** (`VLLM_ATTENTION_BACKEND`): GLM-5.2 uses sparse MLA (DSA), so the
  relevant backends are `FLASHINFER_MLA_SPARSE` (FlashInfer + TRT-LLM decode) and
  `FLASHMLA_SPARSE`. Adjust if the installed build exposes different names
  (`python -c "from vllm.v1.attention.backends.registry import AttentionBackendEnum as E; print([b.name for b in E])"`).
- **MTP "disable by batch size"** is expressed as a
  `num_speculative_tokens_per_batch_size` schedule of `(start, end, num_spec)`
  tuples — there is no standalone `disable_by_batch_size` flag in 0.23.x.
- **EPLB**: configured via a single `--eplb-config` JSON
  (`window_size`, `step_interval`, `num_redundant_experts`, …).

These are screening configs: tune the placeholder values (e.g. `MAX_NUM_SEQS`,
`MNBT`, `GMU`, `EPLB_REDUNDANT`) via the env vars shown, sweep, and keep the winners.
