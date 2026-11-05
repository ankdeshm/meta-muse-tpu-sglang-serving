# Raw benchmark results

This folder holds the raw files behind every number in the [README](../../README.md) and [SUMMARY.md](../SUMMARY.md).
Each subfolder is one benchmark run (`<model>_sglang_<hardware>_<hosts>_<label>_<UTC start>`).

| File | Contents |
|---|---|
| `bench_<variant>_i<input>_o<output>_c<concurrency>.jsonl` | Raw SGLang `bench_serving` output for one cell: request counts, throughput, TTFT / TPOT / ITL / end-to-end latency percentiles, speculative accept length, server version and arguments. |
| `summary.json` | Run record from [`scripts/run_muse_sweep.py`](../../scripts/run_muse_sweep.py) (or [`scripts/run_v5p_2xtp2_gateway_bench.py`](../../scripts/run_v5p_2xtp2_gateway_bench.py)): the sweep plan, the server arguments per variant, per-cell metrics and the chat sanity check. |
| `backfill_sanity_<variant>.json` | Chat sanity reply recorded after the sweep, where present. |
| `server_log_excerpt_<variant>.txt` | Rank-0 server log lines showing the MoE kernel backend, weight memory per device, KV cache allocation and token budget; for the 8-host `v5p-32` servers, every host's weight-loading line. |
| [`index.csv`](./index.csv) | One row per measured cell, with chips, weight / KV precision, TP / EP / replicas, speculative decoding, the headline metrics and the path of the raw file. |

## Conventions

- **Workload:** random token prompts with fixed input / output lengths (`--dataset-name random --random-range-ratio 1.0`); every request generates exactly `output_len` tokens (EOS ignored). Cells with 1K / 4K prompts use 4 × concurrency requests (8 at `c=1`); cells with prompts of 32K tokens or more use 2–16 requests (2–4 at `c=1`).
- **Output tok/s** = total output tokens / benchmark duration. **Total tok/s** includes input tokens.
- **TTFT / TPOT** are per-request medians (`median_ttft_ms`, `median_tpot_ms`).
- **Chips** counts physical TPU chips: a `v7x` chip has 2 TensorCores, so `TP2` = 1 chip and `TP8` = 4 chips (one host); a `v5p` chip is one device, so `TP4` = 4 chips (one host) and `TP32` = 32 chips (`v5p-32`).
- **Speculative decoding** rows use greedy decoding (temperature 0), except `glimmer_bf16_tp4_dflash16_t1` (T=1.0, top-p 0.95). `accept_length` is the server's running average since start-up.
- `YOUR_GCS_BUCKET` and `YOUR_GCP_PROJECT` stand in for the storage bucket and container registry project.

## Regenerating

```bash
# Recompute the per-cell records (results/summary.json, results/SUMMARY.md) from this folder
python3 scripts/collect_muse_results.py /tmp/muse_summary --local results/raw

# Re-export from a local mirror of the GCS run folders
python3 scripts/export_raw_results.py <mirror_of_benchmark-results> results/raw
```
