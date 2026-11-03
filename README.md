# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p`) with SGLang-TorchTPU

End-to-end serving recipes, multi-replica colocated gateways, `DFLASH` speculative decoding configurations, and empirical benchmarks for Meta's Muse model family on Cloud TPU **`v7x`** and **`v5p` / `v5p-32`** using **SGLang-TorchTPU** (`sglang` `b3d718cb`, TorchTPU `0.2.0`):

- **[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)** (`59.6 GB` BF16; dense 30B, 52 layers = 39 SWA-2048 + 13 global NoPE, 32 Q / 2 KV heads, `131K` context) + **[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)** (`DFLASH` block-16 draft).
- **[`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)** (`533.0 GB` NVFP4) & **[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)** (`1,644.4 GB` BF16; 816B total / 42B active MoE, 256 experts top-8, 62 layers = 46 SWA-2048 + 16 global GQA, 128 Q / 16 KV heads, `262K` context) + **[`RadixArk/Muse-Spark-Open-DFlash2`](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)** (`DFLASH` block-8 draft).

> **Live sweep status:** Full `Muse-Glimmer-30B` sweeps on `v5p` and `v7x` (`1K` through `128K` context, `DFLASH-16` / `DFLASH-8`, and colocated multi-replica gateway) are measured below; `Muse-Spark-Open-816B-A42B` (`NVFP4` + `DFlash2` + Pallas Decode Megakernel on `v7x`, and `BF16` `TP32 EP32` across 8-host `v5p-32`) and additional `v7x` `FP8` / multi-replica `Muse-Glimmer-30B` sweeps are actively running. Full per-cell tables: [`results/SUMMARY.md`](./results/SUMMARY.md) · machine-readable records: [`results/summary.json`](./results/summary.json).

---

## Headline Measured Results — `Muse-Glimmer-30B` (`1024` in / `1024` out, fixed-length decode with EOS ignored)

| Hardware | Variant / Topology | Chips | Conc. 1: Out tok/s (TPOT p50) | Conc. 4 | Conc. 16 | Conc. 32 | Conc. 64 / 128 Peak | Spec Accept Len |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8_dflash16` (BF16 TP8 + **DFLASH-16**) | 4 | **677.6** (`1.21 ms`) | **2,207.3** (`1.40 ms`) | **4,482.4** (`2.16 ms`) | 1,230.6 | — | **5.44 – 6.30** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8` (BF16 TP8, non-speculative) | 4 | **185.8** (`5.30 ms`) | 697.0 (`5.46 ms`) | 2,207.8 (`6.72 ms`) | 3,359.2 (`8.76 ms`) | **4,795.4** (c=64) | — |
| **TPU `v5p` (1 host)** | `glimmer_bf16_2xtp2_gateway` (BF16 **2×TP2 Colocated Gateway**) | 4 | 66.6 (`14.83 ms`) | 288.9 (`13.60 ms`) | 1,014.7 (`14.44 ms`) | 1,852.3 (`16.13 ms`) | **4,691.7** (c=128) | — |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash8` (BF16 TP4 + **DFLASH-8**) | 4 | **460.2** (`1.78 ms`) | 1,402.7 (`2.18 ms`) | **2,847.3** (`3.57 ms`) | **3,472.7** (`6.21 ms`) | — | **4.70 – 5.12** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash16` (BF16 TP4 + **DFLASH-16**) | 4 | **516.5** (`1.52 ms`) | **1,540.8** (`2.15 ms`) | 2,511.3 (`3.51 ms`) | 2,685.7 (`7.50 ms`) | — | **5.64 – 6.12** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash16_greedy` (BF16 TP4 + DFLASH-16, T=0) | 4 | **518.3** (`1.51 ms`) | **1,583.5** (`2.11 ms`) | 2,646.0 (`3.72 ms`) | — | — | **5.64 – 6.20** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4` (BF16 TP4, non-speculative) | 4 | 119.1 (`8.27 ms`) | 439.1 (`8.55 ms`) | 1,452.8 (`10.43 ms`) | 2,302.4 (`12.92 ms`) | **3,322.0** (c=64) | — |
| **TPU `v5p` (2 chips)** | `glimmer_bf16_tp2` (BF16 TP2, half-host = 2 chips) | 2 | 66.6 (`14.81 ms`) | 274.2 (`13.83 ms`) | 936.7 (`16.18 ms`) | 1,542.9 (`18.53 ms`) | **2,340.9** (c=64) | — |
| **TPU `v5p` (1 host)** | `glimmer_bf16w_fp8kv_tp4` (BF16 weights + FP8 e4m3 KV, TP4) | 4 | 118.7 (`8.29 ms`) | — | 1,439.0 (`10.56 ms`) | — | 3,269.3 (c=64) | — |
| **TPU `v5p` (2 chips)** | `glimmer_fp8_tp2` (Online FP8 W8A8, emulated on `v5p`, TP2) | 2 | 22.4 (`44.30 ms`) | 87.4 (`44.40 ms`) | 327.4 (`47.47 ms`) | 595.9 (`51.01 ms`) | 998.5 (c=64) | — |

### Key Performance Highlights

1. **DFLASH Speculative Decoding (`3.65x`–`4.34x` Single-Stream Speedup & Higher Batch Frontier):**
   - On **`v7x` TP8**, `DFLASH-16` with `Muse-Glimmer-30B-assistant` reaches **`677.6` output tok/s (`1.21 ms` TPOT p50, `79 ms` TTFT)** at concurrency 1 (**`3.65x`** over non-speculative `185.8` tok/s) and **`4,482.4` output tok/s (`2.16 ms` TPOT p50, accept length `6.30`)** at concurrency 16 (**`2.03x`** over non-speculative conc=16 `2,207.8` tok/s).
   - On **`v5p` TP4**, `DFLASH-16` reaches **`516.5` output tok/s (`1.52 ms` TPOT p50, accept length `5.64`)** at concurrency 1 (**`4.34x`** over `119.1` tok/s), while `DFLASH-8` scales to **`3,472.7` output tok/s at concurrency 32 (`6.21 ms` TPOT)** — surpassing non-speculative `TP4` at concurrency 64 (`3,322.0` tok/s at `17.09 ms` TPOT) with **`2.75x` lower per-token latency**.
2. **Colocated Multi-Replica Serving (`1.41x` Single-Host Throughput Gain on `v5p`):**
   - Because `Muse-Glimmer-30B` (`59.6 GB` BF16) fits comfortably in 2 `v5p` chips (`190 GB` HBM) and eliminates cross-host/full-host collective overhead, running **two independent `TP2` servers on one 4-chip `v5p` host** (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) fronted by `scripts/muse_multi_replica_proxy.py` reaches **`4,691.7` output tok/s (`1,172.9` output tok/s/chip)** at concurrency 128 — **`1.41x` higher throughput** than a single `TP4` instance (`3,322.0` output tok/s) on the same 4 chips.

---

## Long-Context Scaling (`4K` → `128K` Input Tokens, `1024` Output Tokens)

39 of `Muse-Glimmer-30B`'s 52 layers use `2048`-token sliding-window attention (SWA) and only 13 layers maintain full global KV cache, keeping decode TPOT nearly constant from `1K` out to `127K` context:

| Hardware & Variant | `4096` in (c=1 / Peak) | `32768` in (c=1 / Peak) | `65536` (`64K`) in (c=1 / c=4) | `126976` (`~128K`) in (c=1) |
|---|---|---|---|---|
| **`v7x` `glimmer_bf16_tp8`** | `179.6` (`5.31 ms`) / **`2,907.3`** (c=64, `14,534` tot tok/s) | `154.1` (`5.40 ms`, TTFT `1.11 s`) / **`526.1`** (c=8, `17,359` tot tok/s) | `128.7` (`5.50 ms`, TTFT `2.31 s`) / **`269.5`** (`17,518` tot tok/s) | **`94.0` out tok/s** (`5.75 ms` TPOT, TTFT `5.01 s`, `11,753` tot tok/s) |
| **`v7x` `glimmer_bf16_tp8_dflash16`** | **`618.4`** (`1.13 ms`) / **`2,564.5`** (c=16, `12,820` tot tok/s) | _(running with `swa_full_tokens_ratio=1.0`)_ | _(running)_ | _(running)_ |
| **`v5p` `glimmer_bf16_tp4_dflash16`** | **`473.9`** (`1.40 ms`, acc `5.70`) / **`1,838.8`** (c=32) | **`296.8`** (`0.87 ms` TPOT, TTFT `2.04 s`, acc `5.63`) / **`373.7`** (c=4, `12,331` tot tok/s) | — | — |
| **`v5p` `glimmer_bf16_tp4_dflash8`** | **`418.0`** (`1.53 ms`, acc `4.85`) / **`2,233.4`** (c=32, `11,165` tot tok/s) | **`262.9`** (`1.42 ms` TPOT, TTFT `2.03 s`, acc `4.77`) / **`339.0`** (c=4, `11,186` tot tok/s) | — | — |
| **`v5p` `glimmer_bf16_tp4`** | `114.4` (`8.30 ms`) / **`1,964.6`** (c=64, `9,821` tot tok/s) | `96.3` (`8.42 ms`, TTFT `2.00 s`) / **`315.0`** (c=8, `10,393` tot tok/s) | `78.4` (`8.55 ms`, TTFT `4.31 s`) / **`153.3`** (`9,964` tot tok/s) | **`54.9` out tok/s** (`8.77 ms` TPOT, TTFT `9.68 s`, `6,865` tot tok/s) |

---

## Key TPU Serving Configurations & Architectural Notes

1. **Colocated Multi-Replica Serving (`SGLANG_TPU_PD_COLOCATED=1`):**
   - Plain `--dp-size > 1` inside a single SGLang-TorchTPU server process attempts to open `/dev/vfio/0` from multiple scheduler workers and fails. Setting `SGLANG_TPU_PD_COLOCATED=1`, per-replica `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=replica_<offset>`, and `--base-gpu-id <r * tp_size>` cleanly partitions chips on a single TPU host across independent server instances behind [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py) (async `aiohttp` least-inflight streaming gateway).
2. **Text-Only Configuration on `Muse-Glimmer-30B` vs `Muse-Spark-Open-816B-A42B`:**
   - `Muse-Glimmer-30B` requires `--json-model-override-args '{"language_model_only": true}'` so `torch.compile` skips tracing multimodal `msgspec.Struct` schedule items (whereas `--language-only` is for encoder disaggregation and is rejected).
   - `Muse-Spark-Open-816B-A42B` (`MuseSparkForConditionalGeneration`) defaults `enable_multimodal=False` in SGLang's model config automatically and uses `--reasoning-parser muse --tool-call-parser muse`.
3. **DFLASH Draft Overrides & Hybrid SWA Pool Sizing:**
   - `Muse-Glimmer-30B-assistant/config.json` omits `vocab_size`; `run_muse_sweep.py` injects `"vocab_size": 202048` into the local `/dev/shm` draft copy at startup.
   - `RadixArk/Muse-Spark-Open-DFlash2` (`5` SWA layers, block size `8`) is configured with `"is_hybrid_swa": true` and `"hybrid_layer_pattern": [1, 1, 1, 1, 1]` so draft KV slots share the SWA pool rather than allocating `262K` full-attention KV slots.
   - For high-concurrency (`>=32`) or `32K`–`128K` speculative decoding on hybrid SWA models, setting `--swa-full-tokens-ratio 0.5`–`1.0` prevents SWA pool eviction stalls.
4. **Precision & Attention Head Sharding:**
   - `Muse-Glimmer-30B` has `num_key_value_heads=2`; unfused online FP8 (`--quantization fp8`) requires `tp_size <= 2` (`4×FP8-TP2` on `v7x`), while BF16 `QKVParallelLinear` replicates KV heads cleanly up to `TP8`. `v5p` lacks hardware FP8 MXUs (emulated FP8 is ~3x slower than BF16), whereas `v7x` provides native FP8 and NVFP4 (`modelopt_fp4`) support.
   - `Muse-Spark-Open-816B-A42B` has `16` KV heads (replicating cleanly to `TP32` across an 8-host `v5p-32` slice) and `256` routed experts (`top-8`), supporting either expert-TP (`ep_size=1` with `pallas_tpu_fused_rs` on single-host `v7x`) or expert parallelism (`ep_size=32`, 8 whole experts per chip across `v5p-32`, with BF16 experts quantized to FP8 on load).

---

## Repository Layout

- [`scripts/run_muse_sweep.py`](./scripts/run_muse_sweep.py) — Unified SGLang-TorchTPU benchmark harness supporting single-host, colocated multi-replica (`replicas > 1`), and multi-host TPU slice (`nnodes > 1`) sweeps, `/dev/shm` or GCSFuse (`weight_loader_drop_cache_after_load`) weight loading, and `DFLASH` draft overrides.
- [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py) — Async `aiohttp` least-inflight HTTP reverse proxy for colocated multi-replica TPU servers.
- [`scripts/stage_spark_sharded.py`](./scripts/stage_spark_sharded.py) — Multi-pod Indexed Job HuggingFace-to-GCS parallel shard downloader with byte-exact completeness verification.
- [`scripts/collect_muse_results.py`](./scripts/collect_muse_results.py) — Aggregates per-run `summary.json` artifacts from GCS into [`results/SUMMARY.md`](./results/SUMMARY.md) and [`results/summary.json`](./results/summary.json).
- [`scripts/plan_*.json`](./scripts/) — Declarative sweep plans for `Muse-Spark-Open-816B-A42B` (`v7x` NVFP4 & `v5p-32` BF16) and `Muse-Glimmer-30B` (`v7x` & `v5p`).
- [`k8s/`](./k8s/) — Sanitized GKE Job manifests for single-host `v7x`/`v5p` sweeps, 8-host Indexed `v5p-32` TPU slice serving, and sharded GCS model staging.
- [`results/`](./results/) — Consolidated [`SUMMARY.md`](./results/SUMMARY.md) and [`summary.json`](./results/summary.json).
