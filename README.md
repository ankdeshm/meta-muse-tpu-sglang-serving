# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p-32`) with SGLang-TorchTPU

End-to-end serving recipes, **62-layer Pallas Decode Megakernel** configurations, **`DFLASH` block-speculative decoding** workflows, **colocated multi-replica gateways**, and **multi-host `v5p-32` expert-parallel** benchmarks for Meta's Muse model family on Cloud TPU **`v7x` (Ironwood)** and **`v5p` / `v5p-32`** using **SGLang-TorchTPU** (`sglang` `b3d718cb`, TorchTPU `0.2.0`):

- **[`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)** (`533.0 GB` NVFP4) & **[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)** (`1,644.4 GB` BF16; 816B total / 42B active MoE, 256 routed experts top-8 + shared expert, 62 layers = 46 SWA-2048 + 16 global GQA, 128 Q / 16 KV heads, `262K` context) + **[`RadixArk/Muse-Spark-Open-DFlash2`](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)** (`DFLASH` block-8 draft).
- **[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)** (`59.6 GB` BF16; dense 30B, 52 layers = 39 SWA-2048 + 13 global NoPE, 32 Q / 2 KV heads, `131K` context) + **[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)** (`DFLASH` block-16 draft).

Full per-cell benchmark tables (32 serving variants across 52 GKE runs): [`results/SUMMARY.md`](./results/SUMMARY.md) · machine-readable records: [`results/summary.json`](./results/summary.json).

---

## 1. Headline Results — `Muse-Spark-Open-816B-A42B` (`1024` in / `1024` out, EOS ignored)

All `Muse-Spark-Open-816B-A42B` speculative runs use **real `RadixArk/Muse-Spark-Open-DFlash2` draft generation & target verification** (achieving **`6.25 – 7.17` accepted tokens per step**, substantially exceeding the `4.05` simulated acceptance assumed in synthetic benchmarks).

| Hardware | Variant / Topology | Chips | Conc. 1: Out tok/s (TPOT p50) | Conc. 4 | Conc. 8 | Conc. 16 | Conc. 32 / 64 Peak | Spec Accept Len |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_dflash8_megakernel` (**62-Layer Pallas Decode Megakernel + DFLASH-8**) | 4 | **761.9** (`0.97 ms`) | **955.8** (`3.29 ms`) | 966.5 (`3.28 ms`) | **998.4** (`3.31 ms`) | — (`mrr=4`) | **6.25 – 6.74** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_dflash8` (NVFP4 TP8 EP1 + **DFLASH-8**, FP8 KV) | 4 | **355.2** (`2.35 ms`) | 664.1 (`4.43 ms`) | **974.5** (`6.18 ms`) | **1,360.2** (`9.12 ms`) | **1,861.9** (c=32, `465.5`/chip) | **6.71 – 6.99** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1` (NVFP4 TP8 **EP1 Expert-TP** `pallas_tpu_fused_rs`, BF16 KV) | 4 | **91.5** (`10.78 ms`) | 261.1 (`14.78 ms`) | 373.6 (`20.64 ms`) | 522.3 (`29.35 ms`) | **1,036.2** (c=64, `259.0`/chip) | — |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_bs8_recipe` (NVFP4 TP8 EP1, RadixCache on, `mrr=8`) | 4 | 91.4 (`10.79 ms`) | 263.8 (`14.81 ms`) | 381.0 (`20.52 ms`) | 381.1 (`20.54 ms`) | 384.6 (c=32) | — |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep8` (NVFP4 TP8 **EP8** 32 whole experts/TC, FP8 KV) | 4 | 57.7 (`17.14 ms`) | 141.2 (`27.68 ms`) | — | 286.4 (`54.40 ms`) | 421.3 (c=32) | — |
| **TPU `v5p-32` (8 hosts)** | `spark_bf16_v5p32_tp32_ep32_dflash8` (BF16 **TP32 EP32 + DFLASH-8**) | 32 | **148.0** (`5.97 ms`) | **382.2** (`7.74 ms`) | **685.9** (`8.68 ms`) | **1,119.8** (`11.44 ms`) | **1,540.1** (c=32) | **6.64 – 7.02** |
| **TPU `v5p-32` (8 hosts)** | `spark_bf16_v5p32_tp32_ep32` (BF16 **TP32 EP32**, 8 experts/chip, non-speculative) | 32 | 40.0 (`24.81 ms`) | 136.4 (`28.89 ms`) | 242.4 (`32.37 ms`) | 414.3 (`37.57 ms`) | **1,121.5** (c=64) | — |

### Key `Muse-Spark-Open-816B-A42B` Findings

1. **Sub-Millisecond TPOT (`0.97 ms`, `761.9` out tok/s at `c=1`) on an 816B Parameter MoE via the 62-Layer Pallas Decode Megakernel + `DFLASH-8`:**
   - Enabling `SGLANG_TPU_MUSE_MEGAKERNEL=1` with `SGLANG_TPU_KV_PACK_HEADS=1` (`bfloat16` packed KV, `pallas_tpu_rpa_v3`, `max_running_requests=4` = `32` speculative verification rows) fuses all 62 decoder layers of `Muse-Spark-Open-816B-A42B-NVFP4` on a single 4-chip `v7x` host (`768 GB` HBM).
   - At `1024/1024 c=1`, single-stream output throughput reaches **`761.9` output tok/s (`190.5` tok/s/chip, `154 ms` TTFT p50, `0.97 ms` TPOT p50)** — **`8.33x` faster** than non-speculative `TP8 EP1` (`91.5` tok/s, `10.78 ms` TPOT) and **`2.15x` faster** than standard `DFLASH-8` (`355.2` tok/s, `2.35 ms` TPOT), scaling to **`955.8`–`998.4` output tok/s** at `c=4..16`.
2. **High-Concurrency Batch Frontier (`1,861.9` out tok/s = `465.5` out tok/s/chip) with Standard `DFLASH-8` + FP8 KV:**
   - When `max_running_requests` is raised beyond the megakernel's 32-row verification window (`spark_nvfp4_tp8_ep1_dflash8` with `fp8_e4m3` KV), batch throughput scales to **`1,861.9` output tok/s (`3,723.8` total tok/s, `12.76 ms` TPOT p50, `6.99` accept length)** at `c=32` on a single `v7x` host — **`1.80x` higher peak throughput** than non-speculative `c=64` (`1,036.2` tok/s) at **`4.5x` lower TPOT** (`12.76 ms` vs `57.48 ms`).
3. **Single-Host `EP=1` (`pallas_tpu_fused_rs`) Beats `EP=8` by `1.59x`–`1.85x` on `v7x`:**
   - On 1-host `v7x` (`8` TensorCores connected over high-bandwidth ICI), sharding all 256 experts tensor-parallel across the 8 ranks with `pallas_tpu_fused_rs` (`EP=1`) achieves **`91.5` → `742.7` out tok/s** (`c=1..32`), outperforming `EP=8` (`57.7` → `421.3` out tok/s) by **`1.59x`–`1.85x`** by eliminating token-routing load imbalance and separate all-to-all dispatch/combine collectives.
4. **Full `1.64 TB` BF16 Checkpoint Across 8-Host TPU `v5p-32` (`TP32 EP32` + `DFLASH-8`):**
   - Across 8 `v5p` hosts (`32` chips in `2x4x4` 3D torus, `3.04 TB` HBM), `TP32 EP32` places 8 whole experts per chip and serves the unquantized `1.64 TB` checkpoint out to **`258,048` (`~262K`) context**. Adding `DFLASH-8` boosts `c=1` throughput **`3.70x`** (`40.0` → **`148.0` out tok/s**, `24.81 ms` → **`5.97 ms` TPOT**) and lifts peak batch throughput from `1,121.5` (`c=64`) to **`1,540.1` out tok/s (`c=32`, `14.61 ms` TPOT)**.

### `Muse-Spark-Open-816B-A42B` Long-Context Scaling (`4K` → `258K` Input Tokens, `1024` Output Tokens)

| Hardware & Variant | `4096` in (`c=1` / Peak) | `32768` in (`c=1` / `c=4`) | `65536` (`64K`) in (`c=1` / Peak) | `131072` (`128K`) in (`c=1`) | `258048` (`~262K`) in (`c=1`) |
|---|---|---|---|---|---|
| **`v7x` Megakernel + `DFLASH-8`** | **`672.0`** (`0.96 ms`) / **`796.5`** (c=4) | **`188.9`** (`0.98 ms`, TTFT `4.44 s`) / `188.7` | **`98.0`** (`1.02 ms` TPOT, TTFT `9.44 s`, `6,373` tot tok/s, acc `7.05`) | — | — |
| **`v7x` `TP8 EP1` + `DFLASH-8` (FP8 KV)** | **`349.3`** (`2.33 ms`) / **`937.1`** (c=16, `4,685` tot) | **`151.6`** (`2.38 ms`) / **`189.1`** (`6,239` tot) | **`87.7`** (`2.41 ms`) / **`94.8`** (c=2, `6,165` tot) | **`44.4` out tok/s** (`2.49 ms` TPOT, TTFT `20.5 s`, acc `7.17`) | — |
| **`v7x` `TP8 EP1` (BF16 KV / Radix)** | `87.8` (`10.87 ms`) / **`698.8`** (c=64) | `64.9` (`11.17 ms`) / `121.4` (`4,005` tot) | `48.8` / **`99.8`** (Radix c=4, `6,488` tot) | `31.0` / **`44.2`** (Radix, TTFT `10.7 s`) | **`35.8` out tok/s** (Radix, `13.49 ms` TPOT, TTFT `14.8 s`, `9,055` tot) / `16.2` (no-Radix) |
| **`v5p-32` `TP32 EP32` + `DFLASH-8`** | **`149.6`** (`5.84 ms`) / **`754.5`** (c=16, `3,773` tot) | — | **`60.9`** (`5.91 ms`, TTFT `10.9 s`, acc `7.15`) / **`72.0`** (c=2, `4,679` tot) | — | — |
| **`v5p-32` `TP32 EP32` (BF16)** | `39.4` (`24.87 ms`) / **`785.3`** (c=64, `3,927` tot) | `34.8` (`25.00 ms`, TTFT `3.84 s`) / `90.5` | `30.5` (`25.15 ms`, TTFT `7.83 s`) / `66.4` (`4,317` tot) | `24.2` (`25.50 ms`) / **`33.8`** (c=2, `4,359` tot) | **`16.6` out tok/s** (`26.08 ms` TPOT, TTFT `34.9 s`, `4,207` tot) |

---

## 2. Headline Results — `Muse-Glimmer-30B` (`1024` in / `1024` out, EOS ignored)

| Hardware | Variant / Topology | Chips | Conc. 1: Out tok/s (TPOT p50) | Conc. 4 | Conc. 16 | Conc. 32 | Conc. 64 / 128 Peak | Peak Out tok/s/chip |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TPU `v7x` (1 host)** | `glimmer_fp8_4xtp2_gateway` (**4×FP8-TP2 Colocated Gateway**, FP8 KV) | 4 | 126.7 (`7.79 ms`) | 505.7 (`7.82 ms`) | 1,931.9 (`7.89 ms`) | 3,706.6 (`8.29 ms`) | **11,502.2** (c=128, `10.50 ms`) | **2,875.5** |
| **TPU `v7x` (1 chip)** | `glimmer_fp8_tp2` (FP8 W8A8 **TP2 = 1 chip**, FP8 KV) | 1 | 127.0 (`7.80 ms`) | 495.1 (`7.79 ms`) | 1,750.9 (`8.75 ms`) | 2,899.7 (`10.51 ms`) | **4,254.7** (c=64, `13.97 ms`) | **4,254.7** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_4xtp2_gateway` (BF16 **4×TP2 Colocated Gateway**) | 4 | 82.8 (`11.96 ms`) | 331.1 (`11.96 ms`) | 1,255.6 (`12.34 ms`) | 2,417.0 (`12.67 ms`) | **7,563.5** (c=128, `15.93 ms`) | **1,890.9** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_2xtp4_gateway` (BF16 **2×TP4 Colocated Gateway**) | 4 | 133.5 (`7.38 ms`) | 531.4 (`7.36 ms`) | 1,805.0 (`8.33 ms`) | 3,163.4 (`9.41 ms`) | **7,482.5** (c=128, `15.44 ms`) | **1,870.6** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8_dflash16` (BF16 TP8 + **DFLASH-16**, `5.44–6.30` acc) | 4 | **677.6** (`1.21 ms`) | **2,207.3** (`1.40 ms`) | **4,482.4** (`2.16 ms`) | 1,230.6 | — | 1,120.6 |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8` (BF16 TP8, non-speculative) | 4 | **185.8** (`5.30 ms`) | 697.0 (`5.46 ms`) | 2,207.8 (`6.72 ms`) | 3,359.2 (`8.76 ms`) | **4,795.4** (c=64, `12.12 ms`) | 1,198.8 |
| **TPU `v7x` (1 chip)** | `glimmer_bf16_tp2` (BF16 **TP2 = 1 chip**) | 1 | 82.9 (`11.96 ms`) | 315.2 (`12.32 ms`) | 1,123.7 (`13.77 ms`) | 1,900.5 (`15.92 ms`) | **3,087.8** (c=64, `18.87 ms`) | **3,087.8** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_2xtp2_gateway` (BF16 **2×TP2 Colocated Gateway**) | 4 | 66.6 (`14.83 ms`) | 288.9 (`13.60 ms`) | 1,014.7 (`14.44 ms`) | 1,852.3 (`16.13 ms`) | **4,691.7** (c=128, `24.18 ms`) | **1,172.9** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash8` (BF16 TP4 + **DFLASH-8**, `4.70–5.12` acc) | 4 | **460.2** (`1.78 ms`) | 1,402.7 (`2.18 ms`) | **2,847.3** (`3.57 ms`) | **3,472.7** (`6.21 ms`) | — | 868.2 |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash16` (BF16 TP4 + **DFLASH-16**, `5.64–6.20` acc) | 4 | **518.3** (`1.51 ms`) | **1,583.5** (`2.11 ms`) | 2,646.0 (`3.72 ms`) | 2,685.7 (`7.50 ms`) | — | 671.4 |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4` (BF16 TP4, non-speculative) | 4 | 119.1 (`8.27 ms`) | 439.1 (`8.55 ms`) | 1,452.8 (`10.43 ms`) | 2,302.4 (`12.92 ms`) | **3,322.0** (c=64, `17.09 ms`) | 830.5 |

### `Muse-Glimmer-30B` Long-Context Scaling (`4K` → `127K` Input Tokens, `1024` Output Tokens)

| Hardware & Variant | `4096` in (`c=1` / Peak) | `32768` in (`c=1` / Peak) | `65536` (`64K`) in (`c=1` / Peak) | `126976` (`~128K`) in (`c=1` / Peak) |
|---|---|---|---|---|
| **`v7x` `glimmer_fp8_4xtp2_gateway`** | `123.9` (`7.82 ms`) / **`5,593.0`** (c=64, `27,959` tot tok/s) | — | `90.4` (`8.02 ms`, TTFT `3.11 s`) / **`353.8`** (c=4, `22,996` tot) | `63.1` (`8.26 ms`) / **`250.8`** (c=4, **`31,349` tot tok/s**) |
| **`v7x` `glimmer_bf16_tp8_dflash16` (+ `longctx`)** | **`618.4`** (`1.13 ms`) / **`2,564.5`** (c=16, `12,820` tot) | **`469.6`** (`0.73 ms` TPOT, TTFT `1.09 s`, acc `7.04`) / **`573.0`** (c=4, `18,910` tot) | **`303.1`** (`0.74 ms` TPOT, TTFT `2.29 s`, acc `5.38`, `19,703` tot) | **`163.5` out tok/s** (`1.20 ms` TPOT, TTFT `5.02 s`, acc `5.52`, **`20,441` tot tok/s**) |
| **`v7x` `glimmer_bf16_2xtp4_gateway`** | `128.2` (`7.41 ms`) / **`3,466.6`** (c=64, `17,330` tot) | — | `88.8` (`7.63 ms`) / **`267.5`** (c=4, `17,389` tot) | `63.1` / **`126.0`** (c=2, `15,744` tot) |
| **`v7x` `glimmer_bf16_tp8`** | `179.6` (`5.31 ms`) / **`2,907.3`** (c=64, `14,534` tot) | `154.1` (`5.40 ms`, TTFT `1.11 s`) / **`526.1`** (c=8, `17,359` tot) | `128.7` (`5.50 ms`, TTFT `2.31 s`) / **`269.5`** (c=4, `17,518` tot) | **`94.0` out tok/s** (`5.75 ms` TPOT, TTFT `5.01 s`, `11,753` tot) |
| **`v5p` `glimmer_bf16_tp4_dflash16`** | **`473.9`** (`1.40 ms`, acc `5.70`) / **`1,838.8`** (c=32) | **`296.8`** (`0.87 ms` TPOT, TTFT `2.04 s`, acc `5.63`) / **`373.7`** (c=4, `12,331` tot) | — | — |
| **`v5p` `glimmer_bf16_2xtp2_gateway`** | `64.2` (`14.86 ms`) / **`2,043.9`** (c=64, `10,218` tot) | `54.7` / **`290.4`** (c=8, `9,583` tot) | `44.8` / **`140.3`** (c=4, `9,117` tot) | `31.4` / **`62.7`** (c=2, `7,836` tot) |
| **`v5p` `glimmer_bf16_tp4`** | `114.4` (`8.30 ms`) / **`1,964.6`** (c=64, `9,821` tot) | `96.3` (`8.42 ms`, TTFT `2.00 s`) / **`315.0`** (c=8, `10,393` tot) | `78.4` (`8.55 ms`, TTFT `4.31 s`) / **`153.3`** (`9,964` tot) | **`54.9` out tok/s** (`8.77 ms` TPOT, TTFT `9.68 s`, `6,865` tot) |

### Key `Muse-Glimmer-30B` Findings

1. **`11,502.2` Output tok/s (`2,875.5` tok/s/chip) on 1-Host TPU `v7x` via `4×FP8-TP2` Colocated Multi-Replica Gateway:**
   - Because `Muse-Glimmer-30B` has `num_key_value_heads=2`, single-chip `TP2` (`2` TensorCores = `1` `v7x` chip, `192 GB` HBM) eliminates all inter-chip ICI collectives AND unlocks native `v7x` FP8 W8A8 MXU execution (`--quantization fp8 --kv-cache-dtype fp8_e4m3`), delivering **`4,254.7` output tok/s on a single `v7x` chip** and **`11,502.2` output tok/s (`22,993` total tok/s)** across all 4 chips (`4×FP8-TP2` behind `muse_multi_replica_proxy.py`) — **`2.40x` higher throughput** than single-instance `BF16 TP8` (`4,795.4` tok/s) on the same host.
2. **Sub-Millisecond Long-Context Decode (`0.73–0.74 ms` TPOT at `32K`–`64K`, `1.20 ms` at `127K`) with `DFLASH-16`:**
   - Setting `--swa-full-tokens-ratio 1.0` eliminates hybrid SWA KV eviction stalls under long prompts, allowing `glimmer_bf16_tp8_dflash16_longctx` on `v7x` to sustain **`469.6` out tok/s (`0.73 ms` TPOT)** at `32K`, **`303.1` out tok/s (`0.74 ms` TPOT)** at `64K`, and **`163.5` out tok/s (`1.20 ms` TPOT, `20,441` total tok/s)** at `127K` context.

---

## 3. Critical TPU Serving Recipes & Architectural Notes

1. **62-Layer `Muse-Spark` Pallas Decode Megakernel (`SGLANG_TPU_MUSE_MEGAKERNEL=1`):**
   - Requires packed BF16 KV layout (`--kv-cache-dtype bfloat16 --attention-backend pallas_tpu_rpa_v3` + `SGLANG_TPU_KV_PACK_HEADS=1`, `page_size` in `{64, 256}`) and verification row count `max_running_requests * speculative_num_draft_tokens <= 32` (`--max-running-requests 4` with `DFLASH-8`).
2. **Colocated Multi-Replica Serving (`SGLANG_TPU_PD_COLOCATED=1`):**
   - Plain `--dp-size > 1` inside a single SGLang-TorchTPU server process attempts to open `/dev/vfio/0` from multiple scheduler workers. Setting `SGLANG_TPU_PD_COLOCATED=1`, per-replica `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=replica_<offset>`, and `--base-gpu-id <r * tp_size>` cleanly partitions chips on a single TPU host across independent server instances behind [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py).
3. **Text-Only Configuration & `DFLASH` Draft Overrides:**
   - `Muse-Glimmer-30B` requires `--json-model-override-args '{"language_model_only": true}'` so `torch.compile` skips tracing multimodal `msgspec.Struct` schedule items; `Muse-Glimmer-30B-assistant/config.json` requires injecting `"vocab_size": 202048`.
   - `RadixArk/Muse-Spark-Open-DFlash2` (`5` SWA layers, block size `8`) uses `"is_hybrid_swa": true` and `"hybrid_layer_pattern": [1, 1, 1, 1, 1]` so draft KV slots share the SWA pool rather than allocating `262K` full-attention KV slots.
4. **Multi-Host `v5p-32` Readiness & Weight Streaming:**
   - On 8-host `v5p-32` (`nnodes=8`), only worker `0` binds the HTTP server (`GET /v1/models` returns `200` without triggering synthetic `/health` generation), and `--weight-loader-drop-cache-after-load` streams the `1.64 TB` BF16 checkpoint from GCSFuse without exhausting host page cache.

---

## Repository Layout

- [`scripts/run_muse_sweep.py`](./scripts/run_muse_sweep.py) — Unified SGLang-TorchTPU benchmark harness supporting single-host, colocated multi-replica (`replicas > 1`), multi-host TPU slice (`nnodes > 1`), Pallas Decode Megakernel, and `DFLASH` draft overrides.
- [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py) — Async `aiohttp` least-inflight HTTP reverse proxy for colocated multi-replica TPU servers.
- [`scripts/stage_spark_sharded.py`](./scripts/stage_spark_sharded.py) — Multi-pod Indexed Job HuggingFace-to-GCS parallel shard downloader with byte-exact completeness verification.
- [`scripts/collect_muse_results.py`](./scripts/collect_muse_results.py) — Aggregates per-run `summary.json` artifacts from GCS into [`results/SUMMARY.md`](./results/SUMMARY.md) and [`results/summary.json`](./results/summary.json).
- [`scripts/plan_*.json`](./scripts/) — Declarative sweep plans for `Muse-Spark-Open-816B-A42B` (`v7x` NVFP4 & `v5p-32` BF16) and `Muse-Glimmer-30B` (`v7x` & `v5p`).
- [`k8s/`](./k8s/) — Sanitized GKE Job manifests for single-host `v7x`/`v5p` sweeps, Pallas Decode Megakernel jobs, 8-host Indexed `v5p-32` TPU slice serving, and sharded GCS model staging.
- [`results/`](./results/) — Consolidated [`SUMMARY.md`](./results/SUMMARY.md) and [`summary.json`](./results/summary.json).
