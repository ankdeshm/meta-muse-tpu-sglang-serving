# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p-32`) with SGLang-TorchTPU

End-to-end serving recipes, **62-layer Pallas Decode Megakernel** configurations, **`DFLASH` block-speculative decoding** workflows, **colocated multi-replica gateways**, and **multi-host `v5p-32` expert-parallel** benchmarks for Meta's Muse model family on Cloud TPU **`v7x` (Ironwood)** and **`v5p` / `v5p-32`** using **SGLang-TorchTPU** (`sglang` `b3d718cb`, TorchTPU `0.2.0`):

- **[`meta-models/Muse-Spark-Open-816B-A42B-NVFP4`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4)** (`533.0 GB` NVFP4) & **[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)** (`1,644.4 GB` BF16; 816B total / 42B active MoE, 256 routed experts top-8 + shared expert, 62 layers = 46 SWA-2048 + 16 global GQA, 128 Q / 16 KV heads, `262K` context) + **[`RadixArk/Muse-Spark-Open-DFlash2`](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2)** (`DFLASH` block-8 draft).
- **[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)** (`59.6 GB` BF16; dense 30B, 52 layers = 39 SWA-2048 + 13 global NoPE, 32 Q / 2 KV heads, `131K` context) + **[`meta-models/Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant)** (`DFLASH` block-16 draft).

Full per-cell benchmark tables (32 serving variants): [`results/SUMMARY.md`](./results/SUMMARY.md) · machine-readable records: [`results/summary.json`](./results/summary.json) · raw benchmark output for every measured cell: [`results/raw/`](./results/raw/) (index: [`results/raw/index.csv`](./results/raw/index.csv)).

**Conventions:** `c` = number of concurrent requests; TTFT = time to first token and TPOT = time per output token (both p50); "accept len" = mean tokens accepted per speculative verify step; "Chips" = physical TPU chips used (one `v7x` chip = 2 TensorCores). Every request decodes exactly the stated output length (EOS ignored).

## Methodology notes

- Speculative-decoding (`DFLASH`) rows use greedy decoding (temperature 0) on synthetic fixed-length prompts, which favours draft acceptance. One sampled data point (T=1.0, top-p 0.95) is included: Glimmer DFLASH-16 on `v5p` accepted ~2.6–3.7 tokens per step and reached 88.4 tok/s at `c=1` (119.1 tok/s without speculation).
- Accept length is the server's running average since start-up; per-cell estimates (from differencing) are shown beside it.
- Long-context cells (prompts of 32K tokens or more) use 2–4 requests at `c=1` (2–16 in total), so treat them as indicative.
- No accuracy benchmark was run; each server was spot-checked with one greedy chat prompt ("capital of France?").
- Multi-replica gateway peaks are measured at `c=128` and single servers up to `c=64`; equal-concurrency (`c=64`) values are shown alongside.
- The `v5p-32` Spark runs load the BF16 checkpoint with the MoE experts quantized to FP8 at load; attention, embeddings and the KV cache are BF16. On `v7x` the NVFP4 checkpoint is served as stored.
- The megakernel comparison baseline is `DFLASH-8` with an FP8 KV cache (`spark_nvfp4_tp8_ep1_dflash8`).

---

## 1. Headline Results — `Muse-Spark-Open-816B-A42B` (`1024` in / `1024` out, EOS ignored)

All `Muse-Spark-Open-816B-A42B` speculative runs use **real `RadixArk/Muse-Spark-Open-DFlash2` draft generation & target verification**, with greedy decoding on synthetic prompts (see Methodology notes). The running-average acceptance was **`6.25 – 7.17` tokens per step**, and per-cell estimates reach the 8-token maximum at long context. The `v5p-32` rows load the BF16 checkpoint with its **MoE experts quantized to FP8 at load** (finding 4).

| Hardware | Variant / Topology | Chips | Conc. 1: Out tok/s (TPOT p50) | Conc. 4 | Conc. 8 | Conc. 16 | Conc. 32 / 48 / 64 Peak | Spec Accept Len (running avg) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TPU `v7x-8` (2 hosts)** | `spark_nvfp4_v7x8_tp16_ep1_dflash8` (**TP16 EP1 16-Way `pallas_tpu_fused_rs` Expert-TP + BF16 DFLASH-8**, FP8 KV across `2x2x2` ICI, `1.536 TB` HBM) | 8 | **365.8** (`2.17 ms`) | **674.2** (`4.45 ms`) | **1,061.2** (`6.07 ms`, TTFT **`234 ms`**) | **1,486.2** (`8.84 ms`, TTFT `395 ms`) | **2,563.8** (c=64, **`5,127.5` tot**, `320.5`/chip; c=32: **2,125.1**; `T=0.6` c=32: **1,777.6**) | **6.44 – 7.14** |
| **TPU `v7x-8` (2 hosts)** | `spark_nvfp4_2xtp8_megakernel_bf16_gateway_8chip` (**2×TP8 62-Layer Pallas Decode Megakernel + FP8 DFLASH-8 Distributed Gateway**, BF16 packed KV, `64` verify rows across 8 chips) | 8 | **744.5** (`0.96 ms`; c=2: **1,271.9** at `0.95 ms`) | **1,645.9** (`1.86 ms`) | **1,971.2** (`3.19 ms`, TTFT **`302 ms`**) | **2,031.9** (`3.27 ms`) | **2,047.1** (c=32, `255.9`/chip; `T=0.6` c=8: **1,696.8**) | **6.19 – 7.03** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_dflash8_megakernel` (**62-Layer Pallas Decode Megakernel + DFLASH-8**, BF16 KV, max 4 running requests) | 4 | **761.9** (`0.97 ms`) | **955.8** (`3.29 ms`) | 966.5 (`3.28 ms`; queued, TTFT `3.9 s`) | **998.4** (`3.31 ms`; queued, TTFT `12.0 s`) | — (`mrr=4`) | **6.25 – 6.74** (per cell ≈6.3 – 7.0) |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_dflash8_etp_opt` (NVFP4 TP8 **Fused-RS ETP Opt + FP8 DFlash2 + BSearch**, FP8 KV, greedy) | 4 | **342.2** (`2.31 ms`) | **675.5** (`4.53 ms`) | **989.9** (`6.09 ms`) | **1,466.3** (`8.95 ms`) | **2,207.0** (c=48, `551.8`/chip; c=32: **2,079.3**) | **6.35 – 6.95** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_dflash8_sampled_t06` (NVFP4 TP8 Fused-RS ETP Opt + **DFLASH-8**, **sampled `T=0.6`**, top-p `0.95`) | 4 | **355.6** (`2.35 ms`) | 606.5 (`4.68 ms`) | — | 1,269.0 (`9.41 ms`) | **1,813.3** (c=32, `453.3`/chip) | **6.10 – 6.90** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_dflash8_sampled_t10` (NVFP4 TP8 Fused-RS ETP Opt + **DFLASH-8**, **sampled `T=1.0`**, top-p `0.95`) | 4 | **249.4** (`3.25 ms`) | 425.3 (`8.56 ms`) | — | 992.8 (`12.00 ms`) | **1,443.5** (c=32, `360.9`/chip) | **4.21 – 4.86** |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_dflash8` (NVFP4 TP8 EP1 + **DFLASH-8**, FP8 KV baseline) | 4 | **355.2** (`2.35 ms`) | 664.1 (`4.43 ms`) | 974.5 (`6.18 ms`) | 1,360.2 (`9.12 ms`) | 1,861.9 (c=32, `465.5`/chip) | 6.71 – 6.99 |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1` (NVFP4 TP8 **EP1 Expert-TP** `pallas_tpu_fused_rs`, BF16 KV) | 4 | **91.5** (`10.78 ms`) | 261.1 (`14.78 ms`) | 373.6 (`20.64 ms`) | 522.3 (`29.35 ms`) | **1,036.2** (c=64, `259.0`/chip) | — |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep1_bs8_recipe` (NVFP4 TP8 EP1, RadixCache on, `mrr=8`) | 4 | 91.4 (`10.79 ms`) | 263.8 (`14.81 ms`) | 381.0 (`20.52 ms`) | 381.1 (`20.54 ms`) | 384.6 (c=32) | — |
| **TPU `v7x` (1 host)** | `spark_nvfp4_tp8_ep8` (NVFP4 TP8 **EP8** 32 whole experts/TC, FP8 KV) | 4 | 57.7 (`17.14 ms`) | 141.2 (`27.68 ms`) | — | 286.4 (`54.40 ms`) | 421.3 (c=32) | — |
| **TPU `v5p-32` (8 hosts)** | `spark_bf16_v5p32_tp32_ep32_dflash8` (BF16 checkpoint, MoE experts quantized to FP8 at load; **TP32 EP32 + DFLASH-8**) | 32 | **148.0** (`5.97 ms`) | **382.2** (`7.74 ms`) | **685.9** (`8.68 ms`) | **1,119.8** (`11.44 ms`) | **1,540.1** (c=32) | **6.64 – 7.02** (per cell ≈6.5 – 7.2) |
| **TPU `v5p-32` (8 hosts)** | `spark_bf16_v5p32_tp32_ep32` (BF16 checkpoint, MoE experts quantized to FP8 at load; **TP32 EP32**, 8 experts/chip, non-speculative) | 32 | 40.0 (`24.81 ms`)† | 136.4 (`28.89 ms`)† | 242.4 (`32.37 ms`)† | 414.3 (`37.57 ms`) | **1,121.5** (c=64) | — |

† Measured in a separate sweep on the same server.

### Key `Muse-Spark-Open-816B-A42B` Findings

1. **Sub-Millisecond TPOT (`0.97 ms`, `761.9` out tok/s at `c=1`) on an 816B Parameter MoE via the 62-Layer Pallas Decode Megakernel + `DFLASH-8`:**
   - Enabling `SGLANG_TPU_MUSE_MEGAKERNEL=1` with `SGLANG_TPU_KV_PACK_HEADS=1` (`bfloat16` packed KV, `pallas_tpu_rpa_v3`, `max_running_requests=4` = `32` speculative verification rows) fuses all 62 decoder layers of `Muse-Spark-Open-816B-A42B-NVFP4` on a single 4-chip `v7x` host (`768 GB` HBM).
   - At `1024/1024 c=1`, single-stream output throughput reaches **`761.9` output tok/s (`190.5` tok/s/chip, `154 ms` TTFT p50, `0.97 ms` TPOT p50)** — **`8.33x` faster** than non-speculative `TP8 EP1` (`91.5` tok/s, `10.78 ms` TPOT) and **`2.15x` faster** than standard `DFLASH-8` (`355.2` tok/s, `2.35 ms` TPOT), reaching **`955.8`–`998.4` output tok/s** at `c=4..16`.
   - **2-Host 8-Chip Linear Megakernel Scaling (`2×TP8` Distributed Gateway, `2,047.1` out tok/s & `16,847` Long-Context Total tok/s):** Deploying two `TP8` 62-layer Megakernel + FP8 `DFLASH-8` servers across 2 `v7x` hosts (`8` chips / `16` TensorCores = `64` concurrent speculative verification rows) doubles unqueued concurrency from `c=4` to `c=8`, cutting `c=8` median TTFT **`12.9x`** (`3,884 ms` → **`302 ms`**) while reaching **`1,971.2` out tok/s at `c=8`**, **`2,047.1` out tok/s (`4,094.3` total tok/s, `2.05x` 1-host peak) at `c=32`**, **`1,696.8` out tok/s under non-greedy `T=0.6` sampling** (`1.00 ms` TPOT at `c=1`), **`510.5` out tok/s (`16,846.8` total tok/s)** at `32K` context (`c=8`), and **`246.9` out tok/s (`16,049.9` total tok/s, `1.04 ms` TPOT)** at `64K` context (`c=2`).
   - **2-Host 8-Chip Single-Slice `TP16 EP1` + `DFLASH-8` Peak Batch Frontier (`2,563.8` out tok/s / `5,127.5` total tok/s at `c=64`):** Sharding all 256 routed experts 16-way across a 2-host `2x2x2` `v7x-8` ICI slice (`16` TensorCores, `1.536 TB` HBM, FP8 KV, BF16 8-KV-head `DFlash2` draft replicated across `TP16`) eliminates the `mrr=4` megakernel verification-row ceiling, reaching **`365.8` out tok/s (`2.17 ms` TPOT) at `c=1`**, **`2,125.1` out tok/s (`512 ms` TTFT) at `c=32`**, and **`2,563.8` out tok/s (`5,127.5` total tok/s, `18.71 ms` TPOT, `1,095 ms` TTFT) at `c=64`** — **`+25.2%` higher peak throughput** than `2×TP8` (`2,047.1` tok/s) and **`+66.5%` higher** than 32-chip `v5p-32` `TP32 EP32 + DFLASH-8` (`1,540.1` tok/s) using `4x` fewer chips, plus **`1,777.6` out tok/s at `T=0.6`** (`c=32`) and **`315 ms` TTFT (`11.5x` speedup, `1,258.4` out tok/s)** on `80%` shared-prefix `4K` workloads (`c=32`). *(Note: streaming `TP16` weights directly from GCS FUSE across 8 workers/host requires `--weight-loader-disable-mmap` without `--weight-loader-drop-cache-after-load` to avoid concurrent `posix_fadvise(POSIX_FADV_DONTNEED)` page-invalidation stalls.)*
   - Notes: greedy decoding on synthetic prompts; the `c=1` cell uses 8 requests. The megakernel needs a BF16 KV cache and at most 4 running requests, so above `c=4` requests queue (TTFT `3.9 s` at `c=8`, `12.0 s` at `c=16`) and throughput plateaus at ~1,000 tok/s. The `2.15x` baseline is `DFLASH-8` with an FP8 KV cache. At 64K input the per-stream TPOT stays at ~`1 ms`, and end-to-end throughput is `98.0` tok/s (TTFT `9.4 s`, 3 requests).
2. **High-Concurrency Batch Frontier (`2,207.0` out tok/s = `551.8` out tok/s/chip) & Sampled Speculative Decoding (`T=0.6` / `T=1.0`) with Fused-RS ETP + FP8 `DFlash2`:**
   - Combining Fused Reduce-Scatter Expert-TP optimizations (`SGLANG_TPU_FUSED_RS_ETP_LOCAL_ROWS=1`, `TILED_REDUCE=1`, `FUSED_COMBINE=1`), FP8 weight-only `DFlash2` draft linears (`SGLANG_TPU_DFLASH_FP8_DRAFT=1`, `ROW_PARALLEL_FC=1`, `CONV_MATMUL_EXPAND=1`), and binary-search TPU sampling (`SGLANG_TPU_ENABLE_BSEARCH_SAMPLING=1`) pushes single-host `v7x` output throughput to **`2,079.3` out tok/s at `c=32`** (`11.93 ms` TPOT) and **`2,207.0` out tok/s (`4,414.1` total tok/s, `551.8` out tok/s/chip, `11.08 ms` TPOT) at `c=48`** — **`2.13x` higher peak throughput** than non-speculative `c=64` (`1,036.2` tok/s) at **`5.2x` lower TPOT**.
   - Under **non-greedy stochastic sampling** (`top_p=0.95`), real `DFLASH-8` sustains **`6.10–6.90` accepted tokens/step at `T=0.6`** (**`355.6` out tok/s at `c=1` → `1,813.3` out tok/s at `c=32`**) and **`4.21–4.86` accepted tokens/step even at `T=1.0`** (**`249.4` out tok/s at `c=1` → `1,443.5` out tok/s at `c=32`**, still **`2.73x` faster at `c=1`** and **`1.94x` faster at `c=32`** than non-speculative `TP8 EP1`).
   - With RadixCache enabled on `80%` shared-prefix `4096/1024` workloads (`spark_nvfp4_tp8_dflash8_prefix80`), median TTFT drops **`2.9x–7.5x`**: from `540.2 ms` → **`186.6 ms`** (`c=1`), `1,071.9 ms` → **`234.1 ms`** (`c=8`, `618.8` out tok/s), and `2,006.5 ms` → **`266.9 ms`** (`c=16`, `814.5` out tok/s).
3. **Single-Host `EP=1` (`pallas_tpu_fused_rs`) Beats `EP=8` by `1.59x`–`1.85x` on `v7x`:**
   - On 1-host `v7x` (`8` TensorCores connected over high-bandwidth ICI), sharding all 256 experts tensor-parallel across the 8 ranks with `pallas_tpu_fused_rs` (`EP=1`) achieves **`91.5` → `742.7` out tok/s** (`c=1..32`), outperforming `EP=8` (`57.7` → `421.3` out tok/s) by **`1.59x`–`1.85x`** by eliminating token-routing load imbalance and separate all-to-all dispatch/combine collectives. (The EP8 run used an FP8 KV cache, the EP1 run a BF16 KV cache.)
4. **`1.64 TB` BF16 Checkpoint Across 8-Host TPU `v5p-32` (`TP32 EP32` + `DFLASH-8`), MoE Experts Quantized to FP8 at Load:**
   - Across 8 `v5p` hosts (`32` chips in `2x4x4` 3D torus, `3.04 TB` HBM), `TP32 EP32` places 8 whole experts per chip and serves out to **`258,048` (`~262K`) context**. The BF16 checkpoint is loaded with its MoE experts quantized to FP8 at load (TPU Pallas FP8 fused-RS MoE backend; `25.1 GB` of weights per chip vs ~`51 GB` for BF16), while attention, embeddings and the KV cache are BF16. Adding `DFLASH-8` (greedy) boosts `c=1` throughput **`3.70x`** (`40.0` → **`148.0` out tok/s**, `24.81 ms` → **`5.97 ms` TPOT**) and lifts peak batch throughput from `1,121.5` (`c=64`) to **`1,540.1` out tok/s (`c=32`, `14.61 ms` TPOT)**.
   - Loading the checkpoint over GCSFuse took 35–75 min per host, and the servers were ready after 1.5–1.6 h.

### `Muse-Spark-Open-816B-A42B` Long-Context Scaling (`4K` → `258K` Input Tokens, `1024` Output Tokens)

| Hardware & Variant | `4096` in (`c=1` / Peak) | `32768` in (`c=1` / `c=4`) | `65536` (`64K`) in (`c=1` / Peak) | `131072` (`128K`) in (`c=1`) | `258048` (`~262K`) in (`c=1`) |
|---|---|---|---|---|---|
| **`v7x-8` `TP16 EP1` + `DFLASH-8` (8 chips, FP8 KV)** | **`364.3`** (`2.18 ms`) / **`1,266.5`** (c=32, **`6,332` tot**; `prefix80` c=32: **`1,258.4`** at **`315 ms` TTFT**) | — | **`122.0`** (`2.22 ms`, TTFT `8.66 s`) / **`151.2`** (c=4, **`9,829` tot tok/s**, acc `7.13`) | `133.0`* (`2.30 ms`; c=2: `474.7`*) | `72.2`* (`2.41 ms`, acc `7.14`) |
| **`v7x-8` `2×TP8` Megakernel + `DFLASH-8` Gateway (8 chips)** | **`719.3`** (`0.95 ms`) / **`1,783.5`** (c=16, **`8,917` tot**; c=8: **`1,727.0`** at `579 ms` TTFT) | **`205.7`** (`0.99 ms`, TTFT `3.97 s`) / **`510.5`** (c=8, **`16,847` tot**) | **`166.3`** (`1.04 ms`, TTFT `5.10 s`) / **`246.9`** (c=2, **`16,050` tot tok/s**, acc `7.18`) | — | — |
| **`v7x` Megakernel + `DFLASH-8`** | **`672.0`** (`0.96 ms`) / **`796.5`** (c=4) | **`188.9`** (`0.98 ms`, TTFT `4.44 s`) / `188.7` | **`98.0`** (`1.02 ms` TPOT, TTFT `9.44 s`, `6,373` tot tok/s, acc `7.05` running avg, ≈8 per cell = the maximum) | — | — |
| **`v7x` `TP8` Fused-RS ETP Opt + `DFLASH-8` (`etp_opt`)** | **`348.4`** (`2.33 ms`) / **`1,355.0`** (c=32, **`6,775` tot**) | — | — | — | **`37.1` out tok/s** (**`2.67 ms` TPOT**, TTFT `24.9 s`, **`9,384` tot tok/s**, acc `7.12`) |
| **`v7x` `TP8 EP1` + `DFLASH-8` (FP8 KV)** | **`349.3`** (`2.33 ms`) / **`937.1`** (c=16, `4,685` tot) | **`151.6`** (`2.38 ms`) / **`189.1`** (`6,239` tot) | **`87.7`** (`2.41 ms`) / **`94.8`** (c=2, `6,165` tot) | **`44.4` out tok/s** (`2.49 ms` TPOT, TTFT `20.5 s`, acc `7.17` running avg, ≈7.9 per cell) | — |
| **`v7x` `TP8 EP1` (BF16 KV, radix cache off)** | `87.8` (`10.87 ms`) / **`698.8`** (c=64) | `64.9` (`11.17 ms`) / `121.4` (`4,005` tot) | `48.8` (TTFT `9.3 s`) / `75.3` (c=4) | `31.0` (TTFT `20.6 s`) / `36.4` (c=2) | **`16.2` out tok/s** (`13.47 ms` TPOT, TTFT `49.3 s`)‡ |
| **`v5p-32` `TP32 EP32` + `DFLASH-8`** (FP8 experts) | **`149.6`** (`5.84 ms`) / **`754.5`** (c=16, `3,773` tot) | — | **`60.9`** (`5.91 ms`, TTFT `10.9 s`, acc `7.15` running avg, ≈8 per cell = the maximum) / **`72.0`** (c=2, `4,679` tot) | — | — |
| **`v5p-32` `TP32 EP32`** (FP8 experts) | `39.4` (`24.87 ms`) / **`785.3`** (c=64, `3,927` tot) | `34.8` (`25.00 ms`, TTFT `3.84 s`) / `90.5` | `30.5` (`25.15 ms`, TTFT `7.83 s`) / `66.4` (`4,317` tot) | `24.2` (`25.50 ms`) / **`33.8`** (c=2, `4,359` tot) | **`16.6` out tok/s** (`26.08 ms` TPOT, TTFT `34.9 s`, `4,207` tot) |

‡ Long-context rows are measured with the radix (prefix) cache off (or cold-prompt first pass for `etp_opt` `258K`; `*` marks `TP16` `128K`/`258K` cells run with RadixCache enabled). Cells with prompts of 32K tokens or more use 2–4 requests at `c=1`.

---

## 2. Headline Results — `Muse-Glimmer-30B` (`1024` in / `1024` out, EOS ignored)

| Hardware | Variant / Topology | Chips | Conc. 1: Out tok/s (TPOT p50) | Conc. 4 | Conc. 16 | Conc. 32 | Conc. 64 / 128 / 256 Peak | Peak Out tok/s/chip |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **TPU `v7x-8` (2 hosts)** | `glimmer_fp8_8xtp2_gateway_8chip` (**8×FP8-TP2 Distributed Gateway**, FP8 KV) | 8 | 126.1 (`7.79 ms`) | 507.3 (`7.80 ms`) | 2,011.0 (`7.60 ms`) | 3,921.1 (`7.83 ms`) | **22,921.9** (c=256, `10.43 ms`); c=128: **13,597.9**; c=64: 7,221.5 | **2,865.2** |
| **TPU `v7x` (1 host)** | `glimmer_fp8_4xtp2_gateway` (**4×FP8-TP2 Colocated Gateway**, FP8 KV) | 4 | 126.7 (`7.79 ms`) | 505.7 (`7.82 ms`) | 1,931.9 (`7.89 ms`) | 3,706.6 (`8.29 ms`) | **11,502.2** (c=128, `10.50 ms`); c=64: 6,849.3 | **2,875.5** |
| **TPU `v7x` (1 chip)** | `glimmer_fp8_tp2` (FP8 W8A8 **TP2 = 1 chip**, FP8 KV) | 1 | 127.0 (`7.80 ms`) | 495.1 (`7.79 ms`) | 1,750.9 (`8.75 ms`) | 2,899.7 (`10.51 ms`) | **4,254.7** (c=64, `13.97 ms`) | **4,254.7** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_4xtp2_gateway` (BF16 **4×TP2 Colocated Gateway**) | 4 | 82.8 (`11.96 ms`) | 331.1 (`11.96 ms`) | 1,255.6 (`12.34 ms`) | 2,417.0 (`12.67 ms`) | **7,563.5** (c=128, `15.93 ms`); c=64: 4,484.2 | **1,890.9** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_2xtp4_gateway` (BF16 **2×TP4 Colocated Gateway**) | 4 | 133.5 (`7.38 ms`) | 531.4 (`7.36 ms`) | 1,805.0 (`8.33 ms`) | 3,163.4 (`9.41 ms`) | **7,482.5** (c=128, `15.44 ms`); c=64: 5,125.0 | **1,870.6** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8_dflash8` (BF16 TP8 + **DFLASH-8**, `32×8=256` verify tokens, `4.83–5.16` running-avg acc) | 4 | **619.6** (`1.37 ms`) | **2,055.5** (`1.44 ms`) | **4,452.2** (`2.15 ms`) | **5,932.9** (`3.31 ms`) | — | **1,483.2** |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8_dflash16` (BF16 TP8 + **DFLASH-16**, greedy, `5.44–6.30` running-avg acc, ≈5.4–6.4 per cell) | 4 | **677.6** (`1.21 ms`) | **2,207.3** (`1.40 ms`) | **4,482.4** (`2.16 ms`) | —¹ | — | 1,120.6 |
| **TPU `v7x` (1 host)** | `glimmer_bf16_tp8` (BF16 TP8, non-speculative) | 4 | **185.8** (`5.30 ms`) | 697.0 (`5.46 ms`) | 2,207.8 (`6.72 ms`) | 3,359.2 (`8.76 ms`) | **4,795.4** (c=64, `12.12 ms`) | 1,198.8 |
| **TPU `v7x` (1 chip)** | `glimmer_bf16_tp2` (BF16 **TP2 = 1 chip**) | 1 | 82.9 (`11.96 ms`) | 315.2 (`12.32 ms`) | 1,123.7 (`13.77 ms`) | 1,900.5 (`15.92 ms`) | **3,087.8** (c=64, `18.87 ms`) | **3,087.8** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_2xtp2_gateway` (BF16 **2×TP2 Colocated Gateway**) | 4 | 66.6 (`14.83 ms`) | 288.9 (`13.60 ms`) | 1,014.7 (`14.44 ms`) | 1,852.3 (`16.13 ms`) | **4,691.7** (c=128, `24.18 ms`); c=64: 3,083.9 | **1,172.9** |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash8` (BF16 TP4 + **DFLASH-8**, greedy, `4.70–5.12` running-avg acc, ≈4.7–5.4 per cell) | 4 | **460.2** (`1.78 ms`) | 1,402.7 (`2.18 ms`) | **2,847.3** (`3.57 ms`) | **3,472.7** (`6.21 ms`) | — | 868.2 |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash16` (BF16 TP4 + **DFLASH-16**, greedy, `5.64–6.12` running-avg acc, ≈5.4–6.4 per cell) | 4 | **516.5** (`1.52 ms`) | **1,540.8** (`2.15 ms`) | 2,511.3 (`3.51 ms`) | 2,685.7 (`7.50 ms`) | — | 671.4 |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4_dflash16_t1` (BF16 TP4 + **DFLASH-16**, **sampled** T=1.0 / top-p 0.95, `2.62–3.40` running-avg acc, ≈2.6–3.7 per cell) | 4 | 88.4 (`11.22 ms`) | 356.4 (`10.24 ms`) | 840.3 (`15.68 ms`) | 1,072.3 (`25.46 ms`) | — | 268.1 |
| **TPU `v5p` (1 host)** | `glimmer_bf16_tp4` (BF16 TP4, non-speculative) | 4 | 119.1 (`8.27 ms`) | 439.1 (`8.55 ms`) | 1,452.8 (`10.43 ms`) | 2,302.4 (`12.92 ms`) | **3,322.0** (c=64, `17.09 ms`) | 830.5 |

¹ DFLASH-16 at 1K / `c=32` (`32 × 16 = 512` verify tokens) did not produce valid output; capping verification to block-8 (`glimmer_bf16_tp8_dflash8`, `32 × 8 = 256` verify tokens) resolves `c=32` cleanly at **`5,932.9` out tok/s (`3.31 ms` TPOT)**.

### `Muse-Glimmer-30B` Long-Context Scaling (`4K` → `127K` Input Tokens, `1024` Output Tokens)

| Hardware & Variant | `4096` in (`c=1` / Peak) | `32768` in (`c=1` / Peak) | `65536` (`64K`) in (`c=1` / Peak) | `126976` (`~128K`) in (`c=1` / Peak) |
|---|---|---|---|---|
| **`v7x-8` `glimmer_fp8_8xtp2_gateway_8chip` (8 chips)** | `123.9` (`7.81 ms`) / **`11,308.0`** (c=128, **`56,529` tot tok/s**) | — | `90.5` (`8.03 ms`, TTFT `3.09 s`) / **`721.0`** (c=8, **`46,862` tot**) | `63.5` (`8.26 ms`) / **`503.3`** (c=8, **`62,915` tot tok/s**) |
| **`v7x` `glimmer_fp8_4xtp2_gateway` (4 chips)** | `123.9` (`7.82 ms`) / **`5,593.0`** (c=64, `27,959` tot tok/s) | — | `90.4` (`8.02 ms`, TTFT `3.11 s`) / **`353.8`** (c=4, `22,996` tot) | `63.1` (`8.26 ms`) / **`250.8`** (c=4, **`31,349` tot tok/s**) |
| **`v7x` `glimmer_bf16_tp8_dflash8` (4 chips)** | **`559.2`** (`1.19 ms`, acc `4.83`) / **`3,964.4`** (c=32, **`19,818` tot tok/s**, `5.68 ms` TPOT) | — | — | — |
| **`v7x` `glimmer_bf16_tp8_dflash16` (+ `longctx`)** | **`618.4`** (`1.13 ms`) / **`2,564.5`** (c=16, `12,820` tot) | **`469.6`** (`0.73 ms` TPOT, TTFT `1.09 s`, acc `7.04`) / **`573.0`** (c=4, `18,910` tot) | **`303.1`** (`0.74 ms` TPOT, TTFT `2.29 s`, acc `5.38` running avg, ≈`6.8` per cell, `19,703` tot) | **`163.5` out tok/s** (`1.20 ms` TPOT, TTFT `5.02 s`, acc `5.52` running avg, ≈`7.0` per cell, **`20,441` tot tok/s**) |
| **`v7x` `glimmer_bf16_2xtp4_gateway`** | `128.2` (`7.41 ms`) / **`3,466.6`** (c=64, `17,330` tot) | — | `88.8` (`7.63 ms`) / **`267.5`** (c=4, `17,389` tot) | `63.1` / **`126.0`** (c=2, `15,744` tot) |
| **`v7x` `glimmer_bf16_tp8`** | `179.6` (`5.31 ms`) / **`2,907.3`** (c=64, `14,534` tot) | `154.1` (`5.40 ms`, TTFT `1.11 s`) / **`526.1`** (c=8, `17,359` tot) | `128.7` (`5.50 ms`, TTFT `2.31 s`) / **`269.5`** (c=4, `17,518` tot) | **`94.0` out tok/s** (`5.75 ms` TPOT, TTFT `5.01 s`, `11,753` tot) |
| **`v5p` `glimmer_bf16_tp4_dflash16`** | **`473.9`** (`1.40 ms`, acc `5.70` running avg, ≈`6.1` per cell) / **`1,838.8`** (c=32) | **`296.8`** (`0.87 ms` TPOT, TTFT `2.04 s`, acc `5.63` running avg, ≈`6.5` per cell) / **`373.7`** (c=4, `12,331` tot) | — | — |
| **`v5p` `glimmer_bf16_2xtp2_gateway`** | `64.2` (`14.86 ms`) / **`2,043.9`** (c=64, `10,218` tot) | `54.7` / **`290.4`** (c=8, `9,583` tot) | `44.8` / **`140.3`** (c=4, `9,117` tot) | `31.4` / **`62.7`** (c=2, `7,836` tot) |
| **`v5p` `glimmer_bf16_tp4`** | `114.4` (`8.30 ms`) / **`1,964.6`** (c=64, `9,821` tot) | `96.3` (`8.42 ms`, TTFT `2.00 s`) / **`315.0`** (c=8, `10,393` tot) | `78.4` (`8.55 ms`, TTFT `4.30 s`) / **`153.3`** (`9,964` tot) | **`54.9` out tok/s** (`8.77 ms` TPOT, TTFT `9.68 s`, `6,865` tot) |

### Key `Muse-Glimmer-30B` Findings

1. **`22,921.9` Output tok/s Across 2-Host `v7x-8` (`8×FP8-TP2`, `99.6%` Linear Multi-Host Scaling) & `11,502.2` Output tok/s on 1-Host `v7x` (`4×FP8-TP2`):**
   - Because `Muse-Glimmer-30B` has `num_key_value_heads=2`, single-chip `TP2` (`2` TensorCores = `1` `v7x` chip, `192 GB` HBM) eliminates all inter-chip ICI collectives AND unlocks native `v7x` FP8 W8A8 MXU execution (`--quantization fp8 --kv-cache-dtype fp8_e4m3`), delivering **`4,254.7` output tok/s on a single `v7x` chip**, **`11,502.2` output tok/s (`2,875.5` tok/s/chip)** across 4 chips (`c=128`), and **`22,921.9` output tok/s (`45,821.5` total tok/s, `2,865.2` tok/s/chip, `10.43 ms` TPOT)** across 8 chips (`8×FP8-TP2` across 2 `v7x` hosts at `c=256`, **`99.64%` multi-host scaling efficiency**). At long context across 8 chips, `8×FP8-TP2` sustains **`11,308.0` out tok/s (`56,529.1` total tok/s)** at `4K` (`c=128`), **`721.0` out tok/s (`46,861.6` total tok/s)** at `64K` (`c=8`), and **`503.3` out tok/s (`62,915.4` total tok/s)** at `127K` (`c=8`). At equal concurrency (`c=64`) the 4-chip FP8 gateway delivers `6,849.3` tok/s (**`1.43x`** BF16 TP8 `4,795.4` tok/s) and the 8-chip FP8 gateway delivers `7,221.5` tok/s (`8.21 ms` TPOT). FP8 output quality was not evaluated beyond one sanity prompt.
   - Other gateways at equal concurrency (`c=64`): `v7x` BF16 2×TP4 `5,125.0` vs TP8 `4,795.4` (**`1.07x`**); `v7x` BF16 4×TP2 `4,484.2` (**`0.94x`**); `v5p` BF16 2×TP2 `3,083.9` vs TP4 `3,322.0` (**`0.93x`**, i.e. slower). The `c=128`/`c=256` gateway peaks reflect scaling concurrency linearly with replica count (`32` concurrent streams per single-chip replica).
2. **`5,932.9` Output tok/s (`3.31 ms` TPOT at `c=32`) in Unquantized BF16 via `DFLASH-8` (`glimmer_bf16_tp8_dflash8`) & Sub-Millisecond Long-Context Decode (`0.73–0.74 ms` TPOT at `32K`–`64K`) via `DFLASH-16`:**
   - Capping `Muse-Glimmer-30B-assistant` verification to block-8 (`--speculative-num-draft-tokens 8`, `32 × 8 = 256` verify tokens) resolves the `DFLASH-16` `c=32` verification-token limit (`512` tokens) and pushes single-server BF16 `TP8` on 4 `v7x` chips to **`5,932.9` output tok/s (`1,483.2` out tok/s/chip, `3.31 ms` TPOT, `4.83` accept len)** at `1024/1024 c=32` (**`1.77x` faster** than non-speculative BF16 `TP8` at `c=32` `3,359.2` tok/s and **`1.24x` faster** than BF16 `TP8` at `c=64` `4,795.4` tok/s at **`3.66x` lower TPOT**) and **`3,964.4` output tok/s (`19,817.9` total tok/s, `5.68 ms` TPOT)** at `4096/1024 c=32` (**`1.71x`** non-speculative `c=32`).
   - Setting `--swa-full-tokens-ratio 1.0` eliminates hybrid SWA KV eviction stalls under long prompts, allowing `glimmer_bf16_tp8_dflash16_longctx` on `v7x` to sustain **`469.6` out tok/s (`0.73 ms` TPOT)** at `32K`, **`303.1` out tok/s (`0.74 ms` TPOT)** at `64K`, and **`163.5` out tok/s (`1.20 ms` TPOT, `20,441` total tok/s)** at `127K` context (greedy; 2–4 requests per cell).

---

## 3. Critical TPU Serving Recipes & Architectural Notes

1. **62-Layer `Muse-Spark` Pallas Decode Megakernel (`SGLANG_TPU_MUSE_MEGAKERNEL=1`):**
   - Requires packed BF16 KV layout (`--kv-cache-dtype bfloat16 --attention-backend pallas_tpu_rpa_v3` + `SGLANG_TPU_KV_PACK_HEADS=1`; the measured run used `--page-size 128`) and verification row count `max_running_requests * speculative_num_draft_tokens <= 32` (`--max-running-requests 4` with `DFLASH-8`). The megakernel does not support an FP8 KV cache.
2. **Colocated Multi-Replica Serving (`SGLANG_TPU_PD_COLOCATED=1`):**
   - Plain `--dp-size > 1` inside a single SGLang-TorchTPU server process attempts to open `/dev/vfio/0` from multiple scheduler workers. Setting `SGLANG_TPU_PD_COLOCATED=1`, per-replica `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=replica_<offset>`, and `--base-gpu-id <r * tp_size>` cleanly partitions chips on a single TPU host across independent server instances behind [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py).
3. **Text-Only Configuration & `DFLASH` Draft Overrides:**
   - `Muse-Glimmer-30B` requires `--json-model-override-args '{"language_model_only": true}'` so `torch.compile` skips tracing multimodal `msgspec.Struct` schedule items; `Muse-Glimmer-30B-assistant/config.json` requires injecting `"vocab_size": 202048`.
   - `RadixArk/Muse-Spark-Open-DFlash2` (`5` SWA layers, block size `8`) uses `"is_hybrid_swa": true` and `"hybrid_layer_pattern": [1, 1, 1, 1, 1]` so draft KV slots share the SWA pool rather than allocating `262K` full-attention KV slots.
4. **Multi-Host `v5p-32` Readiness & Weight Streaming:**
   - On 8-host `v5p-32` (`nnodes=8`), only worker `0` binds the HTTP server (`GET /v1/models` returns `200` without triggering synthetic `/health` generation), and `--weight-loader-drop-cache-after-load` streams the `1.64 TB` BF16 checkpoint from GCSFuse without exhausting host page cache. The MoE experts are quantized to FP8 at load (TPU Pallas FP8 fused-RS MoE backend, `25.1 GB` per chip). Multi-host start-up uses a TCP start barrier so that all 8 hosts launch the server together, plus `--no-tpu-node-normalization` (the TorchTPU SliceBuilder already orders the 32 workers).

### Supported configurations & known limitations

- `Muse-Glimmer-30B` is served text-only (`--json-model-override-args '{"language_model_only": true}'`).
- Several Glimmer replicas on one host run as colocated servers behind the gateway; `--dp-size` replicas inside one server and DP-attention are not supported for Glimmer in this build.
- FP8 W8A8 needs KV heads ≥ TP (Glimmer has 2 KV heads), so FP8 runs use TP2.
- The Spark decode megakernel needs a BF16 KV cache and at most 4 running requests, so requests queue above `c=4`.
- Spark MoE supports EP or expert-TP, not both (`ep_size` is 1 or equal to `tp_size`).
- On `v5p-32`, Spark loads the BF16 checkpoint with the MoE experts quantized to FP8 at load. TP32 with expert-TP only (EP1) was not measured, because start-up exceeds 2 hours.
- The sweep harness (`scripts/run_muse_sweep.py`) adapts the Spark Pallas kernels to the JAX 0.11 Pallas API (`pl.semaphore_*`) at start-up, and the Spark plans load weights over GCSFuse with `SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE=1`.
- `glimmer_bf16_tp8_dflash16` at 1K / `c=32` did not produce valid output in this configuration and is not reported.
- With the radix cache on, long-context cells reuse the warm-up prompt, so the long-context tables show cache-off values.

Per-variant details: [`results/SUMMARY.md`](./results/SUMMARY.md).

---

## Repository Layout

- [`scripts/run_muse_sweep.py`](./scripts/run_muse_sweep.py) — Unified SGLang-TorchTPU benchmark harness supporting single-host, colocated multi-replica (`replicas > 1`), multi-host TPU slice (`nnodes > 1`), Pallas Decode Megakernel, and `DFLASH` draft overrides.
- [`scripts/muse_multi_replica_proxy.py`](./scripts/muse_multi_replica_proxy.py) — Async `aiohttp` least-inflight HTTP reverse proxy for colocated multi-replica TPU servers.
- [`scripts/stage_spark_sharded.py`](./scripts/stage_spark_sharded.py) — Multi-pod Indexed Job HuggingFace-to-GCS parallel shard downloader with byte-exact completeness verification.
- [`scripts/collect_muse_results.py`](./scripts/collect_muse_results.py) — Aggregates per-run `summary.json` artifacts from GCS (or a local copy such as [`results/raw/`](./results/raw/), `--local`) into [`results/SUMMARY.md`](./results/SUMMARY.md) and [`results/summary.json`](./results/summary.json).
- [`scripts/export_raw_results.py`](./scripts/export_raw_results.py) — Exports the raw per-cell benchmark files, run records and server-log excerpts into [`results/raw/`](./results/raw/) with an index of every measured cell.
- [`scripts/v5p_tp2_replica_recipes.py`](./scripts/v5p_tp2_replica_recipes.py) + [`scripts/run_v5p_2xtp2_gateway_bench.py`](./scripts/run_v5p_2xtp2_gateway_bench.py) — Two colocated `TP2` Glimmer servers on one `v5p` host and the gateway sweep used for `glimmer_bf16_2xtp2_gateway`.
- [`scripts/plan_*.json`](./scripts/) — Declarative sweep plans for `Muse-Spark-Open-816B-A42B` (`v7x` NVFP4 & `v5p-32` BF16) and `Muse-Glimmer-30B` (`v7x` & `v5p`).
- [`k8s/`](./k8s/) — Sanitized GKE Job manifests for single-host `v7x`/`v5p` sweeps, Pallas Decode Megakernel jobs, 8-host Indexed `v5p-32` TPU slice serving, and sharded GCS model staging.
- [`results/`](./results/) — Consolidated [`SUMMARY.md`](./results/SUMMARY.md) and [`summary.json`](./results/summary.json); [`results/raw/`](./results/raw/) holds the raw `bench_serving` output behind every number (see its [README](./results/raw/README.md)).
