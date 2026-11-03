# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p`) — Measured SGLang-TorchTPU Results

- **Generated (UTC):** `2026-11-03T06:14:22.081574+00:00`
- **Source runs:** 35 result directories
- **`Output tok/s/chip`** divides output throughput by the physical TPU chips occupied by that serving topology (`TP2` on `v5p` = 2 chips; `TP2` on `v7x` = 1 chip; `2×TP2` gateway on `v5p` = 4 chips; `4×TP2` / `2×TP4` / `TP8` on `v7x` = 4 chips; `TP32` on `v5p-32` = 32 chips).

## Muse-Spark-Open-816B-A42B (NVFP4 + DFlash2) — TPU `v7x` (1 host = 4 chips / 8 TensorCores)

one TPU `v7x` host (`4` chips = `8` TensorCore ranks, `192 GB` HBM/chip = `768 GB` HBM)

| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `spark_nvfp4_tp8_ep1` | NVFP4 (`modelopt_fp4`), TP8 EP1 (`pallas_tpu_fused_rs` expert-TP), BF16 KV, `262K` context | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |

### Variant status & sanity check

- `spark_nvfp4_tp8_ep1`: **PENDING** (server ready after `-` s)
  - reason: `PENDING`

## Muse-Glimmer-30B (+ DFlash assistant) — TPU `v7x` (1 host = 4 chips / 8 TensorCores)

one TPU `v7x` host (`4` chips = `8` TensorCore ranks, `192 GB` HBM/chip = `768 GB`)

| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `glimmer_bf16_2xtp4_gateway` | BF16, 2 colocated TP4 servers (`SGLANG_TPU_PD_COLOCATED=1`, 2 chips each = 4 chips total) behind async least-inflight HTTP gateway, BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `1024` / `1024` | 1 | 0.181 | 185.8 | 46.4 | 371.3 | 82 / 83 | 5.30 / 5.37 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `1024` / `1024` | 4 | 0.681 | 697.0 | 174.3 | 1,393.3 | 315 / 315 | 5.46 / 5.78 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `1024` / `1024` | 16 | 2.156 | 2,207.8 | 551.9 | 4,413.4 | 551 / 611 | 6.72 / 7.18 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `1024` / `1024` | 32 | 3.280 | 3,359.2 | 839.8 | 6,715.1 | 783 / 1,089 | 8.76 / 9.44 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `1024` / `1024` | 64 | 4.683 | 4,795.4 | 1,198.8 | 9,586.1 | 1,247 / 2,008 | 12.12 / 13.24 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `4096` / `1024` | 1 | 0.175 | 179.6 | 44.9 | 897.9 | 262 / 263 | 5.31 / 5.35 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `4096` / `1024` | 4 | 0.640 | 654.9 | 163.7 | 3,273.6 | 505 / 742 | 5.62 / 5.87 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `4096` / `1024` | 16 | 1.655 | 1,694.8 | 423.7 | 8,472.2 | 1,231 / 2,194 | 8.23 / 9.19 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `4096` / `1024` | 32 | 2.270 | 2,324.7 | 581.2 | 11,621.2 | 2,199 / 4,127 | 11.62 / 13.51 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `4096` / `1024` | 64 | 2.839 | 2,907.3 | 726.8 | 14,533.5 | 3,905 / 8,292 | 18.14 / 21.68 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `32768` / `1024` | 1 | 0.150 | 154.1 | 38.5 | 5,084.4 | 1,106 / 1,108 | 5.40 / 5.45 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `32768` / `1024` | 4 | 0.396 | 405.8 | 101.5 | 13,392.1 | 2,650 / 4,208 | 7.27 / 8.78 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `32768` / `1024` | 8 | 0.514 | 526.1 | 131.5 | 17,359.4 | 4,726 / 8,350 | 10.59 / 14.13 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `65536` / `1024` | 1 | 0.126 | 128.7 | 32.2 | 8,363.6 | 2,308 / 2,310 | 5.50 / 5.55 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `65536` / `1024` | 4 | 0.263 | 269.5 | 67.4 | 17,517.7 | 5,596 / 8,898 | 9.38 / 12.60 | - | `PASS` |
| `glimmer_bf16_tp8` | BF16 weights, TP8 (8 TensorCores), BF16 KV | 4 | `126976` / `1024` | 1 | 0.092 | 94.0 | 23.5 | 11,753.4 | 5,011 / 5,011 | 5.75 / 5.75 | - | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 1 | 0.662 | 677.6 | 169.4 | 1,354.5 | 79 / 84 | 1.21 / 2.66 | 5.44 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 4 | 2.156 | 2,207.3 | 551.8 | 4,412.5 | 83 / 312 | 1.40 / 3.31 | 6.04 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 16 | 4.377 | 4,482.4 | 1,120.6 | 8,960.4 | 88 / 562 | 2.16 / 7.52 | 6.30 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 32 | 1.202 | 1,230.6 | 307.6 | 2,460.0 | 728 / 26,023 | 24.65 / 25.31 | 1.57 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 1 | 0.604 | 618.4 | 154.6 | 3,091.4 | 255 / 256 | 1.13 / 2.84 | 1.61 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 4 | 1.542 | 1,579.0 | 394.8 | 7,893.7 | 255 / 699 | 2.02 / 4.37 | 1.69 | `PASS` |
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 16 | 2.504 | 2,564.5 | 641.1 | 12,820.2 | 263 / 2,190 | 4.15 / 12.67 | 2.00 | `PASS` |

### Variant status & sanity check

- `glimmer_bf16_2xtp4_gateway`: **PENDING** (server ready after `-` s)
  - reason: `PENDING`
- `glimmer_bf16_tp8`: **DONE** (server ready after `641.2` s)
  - sanity (greedy chat): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. Just one short sentence.  Make sure not`
  - earlier attempt: SERVER_FAILED — `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
  - earlier attempt: PENDING — `PENDING`
- `glimmer_bf16_tp8_dflash16`: **DONE** (server ready after `861.2` s) — _7 cells completed (1K conc 1..32, 4K conc 1..16); stopped at 4K conc=32 due to default SWA pool ratio eviction stall (resolved via swa_full_tokens_ratio=1.0)_
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: PENDING — `PENDING`

## Muse-Glimmer-30B (+ DFlash assistant) — TPU `v5p` (1 host = 4 chips)

one TPU `v5p` host (`4` chips, 1 JAX/PJRT device per chip, `95 GB` HBM/chip = `380 GB`), a single host of the `v5p-32` (`2x4x4`) slice used as a standalone 2x2x1 sub-slice

| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 1 | 0.065 | 66.6 | 16.6 | 133.1 | 208 / 210 | 14.83 / 14.83 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 4 | 0.282 | 288.9 | 72.2 | 577.6 | 215 / 391 | 13.60 / 13.78 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 16 | 0.991 | 1,014.7 | 253.7 | 2,028.3 | 930 / 1,452 | 14.44 / 15.21 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 32 | 1.809 | 1,852.3 | 463.1 | 3,702.7 | 938 / 2,351 | 16.13 / 16.91 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 64 | 3.012 | 3,083.9 | 771.0 | 6,164.8 | 1,812 / 3,259 | 19.02 / 20.55 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `1024` / `1024` | 128 | 4.582 | 4,691.7 | 1,172.9 | 9,378.9 | 3,166 / 6,029 | 24.18 / 26.97 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `4096` / `1024` | 1 | 0.063 | 64.2 | 16.0 | 320.8 | 750 / 751 | 14.86 / 14.87 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `4096` / `1024` | 4 | 0.259 | 265.0 | 66.2 | 1,324.5 | 1,112 / 1,477 | 14.01 / 14.37 | - | `PASS` |
| `glimmer_bf16_2xtp2_gateway` | BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV | 4 | `4096` / `1024` | 16 | 0.851 | 871.4 | 217.9 | 4,356.2 | 2,245 / 3,724 | 16.16 / 17.64 | - | `PASS` |
| `glimmer_bf16_tp1_dp4` | BF16, 4 data-parallel replicas x TP1 (one full copy of the weights per chip), BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 1 | 0.065 | 66.6 | 33.3 | 133.1 | 213 / 213 | 14.81 / 14.86 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 4 | 0.268 | 274.2 | 137.1 | 548.1 | 736 / 917 | 13.83 / 14.42 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 16 | 0.915 | 936.7 | 468.3 | 1,872.4 | 934 / 1,654 | 16.18 / 16.89 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 32 | 1.507 | 1,542.9 | 771.5 | 3,084.4 | 2,354 / 3,258 | 18.53 / 20.58 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 64 | 2.286 | 2,340.9 | 1,170.5 | 4,679.6 | 3,788 / 6,130 | 23.68 / 27.09 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 1 | 0.063 | 64.2 | 32.1 | 321.2 | 751 / 752 | 14.84 / 14.84 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 4 | 0.242 | 247.9 | 124.0 | 1,239.4 | 1,496 / 2,218 | 14.68 / 15.41 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 16 | 0.691 | 707.5 | 353.7 | 3,536.7 | 3,735 / 6,700 | 18.98 / 21.90 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 32 | 1.001 | 1,024.9 | 512.5 | 5,123.6 | 6,725 / 12,677 | 24.67 / 30.51 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 64 | 1.292 | 1,322.8 | 661.4 | 6,612.6 | 11,249 / 25,409 | 37.57 / 47.79 | - | `PASS` |
| `glimmer_bf16_tp2_dp2` | BF16, 2 data-parallel replicas x TP2 (2 chips each), BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `1024` / `1024` | 1 | 0.116 | 119.1 | 29.8 | 238.0 | 138 / 139 | 8.27 / 8.27 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `1024` / `1024` | 4 | 0.429 | 439.1 | 109.8 | 877.7 | 557 / 559 | 8.55 / 9.02 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `1024` / `1024` | 16 | 1.419 | 1,452.8 | 363.2 | 2,904.2 | 569 / 1,035 | 10.43 / 10.89 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `1024` / `1024` | 32 | 2.248 | 2,302.4 | 575.6 | 4,602.5 | 1,015 / 1,870 | 12.92 / 13.76 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `1024` / `1024` | 64 | 3.244 | 3,322.0 | 830.5 | 6,640.7 | 2,263 / 3,652 | 17.09 / 19.09 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `4096` / `1024` | 1 | 0.112 | 114.4 | 28.6 | 572.0 | 457 / 458 | 8.30 / 8.30 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `4096` / `1024` | 4 | 0.391 | 400.6 | 100.1 | 2,002.5 | 899 / 1,331 | 9.11 / 9.55 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `4096` / `1024` | 16 | 1.078 | 1,104.3 | 276.1 | 5,520.4 | 2,224 / 3,979 | 12.32 / 14.05 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `4096` / `1024` | 32 | 1.518 | 1,554.7 | 388.7 | 7,771.9 | 3,991 / 7,509 | 16.69 / 20.14 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `4096` / `1024` | 64 | 1.919 | 1,964.6 | 491.2 | 9,821.1 | 6,658 / 15,071 | 26.17 / 32.21 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `32768` / `1024` | 1 | 0.094 | 96.3 | 24.1 | 3,179.2 | 2,003 / 2,016 | 8.42 / 8.42 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `32768` / `1024` | 4 | 0.233 | 238.3 | 59.6 | 7,862.6 | 4,889 / 7,778 | 12.01 / 14.84 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `32768` / `1024` | 8 | 0.308 | 315.0 | 78.7 | 10,393.1 | 8,739 / 15,483 | 16.87 / 23.46 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `65536` / `1024` | 1 | 0.077 | 78.4 | 19.6 | 5,097.1 | 4,305 / 4,310 | 8.55 / 8.55 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `65536` / `1024` | 4 | 0.150 | 153.3 | 38.3 | 9,964.0 | 10,544 / 16,796 | 15.80 / 21.91 | - | `PASS` |
| `glimmer_bf16_tp4` | BF16 weights, TP4 (attention + MLP tensor-parallel over the 4 chips), BF16 KV | 4 | `126976` / `1024` | 1 | 0.054 | 54.9 | 13.7 | 6,865.3 | 9,678 / 9,678 | 8.77 / 8.77 | - | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 1 | 0.504 | 516.5 | 129.1 | 1,032.4 | 135 / 136 | 1.52 / 3.43 | 5.64 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 4 | 1.505 | 1,540.8 | 385.2 | 3,080.1 | 138 / 569 | 2.15 / 4.81 | 6.12 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 16 | 2.452 | 2,511.3 | 627.8 | 5,020.2 | 156 / 1,017 | 3.51 / 12.64 | 6.11 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `1024` / `1024` | 32 | 2.623 | 2,685.7 | 671.4 | 5,368.8 | 842 / 2,449 | 7.50 / 25.49 | 5.69 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 1 | 0.463 | 473.9 | 118.5 | 2,369.0 | 453 / 457 | 1.40 / 3.46 | 5.70 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 4 | 0.988 | 1,012.1 | 253.0 | 5,059.5 | 455 / 1,278 | 3.23 / 6.89 | 5.72 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 16 | 1.438 | 1,472.1 | 368.0 | 7,359.0 | 475 / 4,070 | 7.10 / 24.64 | 5.78 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `4096` / `1024` | 32 | 1.796 | 1,838.8 | 459.7 | 9,192.0 | 1,821 / 10,351 | 12.07 / 34.60 | 5.62 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `32768` / `1024` | 1 | 0.290 | 296.8 | 74.2 | 9,795.0 | 2,036 / 2,041 | 0.87 / 2.37 | 5.63 | `PASS` |
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | `32768` / `1024` | 4 | 0.365 | 373.7 | 93.4 | 12,331.1 | 2,995 / 7,813 | 6.98 / 12.33 | 5.63 | `PASS` |
| `glimmer_bf16_tp4_dflash16_greedy` | same as glimmer_bf16_tp4_dflash16 but greedy decoding (temperature 0) - best-case draft acceptance | 4 | `1024` / `1024` | 1 | 0.506 | 518.3 | 129.6 | 1,036.2 | 135 / 135 | 1.51 / 3.42 | 5.64 | `PASS` |
| `glimmer_bf16_tp4_dflash16_greedy` | same as glimmer_bf16_tp4_dflash16 but greedy decoding (temperature 0) - best-case draft acceptance | 4 | `1024` / `1024` | 4 | 1.546 | 1,583.5 | 395.9 | 3,165.4 | 137 / 565 | 2.11 / 4.76 | 6.20 | `PASS` |
| `glimmer_bf16_tp4_dflash16_greedy` | same as glimmer_bf16_tp4_dflash16 but greedy decoding (temperature 0) - best-case draft acceptance | 4 | `1024` / `1024` | 16 | 2.584 | 2,646.0 | 661.5 | 5,289.4 | 155 / 1,000 | 3.72 / 12.49 | 6.15 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `1024` / `1024` | 1 | 0.086 | 88.4 | 22.1 | 176.7 | 152 / 154 | 11.22 / 16.52 | 2.62 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `1024` / `1024` | 4 | 0.348 | 356.4 | 89.1 | 712.5 | 176 / 565 | 10.24 / 18.95 | 2.98 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `1024` / `1024` | 16 | 0.821 | 840.3 | 210.1 | 1,679.8 | 212 / 1,009 | 15.68 / 30.03 | 3.23 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `1024` / `1024` | 32 | 1.047 | 1,072.3 | 268.1 | 2,143.7 | 1,451 / 5,299 | 25.46 / 56.20 | 3.33 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `4096` / `1024` | 1 | 0.102 | 104.5 | 26.1 | 522.4 | 469 / 469 | 9.61 / 15.20 | 3.32 | `PASS` |
| `glimmer_bf16_tp4_dflash16_t1` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), sampled decoding T=1.0 / top-p 0.95 (model-card recommendation; top-k 64 is not settable from this client) | 4 | `4096` / `1024` | 16 | 0.691 | 707.4 | 176.8 | 3,536.1 | 532 / 3,769 | 15.52 / 43.51 | 3.40 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `1024` / `1024` | 1 | 0.449 | 460.2 | 115.0 | 919.9 | 134 / 135 | 1.78 / 3.75 | 4.70 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `1024` / `1024` | 4 | 1.370 | 1,402.7 | 350.7 | 2,804.1 | 136 / 567 | 2.18 / 4.53 | 5.12 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `1024` / `1024` | 16 | 2.781 | 2,847.3 | 711.8 | 5,691.9 | 143 / 1,013 | 3.57 / 10.77 | 5.09 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `1024` / `1024` | 32 | 3.391 | 3,472.7 | 868.2 | 6,942.0 | 672 / 1,901 | 6.21 / 19.97 | 4.84 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `4096` / `1024` | 1 | 0.408 | 418.0 | 104.5 | 2,089.7 | 454 / 456 | 1.53 / 3.77 | 4.85 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `4096` / `1024` | 4 | 0.939 | 961.2 | 240.3 | 4,805.1 | 454 / 1,280 | 3.09 / 6.17 | 4.86 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `4096` / `1024` | 16 | 1.501 | 1,537.0 | 384.2 | 7,683.3 | 462 / 3,785 | 7.69 / 19.52 | 4.89 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `4096` / `1024` | 32 | 2.181 | 2,233.4 | 558.4 | 11,164.9 | 1,522 / 10,195 | 9.65 / 28.11 | 4.77 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `32768` / `1024` | 1 | 0.257 | 262.9 | 65.7 | 8,675.3 | 2,030 / 2,035 | 1.42 / 2.71 | 4.77 | `PASS` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | `32768` / `1024` | 4 | 0.331 | 339.0 | 84.7 | 11,185.8 | 2,036 / 7,794 | 7.83 / 11.05 | 4.76 | `PASS` |
| `glimmer_bf16_tp4_dpattn2` | BF16, DP-attention: 2 attention replicas x attention-TP2 (no KV-head replication), MLP TP4, BF16 KV; text-only (language_model_only=true: vision tower not built, required because VisionAttention does not support attn-TP < TP) | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16_tp4_dpattn4` | BF16, DP-attention: 4 attention replicas x attention-TP1, MLP TP4, BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 1 | 0.116 | 118.7 | 29.7 | 237.4 | 140 / 141 | 8.29 / 8.29 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 16 | 1.405 | 1,439.0 | 359.7 | 2,876.6 | 573 / 1,004 | 10.56 / 10.99 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 64 | 3.193 | 3,269.3 | 817.3 | 6,535.4 | 1,902 / 3,593 | 17.75 / 19.43 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `4096` / `1024` | 1 | 0.111 | 114.0 | 28.5 | 570.0 | 461 / 463 | 8.33 / 8.33 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `4096` / `1024` | 16 | 1.067 | 1,092.2 | 273.1 | 5,459.9 | 2,242 / 4,009 | 12.46 / 14.20 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `4096` / `1024` | 64 | 1.888 | 1,933.5 | 483.4 | 9,665.6 | 6,712 / 15,182 | 26.63 / 32.72 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `32768` / `1024` | 1 | 0.094 | 95.8 | 23.9 | 3,159.8 | 2,015 / 2,016 | 8.48 / 8.48 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `32768` / `1024` | 4 | 0.230 | 235.1 | 58.8 | 7,759.1 | 4,944 / 7,862 | 12.19 / 15.03 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `32768` / `1024` | 8 | 0.302 | 309.7 | 77.4 | 10,221.4 | 8,828 / 15,631 | 17.21 / 23.86 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `65536` / `1024` | 1 | 0.076 | 77.7 | 19.4 | 5,048.8 | 4,348 / 4,354 | 8.63 / 8.63 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `65536` / `1024` | 4 | 0.147 | 150.9 | 37.7 | 9,809.8 | 10,646 / 16,973 | 16.11 / 22.29 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 1 | 0.022 | 22.4 | 11.2 | 44.8 | 416 / 417 | 44.30 / 44.30 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 4 | 0.085 | 87.4 | 43.7 | 174.6 | 1,374 / 1,691 | 44.40 / 45.49 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 16 | 0.320 | 327.4 | 163.7 | 654.5 | 1,705 / 3,023 | 47.47 / 47.95 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 32 | 0.582 | 595.9 | 298.0 | 1,191.2 | 3,043 / 5,680 | 51.01 / 52.31 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 64 | 0.975 | 998.5 | 499.3 | 1,996.1 | 5,683 / 10,952 | 58.84 / 62.71 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `4096` / `1024` | 1 | 0.021 | 21.9 | 11.0 | 109.5 | 1,389 / 1,390 | 44.33 / 44.33 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `4096` / `1024` | 4 | 0.082 | 84.2 | 42.1 | 420.7 | 2,708 / 3,842 | 45.85 / 46.97 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `4096` / `1024` | 16 | 0.271 | 277.4 | 138.7 | 1,386.8 | 6,758 / 11,326 | 52.05 / 56.51 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `4096` / `1024` | 32 | 0.440 | 450.2 | 225.1 | 2,250.6 | 12,158 / 21,639 | 60.19 / 69.44 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `4096` / `1024` | 64 | 0.621 | 636.1 | 318.1 | 3,180.0 | 20,317 / 46,016 | 81.10 / 99.56 | - | `PASS` |
| `glimmer_fp8_tp4` | FP8 W8A8 online quantization (channelwise, emulated on v5p which has no FP8 MXU), TP4, FP8 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |

### Variant status & sanity check

- `glimmer_bf16_2xtp2_gateway`: **RUNNING** (server ready after `1.0` s)
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
- `glimmer_bf16_tp1_dp4`: **SERVER_FAILED** (server ready after `60.1` s)
  - reason: `RuntimeError: failed to transfer to 'tpu' device: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/0' because it is already opened by another process on this host.`
  - earlier attempt: SERVER_FAILED — `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
- `glimmer_bf16_tp2`: **DONE** (server ready after `521.1` s)
  - sanity (greedy chat): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. The instruction: In one short sentence.`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: PENDING — `PENDING`
- `glimmer_bf16_tp2_dp2`: **SERVER_FAILED** (server ready after `60.2` s)
  - reason: `RuntimeError: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/1' because it is already opened by another process on this host.`
  - earlier attempt: SERVER_FAILED — `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
- `glimmer_bf16_tp4`: **DONE** (server ready after `521.1` s)
  - sanity (greedy chat): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. The instruction: In one short sentence.`
  - earlier attempt: SERVER_FAILED — `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
  - earlier attempt: SERVER_FAILED — `torch._dynamo.exc.Unsupported: Unsupported method call`
- `glimmer_bf16_tp4_dflash16`: **DONE** (server ready after `741.2` s)
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: SERVER_FAILED — `TypeError: zeros(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got NoneType"`
- `glimmer_bf16_tp4_dflash16_greedy`: **RUNNING** (server ready after `741.2` s)
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
  - earlier attempt: SERVER_FAILED — `TypeError: zeros(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got NoneType"`
- `glimmer_bf16_tp4_dflash16_t1`: **DONE** (server ready after `761.2` s)
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
- `glimmer_bf16_tp4_dflash8`: **DONE** (server ready after `721.2` s)
  - sanity (greedy chat): `o extra fluff.  Answer: The capital of France is Paris.  That's it.  Potential nuance: Should we follow exactly? Yes.  Output one short sentence.  Done.assistant to=userThe capital of France is Paris.`
  - earlier attempt: PENDING — `PENDING`
- `glimmer_bf16_tp4_dpattn2`: **SERVER_FAILED** (server ready after `120.2` s)
  - reason: `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.`
  - earlier attempt: PENDING — `PENDING`
  - earlier attempt: SERVER_FAILED — `ValueError: VisionAttention shards over the attention TP group (2 ranks) but all-reduces over the full TP group (4 ranks), so it does not support --attn-dp-size 2 yet. Use --attn-dp-size equal to --tp`
- `glimmer_bf16_tp4_dpattn4`: **SERVER_FAILED** (server ready after `120.2` s)
  - reason: `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.`
- `glimmer_bf16w_fp8kv_tp4`: **DONE** (server ready after `541.2` s)
  - sanity (greedy chat): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  Follow instruction.  No policy violation.  We can comply.  Possibly add nothing else. Just one short sentence.  Make sure `
- `glimmer_fp8_tp2`: **DONE** (server ready after `841.2` s)
  - sanity (greedy chat): `entially ensure short. Yes.  No policy violation.  We can comply.  Possibly add period.  Make sure it's one short sentence.  Output: The capital of France is Paris.  That's fine.  Potential nuance: Sh`
- `glimmer_fp8_tp4`: **SERVER_FAILED** (server ready after `60.1` s)
  - reason: `ValueError: Muse Glimmer unfused q/k/v needs num_key_value_heads >= tp_size (got 2 kv heads, tp_size=4); the KV-head replication QKVParallelLinear does is not reproduced here.`
  - earlier attempt: SERVER_FAILED — `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`

