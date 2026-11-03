# Meta Muse on TPU with SGLang-TorchTPU - measured results

- Generated (UTC): `2026-11-03T04:24:48.994181+00:00`
- Source runs: 30 result folders under `gs://YOUR_GCS_BUCKET/benchmark-results/` (prefix `muse_`)
- `Output tok/s/chip` divides by the chips the variant occupies (TP2 on v5p = 2 chips, TP2 on v7x = 1 chip).

## v5p

one TPU `v5p` host (`4` chips, 1 JAX/PJRT device per chip, `95 GB` HBM/chip = `380 GB`), a single host of the `v5p-32` (`2x4x4`) slice used as a standalone 2x2x1 sub-slice

| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `glimmer_bf16_tp1_dp4` | BF16, 4 data-parallel replicas x TP1 (one full copy of the weights per chip), BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 1 | 0.065 | 66.6 | 33.3 | 133.1 | 213 / 213 | 14.81 / 14.86 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 4 | 0.268 | 274.2 | 137.1 | 548.1 | 736 / 917 | 13.83 / 14.42 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 16 | 0.915 | 936.7 | 468.3 | 1,872.4 | 934 / 1,654 | 16.18 / 16.89 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 32 | 1.507 | 1,542.9 | 771.5 | 3,084.4 | 2,354 / 3,258 | 18.53 / 20.58 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `1024` / `1024` | 64 | 2.286 | 2,340.9 | 1,170.5 | 4,679.6 | 3,788 / 6,130 | 23.68 / 27.09 | - | `PASS` |
| `glimmer_bf16_tp2` | BF16, TP2 on 2 of the 4 chips (single replica; plain DP is not supported by this SGLang-TorchTPU build) | 2 | `4096` / `1024` | 1 | 0.063 | 64.2 | 32.1 | 321.2 | 751 / 752 | 14.84 / 14.84 | - | `PASS` |
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
| `glimmer_bf16_tp4_dflash16` | BF16, TP4, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |
| `glimmer_bf16_tp4_dflash16_greedy` | same as glimmer_bf16_tp4_dflash16 but greedy decoding (temperature 0) - best-case draft acceptance | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |
| `glimmer_bf16_tp4_dflash8` | BF16, TP4, BF16 KV + DFLASH (assistant draft) verifying 8 draft tokens per step (speculative_num_draft_tokens=8, checkpoint block 16), model-default sampling | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |
| `glimmer_bf16_tp4_dpattn2` | BF16, DP-attention: 2 attention replicas x attention-TP2 (no KV-head replication), MLP TP4, BF16 KV; text-only (language_model_only=true: vision tower not built, required because VisionAttention does not support attn-TP < TP) | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16_tp4_dpattn4` | BF16, DP-attention: 4 attention replicas x attention-TP1, MLP TP4, BF16 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 1 | 0.116 | 118.7 | 29.7 | 237.4 | 140 / 141 | 8.29 / 8.29 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 16 | 1.405 | 1,439.0 | 359.7 | 2,876.6 | 573 / 1,004 | 10.56 / 10.99 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `1024` / `1024` | 64 | 3.193 | 3,269.3 | 817.3 | 6,535.4 | 1,902 / 3,593 | 17.75 / 19.43 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `4096` / `1024` | 1 | 0.111 | 114.0 | 28.5 | 570.0 | 461 / 463 | 8.33 / 8.33 | - | `PASS` |
| `glimmer_bf16w_fp8kv_tp4` | BF16 weights, TP4, FP8 (e4m3) KV cache | 4 | `4096` / `1024` | 16 | 1.067 | 1,092.2 | 273.1 | 5,459.9 | 2,242 / 4,009 | 12.46 / 14.20 | - | `PASS` |
| `glimmer_fp8_tp2` | FP8 W8A8 online quantization (channelwise; v5p has no FP8 MXU), TP2 on 2 of the 4 chips (FP8 requires tp<=2: kv_heads=2), FP8 KV | 2 | `1024` / `1024` | 1 | 0.022 | 22.4 | 11.2 | 44.8 | 416 / 417 | 44.30 / 44.30 | - | `PASS` |
| `glimmer_fp8_tp4` | FP8 W8A8 online quantization (channelwise, emulated on v5p which has no FP8 MXU), TP4, FP8 KV | 4 | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |

### Variant status

- `glimmer_bf16_tp1_dp4`: **SERVER_FAILED** (server ready after `60.1` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dp4_20261103T034747Z/`)
  - reason: `RuntimeError: failed to transfer to 'tpu' device: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/0' because it is already opened by another process on this host.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dp4_20261103T034531Z/`: SERVER_FAILED - `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
- `glimmer_bf16_tp2`: **RUNNING** (server ready after `521.1` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp2_20261103T040242Z/`)
  - sanity (greedy chat, answer tail): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. The instruction: In one short sentence.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp2_20261103T035325Z/`: PENDING - `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp2_20261103T040137Z/`: PENDING - `PENDING`
- `glimmer_bf16_tp2_dp2`: **SERVER_FAILED** (server ready after `60.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dp2_20261103T034746Z/`)
  - reason: `RuntimeError: [TorchTPU] Failed to acquire a TPU device node '/dev/vfio/1' because it is already opened by another process on this host.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dp2_20261103T034530Z/`: SERVER_FAILED - `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
- `glimmer_bf16_tp4`: **RUNNING** (server ready after `521.1` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp4_20261103T040134Z/`)
  - sanity (greedy chat, answer tail): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. The instruction: In one short sentence.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp4_20261103T034527Z/`: SERVER_FAILED - `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_tp4_20261103T034744Z/`: SERVER_FAILED - `torch._dynamo.exc.Unsupported: Unsupported method call`
- `glimmer_bf16_tp4_dflash16`: **PENDING** (server ready after `-` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash_20261103T041907Z/`)
  - reason: `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash_20261103T034528Z/`: PENDING - `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash_20261103T034745Z/`: PENDING - `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash_20261103T040136Z/`: PENDING - `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash_20261103T040242Z/`: SERVER_FAILED - `TypeError: zeros(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got NoneType"`
- `glimmer_bf16_tp4_dflash16_greedy`: **PENDING** (server ready after `-` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_greedy_20261103T041908Z/`)
  - reason: `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_greedy_20261103T040233Z/`: SERVER_FAILED - `TypeError: zeros(): argument 'size' failed to unpack the object at pos 2 with error "type must be tuple of ints,but got NoneType"`
- `glimmer_bf16_tp4_dflash8`: **PENDING** (server ready after `-` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash8_20261103T041939Z/`)
  - reason: `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dflash8_20261103T040749Z/`: PENDING - `PENDING`
- `glimmer_bf16_tp4_dpattn2`: **SERVER_FAILED** (server ready after `120.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dpattn2_20261103T035720Z/`)
  - reason: `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dpattn2_20261103T035323Z/`: PENDING - `PENDING`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dpattn2_20261103T035435Z/`: SERVER_FAILED - `ValueError: VisionAttention shards over the attention TP group (2 ranks) but all-reduces over the full TP group (4 ranks), so it does not support --attn-dp-size 2 yet. Use --attn-dp-size equal to --tp`
- `glimmer_bf16_tp4_dpattn4`: **SERVER_FAILED** (server ready after `120.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_dpattn4_20261103T035324Z/`)
  - reason: `RuntimeError: Rank 0 scheduler died during initialization (exit code: -3). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer. Run `dmesg -T | grep -i oom` to check.`
- `glimmer_bf16w_fp8kv_tp4`: **RUNNING** (server ready after `541.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_fp8kv_20261103T040751Z/`)
  - sanity (greedy chat, answer tail): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  Follow instruction.  No policy violation.  We can comply.  Possibly add nothing else. Just one short sentence.  Make sure `
- `glimmer_fp8_tp2`: **RUNNING** (server ready after `841.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_fp8tp2_20261103T040234Z/`)
  - sanity (greedy chat, answer tail): `entially ensure short. Yes.  No policy violation.  We can comply.  Possibly add period.  Make sure it's one short sentence.  Output: The capital of France is Paris.  That's fine.  Potential nuance: Sh`
- `glimmer_fp8_tp4`: **SERVER_FAILED** (server ready after `60.1` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_fp8_20261103T034749Z/`)
  - reason: `ValueError: Muse Glimmer unfused q/k/v needs num_key_value_heads >= tp_size (got 2 kv heads, tp_size=4); the KV-head replication QKVParallelLinear does is not reproduced here.`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v5p_1host_fp8_20261103T034533Z/`: SERVER_FAILED - `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`

## v7x

one TPU `v7x` host (`4` chips = `8` TensorCore ranks, `192 GB` HBM/chip = `768 GB`), host `gke-tpu-771db71a-cz6l` of the `tpu-v7x-cluster-8-pool-b` (`2x2x2`) slice used standalone

| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|
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
| `glimmer_bf16_tp8_dflash16` | BF16, TP8, BF16 KV + DFLASH speculative decoding with the Muse-Glimmer-30B-assistant draft (block 16), model-default sampling (T=1.0, top-p 0.95, top-k 64) | 4 | - | - | - | - | - | - | - | - | - | `PENDING` |

### Variant status

- `glimmer_bf16_tp8`: **RUNNING** (server ready after `641.2` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v7x_1host_all_20261103T040138Z/`)
  - sanity (greedy chat, answer tail): `.  Potentially ensure it's short. Could be "Paris is the capital of France."  That's fine.  No policy violation.  We should just answer.  Possibly add no extra. Just one short sentence.  Make sure not`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v7x_1host_all_20261103T034521Z/`: SERVER_FAILED - `ValueError: Model type MuseGlimmerForConditionalGeneration is not supported for encoder disaggregation. Supported architectures: Qwen2VL, Qwen3VL, Qwen3.5, InternS2, Qwen2Audio, Qwen2.5Omni, Dots3-Not`
  - earlier attempt `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v7x_1host_all_20261103T034750Z/`: PENDING - `PENDING`
- `glimmer_bf16_tp8_dflash16`: **PENDING** (server ready after `-` s; `gs://YOUR_GCS_BUCKET/benchmark-results/muse_glimmer30b_sglang_v7x_1host_all_20261103T034521Z/`)
  - reason: `PENDING`

