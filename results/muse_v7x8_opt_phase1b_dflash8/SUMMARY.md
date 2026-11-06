# Meta Muse-Spark-Open-816B-A42B-NVFP4 (+ DFlash2) & Muse-Glimmer-30B (+ DFlash assistant) SGLang-TorchTPU AgentX / 8-Chip Optimization Sweep on TPU v7x-8 (2 hosts = 8 chips / 16 TensorCores, 2x2x2 ICI, 1.536 TB HBM3e)

- **Updated (UTC):** `2026-11-06T02:23:47.443404+00:00`
- **Hardware:** TPU `v7x-8` slice (`2` hosts, `8` physical chips = `16` TensorCore ranks in `2x2x2` ICI, `192 GB` HBM3e/chip = `1.536 TB` HBM3e); engine `SGLang-TorchTPU` (`us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0`).
- **Checkpoint:** `meta-models/Muse-Spark-Open-816B-A42B-NVFP4 + RadixArk/Muse-Spark-Open-DFlash2 & meta-models/Muse-Glimmer-30B + Muse-Glimmer-30B-assistant` (`533.0` GB).
- **Method:** `python -m sglang.benchmark.serving --dataset-name random --random-range-ratio 1.0`, every request decodes exact `output_len` tokens (EOS ignored).

| Variant | Layout / precision | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `spark_nvfp4_tp8_dflash8_megakernel_sal_fp8_r48` | AgentX `SEQ_ALONG_LANE` FP8 (`256 B/tok`, `1M` ctx) 62-layer Pallas Decode Megakernel + `48`-row large-step (`mrr=6`) + `Y_EARLY` MoE send-overlap + batched RPA prefill + Fused-RS ETP + real DFLASH-8 (1 host = 4 chips) | - | - | - | - | - | - | - | - | - | `SERVER_FAILED` |
| `spark_nvfp4_2xtp8_megakernel_sal_r48_gateway_8chip` | 2-host Distributed Gateway (`2 x TP8` AgentX `SEQ_ALONG_LANE` FP8 48-row Megakernel + real DFLASH-8 = `96` verify rows across all `8` v7x chips / `16` TensorCores) | - | - | - | - | - | - | - | - | - | `PENDING` |
| `spark_nvfp4_tp8_dflash8_etp_opt` | NVFP4 TP8 EP1 + Fused-RS ETP (`LOCAL_ROWS`, `TILED_REDUCE`, `SKIP_OUT_ZERO`, `FUSED_COMBINE`) + RadixCache + real DFLASH-8 (`mrr=32`, FP8 KV, `262K` ctx, 1 host = 4 chips) | - | - | - | - | - | - | - | - | - | `PENDING` |

## Variant status / boot info

- `spark_nvfp4_tp8_dflash8_megakernel_sal_fp8_r48`: **SERVER_FAILED**, server ready after `3348.2` s
- `spark_nvfp4_2xtp8_megakernel_sal_r48_gateway_8chip`: **PENDING**, server ready after `-` s
- `spark_nvfp4_tp8_dflash8_etp_opt`: **PENDING**, server ready after `-` s
