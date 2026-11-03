# Meta Muse on Cloud TPU (`v7x` & `v5p`) with SGLang-TorchTPU

Serving and benchmarking **[`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B)** (with its [`Muse-Glimmer-30B-assistant`](https://huggingface.co/meta-models/Muse-Glimmer-30B-assistant) speculative draft) on a single TPU `v7x` host and a single TPU `v5p` host using **SGLang-TorchTPU** (`sglang` build `b3d718cb`, TorchTPU `0.2.0`). `Muse-Spark-Open-816B-A42B` (BF16 / NVFP4) is staged next and waits on gated-repo access for the serving account.

> **Status: interim results** (sweeps still running; this README is updated as runs finish). Per-run tables with every attempt and failure: [`results/SUMMARY.md`](./results/SUMMARY.md) · raw data: [`results/summary.json`](./results/summary.json).

## Headline results (BF16, text-only, `1024` in / `1024` out, fixed-length requests with EOS ignored)

| Hardware | Layout | Conc. 1: out tok/s (TPOT p50) | Conc. 4 | Conc. 16 | Conc. 32 | Conc. 64 | TTFT p50 at conc. 1 |
|---|---|---:|---:|---:|---:|---:|---:|
| **TPU `v7x`, 1 host** (4 chips = 8 TensorCores) | TP8 | **185.8** (`5.30 ms`) | 697.0 | 2,207.8 | 3,359.2 | **4,795.4** | `82 ms` |
| **TPU `v5p`, 1 host** (4 chips) | TP4 | **119.1** (`8.27 ms`) | 439.1 | 1,452.8 | 2,302.4 | **3,322.0** | `138 ms` |
| TPU `v5p`, 2 chips | TP2 | 66.6 (`14.81 ms`) | 274.2 | 936.7 | 1,542.9 | 2,340.9 | `213 ms` |

- **Long context (`v7x` TP8):** `4096` in: 179.6 tok/s at conc. 1, 2,907.3 at conc. 64; `32768` in: 154.1 tok/s at conc. 1 (TTFT `1.11 s`). `64K` / `124K` runs in progress.
- **Throughput per chip on `v5p`:** TP2 reaches 1,170 output tok/s/chip at conc. 64 vs 831 for TP4, so two TP2 replicas per host would out-throughput one TP4 replica (plain `--dp-size` does not work in this build, see below; two pinned servers per host are being tested).
- **Speculative decoding (DFLASH, assistant draft, 16 draft tokens, model-default sampling T=1.0 / top-p 0.95 / top-k 64), `v5p` TP4, conc. 1, `1024` in / `256` out (single probe run):** **349 output tok/s, accept length 4.70, TPOT p50 `2.20 ms`** vs 119 tok/s without speculation (**~2.9x**). Full DFLASH sweeps (16 tokens default sampling, 16 tokens greedy, 8 tokens) are running on `v5p` and queued on `v7x`.
- **FP8 on `v5p`:** `v5p` has no FP8 matrix unit, so FP8 W8A8 is emulated and ~3x slower than BF16 (TP2 conc. 1: 22.4 vs 66.6 tok/s). FP8 KV cache with BF16 weights gives no gain on `v5p` (TP4 conc. 64: 3,269 vs 3,322 tok/s).

## What had to change to run Muse-Glimmer on TPU (all runs)

1. **Text-only serving** via `--json-model-override-args '{"language_model_only": true}'`: serving the full multimodal checkpoint crashes on the first request on TPU (`torch.compile` cannot trace the multimodal-embedding path, a `msgspec.Struct.__setattr__` in `mm_schedule._move_items_to_device`). The vision tower is therefore not built; every number above is text-only.
2. **`language_only` server arg rejected:** it selects encoder-disaggregation mode, which does not support `MuseGlimmerForConditionalGeneration`.
3. **DFLASH draft config:** `Muse-Glimmer-30B-assistant/config.json` has no `vocab_size`, so the TPU spec-decode warmup crashed (`draft_vocab_size=None`). The harness writes `vocab_size=202048` (the target model's vocabulary) into the host-RAM copy of the draft config; the checkpoint itself is unchanged.
4. **Layouts that do not work in this build (recorded, not worked around):** FP8 with TP4 (the model's unfused q/k/v needs `num_key_value_heads (2) >= tp`), plain data parallel `--dp-size > 1` (TorchTPU cannot open the same `/dev/vfio` device from two scheduler processes), and DP-attention (attention-TP1: Pallas RPA DMA shape mismatch on `v5p`; attention-TP2: LM-head all-gather shape mismatch).

## Method

- One TPU host per server, used as a standalone sub-slice (`FORCE_SINGLE_HOST=1`): `v7x` host of a `2x2x2` slice, `v5p` host of a `2x4x4` slice. Several `v5p` variants ran in parallel, one per host.
- Client: fixed-length random-token prompts, `ignore_eos`, streaming; output tok/s = generated tokens / wall time; TTFT / TPOT percentiles from the stream. Each cell after compile warmup.
- Model weights: BF16 checkpoint as published (`59.6 GB`); FP8 variants use SGLang online FP8 quantization (channel-wise).

## Repository layout

- [`scripts/`](./scripts/) - sweep harness (`run_muse_sweep.py`), plans, job generator, HF-to-GCS staging, result collector, DFLASH probe.
- [`k8s/`](./k8s/) - per-variant GKE Job manifests (single-host TPU jobs on `v5p` / `v7x`) and staging jobs.
- [`results/`](./results/) - merged `SUMMARY.md` / `summary.json` across all runs and attempts.
