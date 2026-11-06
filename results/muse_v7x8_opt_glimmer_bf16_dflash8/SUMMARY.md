# Muse-Glimmer-30B BF16 `TP8` + `DFLASH-8` (`Muse-Glimmer-30B-assistant`) on TPU `v7x` (1 Host = 4 Chips / 8 TensorCores)

- **Updated (UTC):** `2026-11-06T03:51:51.372392+00:00`
- **Hardware:** TPU `v7x` host (`4` physical chips = `8` TensorCore ranks, `192 GB` HBM3e/chip = `768 GB` HBM3e); engine `SGLang-TorchTPU` (`v0.2.0`).
- **Checkpoint:** `meta-models/Muse-Glimmer-30B` (`59.6 GB` BF16) + `meta-models/Muse-Glimmer-30B-assistant` (`DFLASH` draft capped to block-8: `--speculative-num-draft-tokens 8`, `32 × 8 = 256` verify tokens).
- **Method:** `python -m sglang.benchmark.serving --dataset-name random --random-range-ratio 1.0`, every request decodes exact `output_len` tokens (EOS ignored).

| Variant | Layout / precision | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |
|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `1024` / `1024` | 1 | 0.605 | **619.6** | 154.9 | 1,238.6 | 77.9 / 79.1 | 1.37 / 2.66 | 4.92 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `1024` / `1024` | 4 | 2.007 | **2,055.5** | 513.9 | 4,109.0 | 81.0 / 84.5 | 1.44 / 3.36 | 5.16 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `1024` / `1024` | 16 | 4.348 | **4,452.2** | 1,113.1 | 8,900.1 | 83.3 / 178.1 | 2.15 / 6.74 | 5.10 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `1024` / `1024` | 32 | 5.794 | **5,932.9** | 1,483.2 | 11,860.1 | 259.3 / 1005.3 | 3.31 / 10.78 | 4.83 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `4096` / `1024` | 1 | 0.546 | **559.2** | 139.8 | 2,795.6 | 255.0 / 256.0 | 1.19 / 2.73 | 4.83 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `4096` / `1024` | 4 | 1.491 | **1,526.4** | 381.6 | 7,630.6 | 256.5 / 514.7 | 2.11 / 4.02 | 4.85 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `4096` / `1024` | 16 | 2.663 | **2,726.4** | 681.6 | 13,629.4 | 256.8 / 821.5 | 4.37 / 11.72 | 4.89 | `PASS` |
| `glimmer_bf16_tp8_dflash8` | BF16 TP8 + `DFLASH` block-8 (`Muse-Glimmer-30B-assistant`, `256` verify tokens at `c=32`), BF16 KV | `4096` / `1024` | 32 | 3.871 | **3,964.4** | 991.1 | 19,817.9 | 751.4 / 1706.0 | 5.68 / 14.98 | 4.78 | `PASS` |
