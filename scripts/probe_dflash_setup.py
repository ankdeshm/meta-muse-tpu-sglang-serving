"""Probe-pod helper: patch the staged draft config and write the DFLASH probe recipe."""
import json
from pathlib import Path

draft_cfg = Path("/dev/shm/models/Muse-Glimmer-30B-assistant/config.json")
cfg = json.loads(draft_cfg.read_text())
cfg["vocab_size"] = 202048
draft_cfg.write_text(json.dumps(cfg, indent=2))
print("patched draft vocab_size", cfg["vocab_size"])

recipe = Path("/workspace/sglang/benchmark/tpu/recipe/configs/muse_custom/probe_glimmer_dflash.py")
recipe.parent.mkdir(parents=True, exist_ok=True)
recipe.write_text('''import json
from copy import deepcopy
from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG
CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update({
    "model_path": "/dev/shm/models/Muse-Glimmer-30B", "served_model_name": "meta-models/Muse-Glimmer-30B",
    "json_model_override_args": json.dumps({"language_model_only": True}),
    "tp_size": 4, "quantization": "unquant", "kv_cache_dtype": "bfloat16",
    "context_length": 8192, "max_running_requests": 4, "torch_compile_max_bs": 4, "tpu_compile_bs": [1, 2, 4],
    "tpu_compile_tokens": [512, 2048], "chunked_prefill_size": 2048, "max_prefill_tokens": 2048,
    "max_total_tokens": 65536, "mem_fraction_static": 0.85, "watchdog_timeout": 3600,
    "speculative_algorithm": "DFLASH", "speculative_draft_model_path": "/dev/shm/models/Muse-Glimmer-30B-assistant",
    "speculative_num_draft_tokens": 16, "speculative_draft_attention_backend": "pallas_tpu_rpa_v3",
    "sampling_backend": "pytorch",
})
CONFIG["runtime_env"].update({"FORCE_SINGLE_HOST": "1", "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
                              "SGLANG_SPEC_TP_SYNC": "rng", "SGLANG_TPU_ALLOW_SPEC_RECOMPILE": "1"})
''')
print("wrote", recipe)
