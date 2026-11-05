"""Writes two SGLang-TorchTPU recipes that serve Muse-Glimmer-30B as independent TP2 replicas on one TPU v5p host.

Replica A uses chips 0-1 (port 30000, `base_gpu_id=0`) and replica B chips 2-3 (port 30001, `base_gpu_id=2`).
`SGLANG_TPU_PD_COLOCATED=1` maps each rank to device `base_gpu_id + local_rank`, and each replica gets its own
compile-cache namespace. Run inside the SGLang-TorchTPU container (pod spec: k8s/muse-probe-v5p.yaml) with the
checkpoint copied to /dev/shm/models/Muse-Glimmer-30B, then start each recipe with the standard launcher from
/workspace/sglang:
    python3 -m benchmark.tpu.launch_server --config recipe/configs/muse_custom/probe_glimmer_tp2_rA.py
    python3 -m benchmark.tpu.launch_server --config recipe/configs/muse_custom/probe_glimmer_tp2_rB.py
and benchmark both through the gateway with scripts/run_v5p_2xtp2_gateway_bench.py.
"""
from pathlib import Path

BASE = '''import json
from copy import deepcopy
from benchmark.tpu.recipe.configs.base import _FP8_BASE_CONFIG
CONFIG = deepcopy(_FP8_BASE_CONFIG)
CONFIG["server_args"].update({
    "model_path": "/dev/shm/models/Muse-Glimmer-30B", "served_model_name": "meta-models/Muse-Glimmer-30B",
    "json_model_override_args": json.dumps({"language_model_only": True}),
    "tp_size": 2, "quantization": "unquant", "kv_cache_dtype": "bfloat16",
    "context_length": 131072, "max_running_requests": 64, "torch_compile_max_bs": 64,
    "tpu_compile_bs": [1, 2, 4, 8, 16, 32, 64], "tpu_compile_tokens": [512, 2048, 8192],
    "chunked_prefill_size": 8192, "max_prefill_tokens": 8192, "page_size": 128,
    "max_total_tokens": 524288, "mem_fraction_static": 0.85, "watchdog_timeout": 3600, "disable_radix_cache": True,
    "port": PORT, "base_gpu_id": BASE_ID,
})
CONFIG["runtime_env"].update({"FORCE_SINGLE_HOST": "1", "SGLANG_TPU_PD_COLOCATED": "1",
                              "TORCH_TPU_TIER2_COMPILATION_CACHE": "replica_BASE_ID"})
'''
d = Path("/workspace/sglang/benchmark/tpu/recipe/configs/muse_custom")
d.mkdir(parents=True, exist_ok=True)
for name, port, base in (("probe_glimmer_tp2_rA", 30000, 0), ("probe_glimmer_tp2_rB", 30001, 2)):
    (d / f"{name}.py").write_text(BASE.replace("PORT", str(port)).replace("BASE_ID", str(base)))
    print("wrote", d / f"{name}.py")
