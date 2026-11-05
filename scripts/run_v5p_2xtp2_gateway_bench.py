#!/usr/bin/env python3
"""Benchmarks two colocated TP2 Muse-Glimmer-30B servers on one TPU v5p host through a least-inflight gateway.

Prerequisite: both replicas from scripts/v5p_tp2_replica_recipes.py are serving on ports 30000 and 30001.
The script starts scripts/muse_multi_replica_proxy.py on port 29999 (override the path with MUSE_PROXY), runs a chat
sanity check, sweeps 1K / 4K / 32K / 64K / ~128K prompts with 1024 output tokens, and writes summary.json, SUMMARY.md
and one bench_glimmer_bf16_2xtp2_gateway_i<in>_o<out>_c<conc>.jsonl per cell under
/models/benchmark-results/muse_glimmer30b_sglang_v5p_1host_2xtp2gw_<UTC timestamp>/.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}][v5p-2xtp2-gw] {msg}", flush=True)


def wait_ready(port: int, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health_generate", timeout=30) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def sanity_generate(port: int, served_name: str) -> dict:
    body = {
        "model": served_name,
        "messages": [{"role": "user", "content": "In one short sentence: what is the capital of France?"}],
        "max_tokens": 256,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read().decode())
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        return {
            "ok": True,
            "latency_s": round(time.time() - t0, 2),
            "content": content[:600],
            "content_tail": content[-300:],
            "reasoning_content": (msg.get("reasoning_content") or "")[:600],
            "usage": out.get("usage"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}


def run_cell(port: int, tokenizer: str, served_name: str, inp: int, out: int, conc: int, out_file: Path) -> dict:
    nprompts = max(4 * conc, 8) if inp <= 16384 else max(2 * conc, 4)
    cmd = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://127.0.0.1:{port}",
        "--dataset-name",
        "random",
        "--random-input-len",
        str(inp),
        "--random-output-len",
        str(out),
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        str(nprompts),
        "--max-concurrency",
        str(conc),
        "--tokenizer",
        tokenizer,
        "--output-file",
        str(out_file),
        "--warmup-requests",
        "2",
        "--model",
        served_name,
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    row = {
        "input_len": inp,
        "output_len": out,
        "concurrency": conc,
        "num_prompts": nprompts,
        "return_code": proc.returncode,
        "wall_s": round(time.time() - t0, 1),
    }
    metrics = {}
    if out_file.exists():
        lines = [l for l in out_file.read_text().splitlines() if l.strip()]
        if lines:
            try:
                metrics = json.loads(lines[-1])
            except Exception:
                metrics = {}
    keep = (
        "completed",
        "duration",
        "request_throughput",
        "input_throughput",
        "output_throughput",
        "total_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p99_itl_ms",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "accept_length",
        "spec_accept_length",
        "total_input_tokens",
        "total_output_tokens",
        "max_concurrency",
    )
    row["metrics"] = {k: metrics[k] for k in keep if k in metrics}
    if proc.returncode != 0 or not row["metrics"]:
        row["error_tail"] = (proc.stdout[-1500:] + "\n" + proc.stderr[-2500:]).strip()
    return row


def fmt(v, nd=1) -> str:
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def write_summary(out_dir: Path, meta: dict, variants: list[dict]) -> None:
    chips = 4
    payload = {"updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **meta, "variants": variants}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    md = [
        f"# {meta['title']}",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Hardware:** {meta['hardware']['desc']}; engine `SGLang-TorchTPU` (`{meta['image']}`), 2 independent TP2 servers (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) fronted by async least-inflight HTTP gateway.",
        "",
        "| Variant | Layout / precision | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip | Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for v in variants:
        for c in v.get("cells", []):
            m = c.get("metrics", {})
            otps = m.get("output_throughput")
            st = "PASS" if m and c["return_code"] == 0 else f"FAIL(rc={c['return_code']})"
            md.append(
                f"| `{v['name']}` | {v['desc']} | `{c['input_len']}` / `{c['output_len']}` | {c['concurrency']} | "
                f"{fmt(m.get('request_throughput'), 3)} | {fmt(otps)} | {fmt(otps / chips if otps else None)} | "
                f"{fmt(m.get('total_throughput'))} | {fmt(m.get('median_ttft_ms'), 0)} / {fmt(m.get('p99_ttft_ms'), 0)} | "
                f"{fmt(m.get('median_tpot_ms'), 2)} / {fmt(m.get('p99_tpot_ms'), 2)} | `{st}` |"
            )
    (out_dir / "SUMMARY.md").write_text("\n".join(md) + "\n")


def main() -> int:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("/models/benchmark-results") / f"muse_glimmer30b_sglang_v5p_1host_2xtp2gw_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gw_port = 29999
    proxy_proc = subprocess.Popen(
        [
            sys.executable,
            os.environ.get("MUSE_PROXY", str(Path(__file__).with_name("muse_multi_replica_proxy.py"))),
            "--port",
            str(gw_port),
            "--backends",
            "http://127.0.0.1:30000,http://127.0.0.1:30001",
        ],
        stdout=open(out_dir / "gateway.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    meta = {
        "title": "Meta Muse-Glimmer-30B on one TPU v5p host (4 chips) via 2xTP2 Colocated Replicas + Least-Inflight Gateway",
        "model_repo": "meta-models/Muse-Glimmer-30B",
        "model_gb": 59.6,
        "image": "us-central1-docker.pkg.dev/YOUR_GCP_PROJECT/vllm/sglang-torchtpu-diffusion:v0.2.0",
        "hardware": {
            "name": "v5p",
            "chips": 4,
            "desc": "one TPU `v5p` host (`4` chips, 1 JAX/PJRT device per chip, `95 GB` HBM/chip = `380 GB`), a single host of the `v5p-32` (`2x4x4`) slice used as a standalone 2x2x1 sub-slice",
        },
        "plan": {
            "server_args": {"tp_size": 2, "dp_size": 2},
            "variants": [
                {
                    "name": "glimmer_bf16_2xtp2_gateway",
                    "server_args": {"tp_size": 2, "dp_size": 2},
                }
            ],
        },
        "node": os.environ.get("HOSTNAME", ""),
    }
    rec = {
        "name": "glimmer_bf16_2xtp2_gateway",
        "desc": "BF16, 2 independent TP2 servers on 1 host (`SGLANG_TPU_PD_COLOCATED=1`, `base_gpu_id=0,2`) behind async least-inflight HTTP gateway, BF16 KV",
        "status": "RUNNING",
        "server_ready_s": 1.0,
        "cells": [],
    }
    variants = [rec]
    try:
        if not wait_ready(gw_port, 60.0):
            rec["status"] = "SERVER_FAILED"
            write_summary(out_dir, meta, variants)
            return 1
        rec["sanity"] = sanity_generate(gw_port, "meta-models/Muse-Glimmer-30B")
        log(f"sanity: {json.dumps(rec['sanity'])[:400]}")
        write_summary(out_dir, meta, variants)

        workloads = [
            (1024, 1024, [1, 4, 16, 32, 64, 128]),
            (4096, 1024, [1, 4, 16, 32, 64]),
            (32768, 1024, [1, 4, 8]),
            (65536, 1024, [1, 4]),
            (126976, 1024, [1, 2]),
        ]
        for inp, out, concs in workloads:
            for conc in concs:
                cell_file = out_dir / f"bench_glimmer_bf16_2xtp2_gateway_i{inp}_o{out}_c{conc}.jsonl"
                log(f"bench in={inp} out={out} conc={conc}")
                cell = run_cell(
                    gw_port,
                    "/dev/shm/models/Muse-Glimmer-30B",
                    "meta-models/Muse-Glimmer-30B",
                    inp,
                    out,
                    conc,
                    cell_file,
                )
                rec["cells"].append(cell)
                m = cell.get("metrics", {})
                log(
                    f"-> rc={cell['return_code']} out_tok/s={m.get('output_throughput')} "
                    f"ttft_p50={m.get('median_ttft_ms')} tpot_p50={m.get('median_tpot_ms')}"
                )
                write_summary(out_dir, meta, variants)
        rec["status"] = "DONE"
        write_summary(out_dir, meta, variants)
        log(f"DONE -> {out_dir}")
    finally:
        try:
            os.killpg(proxy_proc.pid, signal.SIGTERM)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
