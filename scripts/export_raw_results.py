#!/usr/bin/env python3
"""Exports the per-run benchmark artifacts behind the published Muse tables into results/raw/.

For every run directory that contains at least one successful benchmark cell, this copies:
  - bench_<variant>_i<in>_o<out>_c<conc>.jsonl  raw SGLang `bench_serving` output, one file per cell
  - summary.json                                harness record for the run (plan, server args, per-cell
                                                metrics, chat sanity check); server log tails are omitted,
                                                as are variant records without results when that variant
                                                was measured in another exported run
  - backfill_sanity_<variant>.json              chat sanity reply recorded after the sweep, where present
  - server_log_excerpt_<variant>.txt            short excerpt of the rank-0 server log (MoE backend,
                                                weight memory, KV cache allocation, token budget) and,
                                                for multi-host servers, every host's weight-loading line
and writes results/raw/index.csv with one row per measured cell, pointing at its raw file.

Identifiers are replaced with placeholders (gs://YOUR_GCS_BUCKET, YOUR_GCP_PROJECT, tpu-worker-node).
Extra site-specific replacements can be supplied with --sanitize-map (JSON list of [regex, replacement]);
keep that file out of version control.

Usage:
  python3 export_raw_results.py <local_mirror_of_benchmark_results> <out_dir> [--prefix muse_]
          [--sanitize-map map.json]

The local mirror can be created with, for example:
  gcloud storage rsync -r gs://YOUR_GCS_BUCKET/benchmark-results/<run_id>/ <mirror>/<run_id>/
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil

GENERIC_RULES = [
    (r"hf_[A-Za-z0-9]{16,}", "hf_REDACTED"),
    (r"gs://(?!YOUR_GCS_BUCKET/)[a-z0-9][a-z0-9._-]{2,62}/", "gs://YOUR_GCS_BUCKET/"),
    (r"([a-z0-9-]+-docker\.pkg\.dev)/(?!YOUR_GCP_PROJECT/)[a-z][a-z0-9-]{4,28}[a-z0-9]/", r"\1/YOUR_GCP_PROJECT/"),
    (r"\bgke-[a-z0-9-]*-[0-9a-f]{8}-[a-z0-9]{4}\b", "tpu-worker-node"),
    (r"\bgke-tpu-[0-9a-f]{8}-[a-z0-9]{4}\b", "tpu-worker-node"),
]
DROP_TOP_KEYS = ("node",)
DROP_VARIANT_KEYS = ("server_log_tail", "note")
DROP_CELL_KEYS = ("error_tail",)
EXCERPT_PATTERNS = (
    r"Using TPU Pallas .*MoE backend",
    r"Load weight end",
    r"KV Cache is allocated",
    r"max_total_num_tokens=",
    r"[Mm]egakernel (prepared|enabled|layers)",
)
BENCH_RE = re.compile(r"^bench_(?P<variant>.+)_i(?P<i>\d+)_o(?P<o>\d+)_c(?P<c>\d+)\.jsonl$")
GREEDY_RE = re.compile(r"model-default sampling(\s*\([^)]*\))?")


class Sanitizer:
    def __init__(self, extra: list[list[str]] | None = None):
        rules = list(GENERIC_RULES) + [tuple(r) for r in (extra or [])]
        self.rules = [(re.compile(p), r) for p, r in rules]

    def __call__(self, text: str) -> str:
        for rx, rep in self.rules:
            text = rx.sub(rep, text)
        return text


def last_json_line(path: str) -> dict | None:
    lines = [ln for ln in open(path, errors="replace") if ln.strip()]
    if not lines:
        return None
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError:
        return None


def variant_config(summary: dict, v: dict, vname: str) -> dict:
    """Serving configuration for one variant, derived from the plan embedded in the run summary."""
    hw = (summary.get("hardware") or {}).get("name", "")
    plan = summary.get("plan") or {}
    pv = next((x for x in plan.get("variants", []) if x.get("name") == vname), {})
    sa = {**plan.get("server_args", {}), **pv.get("server_args", {})}
    env = {**(plan.get("runtime_env") or {}), **(pv.get("runtime_env") or {})}
    replicas = int(v.get("replicas") or pv.get("replicas") or sa.get("dp_size", 1) or 1)
    tp = int(sa.get("tp_size", 1))
    devices = tp if sa.get("enable_dp_attention") else tp * replicas
    explicit = v.get("chips_used") or pv.get("chips_used")
    chips = float(explicit) if explicit else devices / (2 if "v7x" in hw else 1)
    quant = sa.get("quantization")
    if quant == "modelopt_fp4":
        weights = "NVFP4"
    elif quant == "fp8":
        weights = "FP8 (W8A8)"
    elif "spark" in hw:
        weights = "BF16 checkpoint, MoE experts FP8 at load"
    else:
        weights = "BF16"
    kv = {"fp8_e4m3": "FP8 (e4m3)", "bfloat16": "BF16"}.get(sa.get("kv_cache_dtype") or "", "BF16")
    spec = ""
    if sa.get("speculative_algorithm"):
        spec = f"{sa['speculative_algorithm']}-{sa.get('speculative_num_draft_tokens', '')}".rstrip("-")
    if str(env.get("SGLANG_TPU_MUSE_MEGAKERNEL", "")) == "1":
        spec = (spec + " + decode megakernel").strip(" +")
    return {
        "chips_used": int(chips) if chips == int(chips) else chips,
        "weights": weights,
        "kv_cache": kv,
        "tp": tp,
        "ep": sa.get("ep_size", ""),
        "replicas": replicas,
        "spec_decode": spec,
        "max_running_requests": sa.get("max_running_requests", ""),
    }


def excerpt(log_path: str, sanitize: Sanitizer, max_lines: int = 40) -> str:
    pats = [re.compile(p) for p in EXCERPT_PATTERNS]
    seen, out = set(), []
    with open(log_path, errors="replace") as f:
        for line in f:
            if not any(p.search(line) for p in pats):
                continue
            # rank-0 (or untagged) lines only
            tag = re.match(r"\[[^\]]*?(TP\d+)", line)
            if tag and tag.group(1) != "TP0":
                continue
            body = re.sub(r"^\[[^\]]*\]\s*", "", line.strip())
            if body in seen:
                continue
            seen.add(body)
            out.append(sanitize(line.rstrip()))
            if len(out) >= max_lines:
                break
    return "\n".join(out) + ("\n" if out else "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mirror")
    ap.add_argument("out_dir")
    ap.add_argument("--prefix", default="muse_")
    ap.add_argument("--sanitize-map", default=None)
    args = ap.parse_args()

    extra = json.load(open(args.sanitize_map)) if args.sanitize_map else None
    sanitize = Sanitizer(extra)
    os.makedirs(args.out_dir, exist_ok=True)

    runs = sorted(
        (d for d in glob.glob(os.path.join(args.mirror, f"{args.prefix}*")) if os.path.isdir(d)),
        key=lambda p: p.rstrip("/").rsplit("_", 1)[-1],
    )
    # Pass 1: runs with at least one successful cell, and the variants measured anywhere.
    selected, measured_anywhere = [], set()
    for run_dir in runs:
        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            continue
        summary = json.load(open(summary_path))
        good = []
        for bf in sorted(glob.glob(os.path.join(run_dir, "bench_*.jsonl"))):
            rec = last_json_line(bf)
            m = BENCH_RE.match(os.path.basename(bf))
            if rec and rec.get("output_throughput"):
                good.append((bf, rec))
                if m:
                    measured_anywhere.add(((summary.get("hardware") or {}).get("name", ""), m.group("variant")))
        if good:
            selected.append((run_dir, summary, good))

    rows = []
    for run_dir, summary, good in selected:
        run = os.path.basename(run_dir.rstrip("/"))
        dest = os.path.join(args.out_dir, run)
        os.makedirs(dest, exist_ok=True)
        hw = summary.get("hardware") or {}
        measured_here = {BENCH_RE.match(os.path.basename(bf)).group("variant")
                         for bf, _ in good if BENCH_RE.match(os.path.basename(bf))}
        for k in DROP_TOP_KEYS:
            summary.pop(k, None)
        # Variant records without results are kept only if the variant was never measured in any exported run.
        summary["variants"] = [
            v for v in summary.get("variants", [])
            if v["name"] in measured_here or (hw.get("name", ""), v["name"]) not in measured_anywhere
        ]
        plan_variants = {pv.get("name"): pv for pv in (summary.get("plan") or {}).get("variants", [])}
        for v in summary["variants"]:
            for k in DROP_VARIANT_KEYS:
                v.pop(k, None)
            for c in v.get("cells", []):
                for k in DROP_CELL_KEYS:
                    c.pop(k, None)
        # No sampling arguments were passed to these variants, so bench_serving decoded greedily.
        for rec in [*summary["variants"], *plan_variants.values()]:
            pv = plan_variants.get(rec.get("name"), rec)
            if rec.get("desc") and "--temperature" not in (pv.get("bench_extra") or []):
                rec["desc"] = GREEDY_RE.sub("greedy decoding (temperature 0)", rec["desc"])
        with open(os.path.join(dest, "summary.json"), "w") as f:
            f.write(sanitize(json.dumps(summary, indent=2)) + "\n")
        for extra_file in sorted(glob.glob(os.path.join(run_dir, "backfill_sanity_*.json"))):
            with open(extra_file, errors="replace") as src, open(os.path.join(dest, os.path.basename(extra_file)), "w") as dst:
                dst.write(sanitize(src.read()))

        variants = {v["name"]: v for v in summary["variants"]}
        for bf, rec in good:
            name = os.path.basename(bf)
            m = BENCH_RE.match(name)
            with open(bf, errors="replace") as src, open(os.path.join(dest, name), "w") as dst:
                dst.write(sanitize(src.read()))
            if not m:
                continue
            vname = m.group("variant")
            v = variants.get(vname, {})
            cfg = variant_config(summary, v, vname)
            rows.append({
                "model": (summary.get("model_repo") or "").split(" ")[0],
                "hardware": hw.get("name", ""),
                "variant": vname,
                **cfg,
                "input_len": int(m.group("i")),
                "output_len": int(m.group("o")),
                "concurrency": int(m.group("c")),
                "completed": rec.get("completed"),
                "output_tok_s": round(rec["output_throughput"], 2),
                "total_tok_s": round(rec.get("total_throughput") or 0, 2),
                "ttft_p50_ms": round(rec.get("median_ttft_ms") or 0, 1),
                "tpot_p50_ms": round(rec.get("median_tpot_ms") or 0, 3),
                "accept_length": rec.get("accept_length"),
                "run_id": run,
                "file": f"{run}/{name}",
            })

        for vname in sorted(variants):
            single = os.path.join(run_dir, f"server_{vname}.log")
            workers = sorted(glob.glob(os.path.join(run_dir, f"server_{vname}_w*.log")),
                             key=lambda p: int(re.search(r"_w(\d+)\.log$", p).group(1)))
            text = ""
            if os.path.exists(single):
                text = excerpt(single, sanitize)
            elif workers:
                text = excerpt(workers[0], sanitize)
                loads = []
                for wlog in workers:
                    wid = re.search(r"_w(\d+)\.log$", wlog).group(1)
                    with open(wlog, errors="replace") as f:
                        hit = next((ln.rstrip() for ln in f if "Load weight end" in ln), None)
                    if hit:
                        loads.append(f"[worker {wid}] {sanitize(hit)}")
                if loads:
                    text += "\n# Weight loading, every host of the slice\n" + "\n".join(loads) + "\n"
            if text.strip():
                with open(os.path.join(dest, f"server_log_excerpt_{vname}.txt"), "w") as f:
                    f.write(text)

    rows.sort(key=lambda r: (r["hardware"], r["variant"], r["input_len"], r["concurrency"], r["run_id"]))
    with open(os.path.join(args.out_dir, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        w.writerows(rows)
    print(f"exported {len({r['run_id'] for r in rows})} runs, {len(rows)} cells -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
