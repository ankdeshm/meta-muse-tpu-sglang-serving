#!/usr/bin/env python3
"""Collects every Muse sweep summary.json into one summary (JSON + Markdown).

Usage:
  python3 collect_muse_results.py <out_dir> [--bucket gs://YOUR_GCS_BUCKET/benchmark-results] [--prefix muse_]
  python3 collect_muse_results.py <out_dir> --local <mirror_dir>   # read a local copy of the run folders

Per (hardware, variant) the run with the most valid cells is reported. Cells measured in a separate sweep on the same
server (bench_<variant>_i<in>_o<out>_c<conc>.jsonl written after the main sweep) are merged in and marked with a dagger.
Cells listed in NOT_REPORTED are shown as "-" with a one-line limitation. Speculative-decoding accept length is the
server's running average since start-up; a per-cell estimate is derived by differencing it with each cell's output
tokens and is shown in parentheses. Set MUSE_SANITIZE_PAIRS="name=PLACEHOLDER,..." to replace extra literal names.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess

DEFAULT_BUCKET = os.environ.get("MUSE_RESULTS_BUCKET", "gs://YOUR_GCS_BUCKET/benchmark-results")

SECTION_TITLES = {
    "v7x_spark": "Muse-Spark-Open-816B-A42B (NVFP4 + DFlash2) — TPU `v7x` (1 host = 4 chips / 8 TensorCores)",
    "v5p32_spark": "Muse-Spark-Open-816B-A42B (BF16 checkpoint, MoE experts quantized to FP8 at load, + DFlash2) — "
                   "TPU `v5p-32` (8 hosts = 32 chips, `2x4x4`)",
    "v7x": "Muse-Glimmer-30B (+ DFlash assistant) — TPU `v7x` (1 host = 4 chips / 8 TensorCores)",
    "v5p": "Muse-Glimmer-30B (+ DFlash assistant) — TPU `v5p` (1 host = 4 chips)",
}
SECTION_ORDER = ["v7x_spark", "v5p32_spark", "v7x", "v5p"]

KEEP_METRICS = ("completed", "duration", "request_throughput", "input_throughput", "output_throughput",
                "total_throughput", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "median_tpot_ms",
                "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "p99_itl_ms", "mean_e2e_latency_ms",
                "median_e2e_latency_ms", "accept_length", "spec_accept_length", "total_input_tokens",
                "total_output_tokens", "max_concurrency")
DROP_RECORD_FIELDS = ("attempts", "server_log_tail", "failure_reason", "note")
DROP_CELL_FIELDS = ("error_tail",)

FP8_EXPERTS = ("MoE experts quantized to FP8 at load (TPU Pallas FP8 fused-RS MoE backend, 25.1 GB of weights per chip); "
               "attention, embeddings and KV cache in BF16")
DESC_OVERRIDES = {
    ("v5p32_spark", "spark_bf16_v5p32_tp32_ep32"): f"BF16 checkpoint, {FP8_EXPERTS}; TP32 EP32 (8 whole experts/chip "
        "across 32 v5p chips), `262K` context",
    ("v5p32_spark", "spark_bf16_v5p32_tp32_ep32_dflash8"): f"BF16 checkpoint, {FP8_EXPERTS}; TP32 EP32 + DFLASH "
        "speculative decoding (`RadixArk/Muse-Spark-Open-DFlash2`, block 8)",
}
DESC_REPLACE = [("model-default sampling (T=1.0, top-p 0.95, top-k 64)", "greedy decoding (temperature 0)"),
                ("model-default sampling", "greedy decoding (temperature 0)")]

# Configurations that are not supported, or not measured, in this build: (status, one-line limitation).
LIMITATIONS = {
    ("v5p32_spark", "spark_bf16_v5p32_tp32_ep1"): ("NOT MEASURED", "TP32 with expert-TP only (EP1) was not measured: "
        "start-up exceeds the 2-hour budget (weight loading over GCSFuse takes 1.5-1.75 h per host)."),
    ("v5p32_spark", "spark_bf16_v5p32_tp32_ep8"): ("NOT SUPPORTED", "Muse Spark MoE supports EP or expert-TP, not both: "
        "`ep_size` must be 1 or equal to `tp_size`."),
    ("v5p", "glimmer_bf16_tp1_dp4"): ("NOT SUPPORTED", "`--dp-size` replicas inside one server are not supported for "
        "Glimmer in this build; colocated replicas behind the gateway are (`*_gateway` variants)."),
    ("v5p", "glimmer_bf16_tp2_dp2"): ("NOT SUPPORTED", "`--dp-size` replicas inside one server are not supported for "
        "Glimmer in this build; colocated replicas behind the gateway are (`*_gateway` variants)."),
    ("v5p", "glimmer_bf16_tp4_dpattn2"): ("NOT SUPPORTED", "DP-attention is not supported for Glimmer in this build."),
    ("v5p", "glimmer_bf16_tp4_dpattn4"): ("NOT SUPPORTED", "DP-attention is not supported for Glimmer in this build."),
    ("v5p", "glimmer_fp8_tp4"): ("NOT SUPPORTED", "FP8 W8A8 needs KV heads >= TP (Glimmer has 2 KV heads), so FP8 runs "
        "use TP2."),
}
# Cells that are not reported: (hardware, variant, input_len, concurrency) -> one-line limitation.
_RADIX = ("With the radix cache on, long-context cells reuse the warm-up prompt, so they are not reported; see "
          "`spark_nvfp4_tp8_ep1` (radix cache off).")
NOT_REPORTED = {
    ("v7x", "glimmer_bf16_tp8_dflash16", 1024, 32): "DFLASH-16 at 1K / c=32 did not produce valid output in this "
        "configuration and is not reported.",
    **{("v7x_spark", "spark_nvfp4_tp8_ep1_bs8_recipe", i, c): _RADIX
       for i, c in ((65536, 1), (65536, 4), (131072, 1), (258048, 1))},
}
NOTES = {
    ("v7x", "glimmer_bf16_tp8_dflash16"): "reported for 1K conc 1-16 and 4K conc 1-16; long-context DFLASH-16 runs use "
        "`--swa-full-tokens-ratio 1.0` (`glimmer_bf16_tp8_dflash16_longctx`)",
    ("v7x_spark", "spark_nvfp4_tp8_ep1_dflash8_megakernel"): "the megakernel needs a BF16 KV cache and at most 4 running "
        "requests; above c=4 requests queue (TTFT grows) and throughput plateaus at ~1,000 tok/s",
    ("v7x_spark", "spark_nvfp4_tp8_ep1_bs8_recipe"): "upstream recipe settings (`max_running_requests=8`); above c=8 "
        "requests queue (TTFT grows) and throughput plateaus at ~380 tok/s",
    ("v5p", "glimmer_bf16_tp4_dflash16_greedy"): "repeat of `glimmer_bf16_tp4_dflash16` with an explicit "
        "`--temperature 0` (1K conc 1-16)",
}

METHODOLOGY = """## Methodology notes

- Workload: `python -m sglang.benchmark.serving` with fixed-length synthetic prompts (`--random-range-ratio 1.0`); every request decodes exactly the stated output length (EOS ignored). Cells with 1K/4K prompts use 4 × concurrency requests (8 at c=1); cells with prompts of 32K tokens or more use 2-16 requests (2-4 at c=1), so treat them as indicative.
- Speculative-decoding (`dflash*`) rows use greedy decoding (temperature 0) on synthetic prompts, which favours draft acceptance. One sampled data point is included: `glimmer_bf16_tp4_dflash16_t1` (T=1.0, top-p 0.95).
- Spec accept len is the server's running average since start-up; per-cell estimates (from differencing the running average with each cell's output tokens) are shown in parentheses, capped at the verification width.
- Gateway (multi-replica) peaks are measured at c=128 and single servers up to c=64; compare at c=64.
- The `v5p-32` Spark runs load the BF16 checkpoint with the MoE experts quantized to FP8 at load; attention, embeddings and the KV cache are BF16. On `v7x` the NVFP4 checkpoint is served as stored.
- The megakernel comparison baseline is `spark_nvfp4_tp8_ep1_dflash8` (DFLASH-8 with an FP8 KV cache).
- No accuracy benchmark was run; each server was spot-checked with one greedy chat prompt.
"""


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def fmt(v, nd=1) -> str:
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def sanitize_text(s: str) -> str:
    for pair in filter(None, os.environ.get("MUSE_SANITIZE_PAIRS", "").split(",")):
        name, _, placeholder = pair.partition("=")
        if name:
            s = s.replace(name, placeholder)
    s = re.sub(r"gs://(?!YOUR_GCS_BUCKET)[A-Za-z0-9._-]+", "gs://YOUR_GCS_BUCKET", s)
    s = re.sub(r"(docker\.pkg\.dev/)(?!YOUR_GCP_PROJECT)[A-Za-z0-9-]+/", r"\1YOUR_GCP_PROJECT/", s)
    return re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_REDACTED", s)


class Source:
    """Run folders from GCS or from a local mirror."""

    def __init__(self, bucket: str, local: str | None):
        self.bucket, self.local = bucket.rstrip("/"), (Path(local) if local else None)

    def run_dirs(self, prefix: str) -> list[str]:
        if self.local:
            names = [p.name for p in self.local.iterdir() if p.is_dir()]
        else:
            names = [l.strip().rstrip("/").split("/")[-1] for l in sh(["gcloud", "storage", "ls", f"{self.bucket}/"]).splitlines()]
        return sorted((n for n in names if n.startswith(prefix)), key=lambda n: n.rsplit("_", 1)[-1])

    def read(self, run: str, name: str) -> str | None:
        try:
            if self.local:
                p = self.local / run / name
                return p.read_text() if p.exists() else None
            return sh(["gcloud", "storage", "cat", f"{self.bucket}/{run}/{name}"])
        except Exception:
            return None


def ok_cell(c: dict) -> bool:
    return c.get("return_code") == 0 and bool((c.get("metrics") or {}).get("output_throughput"))


def valid_cell_count(v: dict) -> int:
    return sum(1 for c in v.get("cells", []) if ok_cell(c))


def merge_separate_sweep(src: Source, run: str, rec: dict) -> None:
    """Merge cells (and the sanity reply) measured in a separate sweep on the same server."""
    name = rec.get("name", "")
    for c in rec.get("cells", []):
        if ok_cell(c):
            continue
        raw = src.read(run, f"bench_{name}_i{c.get('input_len')}_o{c.get('output_len')}_c{c.get('concurrency')}.jsonl")
        lines = [l for l in (raw or "").splitlines() if l.strip()]
        try:
            full = json.loads(lines[-1]) if lines else {}
        except Exception:
            full = {}
        m = {k: full[k] for k in KEEP_METRICS if k in full}
        if m.get("output_throughput"):
            c.update(metrics=m, return_code=0, separate_sweep=True)
    if not (rec.get("sanity") or {}).get("ok"):
        for cand in (f"backfill_sanity_{name}.json", "backfill_sanity_tp32_ep32.json" if name.endswith("tp32_ep32") else ""):
            raw = src.read(run, cand) if cand else None
            try:
                sj = json.loads(raw) if raw else None
            except Exception:
                sj = None
            if sj and sj.get("ok"):
                rec["sanity"] = sj
                break


def estimate_cell_accept(rec: dict, width: int | None) -> None:
    """Per-cell accept length from the server's running average (cells are stored in execution order)."""
    cum_tokens = cum_steps = 0.0
    for i, c in enumerate(rec.get("cells", [])):
        m = c.get("metrics") or {}
        a, t = m.get("accept_length") or m.get("spec_accept_length"), m.get("total_output_tokens")
        if c.get("separate_sweep") or not ok_cell(c) or not a or not t:
            if a or not ok_cell(c):
                break
            continue
        total_steps = (cum_tokens + t) / a
        steps = total_steps - cum_steps
        if steps <= 0:
            break
        est = t / steps if i else a
        if width and est > width:
            est, c["accept_len_cell_est_capped"] = float(width), True
        c["accept_len_cell_est"] = round(est, 2)
        cum_tokens, cum_steps = cum_tokens + t, total_steps


def chips_used(hw: str, sa: dict, pvar: dict, v: dict) -> float:
    explicit = v.get("chips_used") or pvar.get("chips_used")
    if explicit:
        return float(explicit)
    replicas = int(v.get("replicas") or pvar.get("replicas") or sa.get("dp_size") or 1)
    tp = int(sa.get("tp_size", 1))
    devices = tp if sa.get("enable_dp_attention") else tp * replicas
    return devices / (2 if "v7x" in hw else 1)


def acc_cell(c: dict, after_unreported: bool) -> str:
    m = c.get("metrics") or {}
    a = m.get("accept_length", m.get("spec_accept_length"))
    if a is None:
        return "-"
    est = c.get("accept_len_cell_est")
    running = "—" if after_unreported else fmt(a, 2)
    if est is None or (abs(est - a) < 0.05 and not after_unreported):
        return running
    return f"{running} (cell ≈{est:.1f}{' = max' if c.get('accept_len_cell_est_capped') else ''})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default="muse_")
    ap.add_argument("--local", help="read run folders from this local mirror instead of GCS")
    args = ap.parse_args()

    bucket = args.bucket.rstrip("/")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = Source(bucket, args.local)
    runs = []
    for d in src.run_dirs(args.prefix):
        raw = src.read(d, "summary.json")
        try:
            runs.append((d, json.loads(raw))) if raw else None
        except Exception:
            continue

    latest: dict[tuple, dict] = {}
    for d, s in runs:
        hw = (s.get("hardware") or {}).get("name", "?")
        plan = s.get("plan") or {}
        for v in s.get("variants", []):
            rec = json.loads(json.dumps(v))
            merge_separate_sweep(src, d, rec)
            pvar = next((x for x in plan.get("variants", []) if x.get("name") == v["name"]), {})
            sa = {**plan.get("server_args", {}), **pvar.get("server_args", {})}
            key = (hw, v["name"])
            for old, new in DESC_REPLACE:
                if old in (rec.get("desc") or "") and "--temperature" not in json.dumps(pvar.get("bench_extra") or []):
                    rec["desc"] = rec["desc"].replace(old, new)
            rec["desc"] = DESC_OVERRIDES.get(key, rec.get("desc", ""))
            rec.update(source=f"{bucket}/{d}/", hardware=hw, hardware_desc=(s.get("hardware") or {}).get("desc", ""),
                       chips=(s.get("hardware") or {}).get("chips", 4), model_repo=s.get("model_repo"),
                       image=s.get("image"), chips_used=chips_used(hw, sa, pvar, v), server_args_effective=sa)
            width = sa.get("speculative_num_draft_tokens")
            estimate_cell_accept(rec, int(width) if width else None)
            prev = latest.get(key)
            if prev is None or (valid_cell_count(rec) >= valid_cell_count(prev)
                                and (valid_cell_count(rec) > 0 or valid_cell_count(prev) == 0)):
                latest[key] = rec

    for key, rec in latest.items():
        for f in DROP_RECORD_FIELDS:
            rec.pop(f, None)
        if key in NOTES:
            rec["notes"] = NOTES[key]
        if key in LIMITATIONS and not valid_cell_count(rec):
            rec["status"], rec["limitation"] = LIMITATIONS[key]
        elif valid_cell_count(rec):
            rec["status"] = "DONE"
        unreported = False
        for c in rec.get("cells", []):
            for f in DROP_CELL_FIELDS:
                c.pop(f, None)
            lim = NOT_REPORTED.get((key[0], key[1], c.get("input_len"), c.get("concurrency")))
            if lim:
                c.update(metrics=None, reported=False, limitation=lim)
                c.pop("accept_len_cell_est", None)
                unreported = unreported or key[1] == "glimmer_bf16_tp8_dflash16"
            elif unreported:
                c["running_accept_includes_unreported_cell"] = True

    def sort_key(r: dict):
        return (SECTION_ORDER.index(r["hardware"]) if r["hardware"] in SECTION_ORDER else 99, r["hardware"], r["name"])

    merged = sorted(latest.values(), key=sort_key)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    (out / "summary.json").write_text(sanitize_text(json.dumps(
        {"generated_at_utc": now, "runs": [d for d, _ in runs], "records": merged}, indent=2)) + "\n")

    md = ["# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p`) — Measured "
          "SGLang-TorchTPU Results", "",
          f"- **Generated (UTC):** `{now}`",
          f"- **Source runs:** {len(runs)} result directories",
          "- **`Output tok/s/chip`** divides output throughput by the physical TPU chips occupied by that serving topology "
          "(`TP2` on `v5p` = 2 chips; `TP2` on `v7x` = 1 chip; `2×TP2` gateway on `v5p` = 4 chips; `4×TP2` / `2×TP4` / "
          "`TP8` on `v7x` = 4 chips; `TP32` on `v5p-32` = 32 chips).", "", METHODOLOGY]
    hws = []
    for r in merged:
        if r["hardware"] not in hws:
            hws.append(r["hardware"])
    for hw in hws:
        recs = [r for r in merged if r["hardware"] == hw]
        md += [f"## {SECTION_TITLES.get(hw, hw)}", "", recs[0]["hardware_desc"], "",
               "| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip "
               "| Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |",
               "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
        footnotes, dagger = [], False
        for r in recs:
            chips = r.get("chips_used") or r["chips"]
            if not r.get("cells"):
                md.append(f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | - | - | - | - | - | - | - | - | - | "
                          f"`{r['status']}` |")
                continue
            for c in r["cells"]:
                if c.get("reported") is False:
                    md.append(f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | `{c['input_len']}` / `{c['output_len']}` | "
                              f"{c['concurrency']} | — | — | — | — | — | — | — | `NOT REPORTED` |")
                    note = f"- `{r['name']}` `{c['input_len']}` / `{c['output_len']}` c={c['concurrency']}: {c['limitation']}"
                    if note not in footnotes:
                        footnotes.append(note)
                    continue
                m = c.get("metrics") or {}
                otps = m.get("output_throughput")
                st = ("PASS†" if c.get("separate_sweep") else "PASS") if ok_cell(c) else f"FAIL(rc={c.get('return_code')})"
                dagger = dagger or bool(c.get("separate_sweep"))
                md.append(f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | `{c['input_len']}` / `{c['output_len']}` | "
                          f"{c['concurrency']} | {fmt(m.get('request_throughput'), 3)} | {fmt(otps)} | "
                          f"{fmt(otps / chips if otps else None)} | {fmt(m.get('total_throughput'))} | "
                          f"{fmt(m.get('median_ttft_ms'), 0)} / {fmt(m.get('p99_ttft_ms'), 0)} | "
                          f"{fmt(m.get('median_tpot_ms'), 2)} / {fmt(m.get('p99_tpot_ms'), 2)} | "
                          f"{acc_cell(c, bool(c.get('running_accept_includes_unreported_cell')))} | `{st}` |")
        md.append("")
        if dagger:
            md += ["† Measured in a separate sweep on the same server.", ""]
        if footnotes:
            md += ["Not reported:", ""] + footnotes + [""]
        md += ["### Variant status & sanity check", ""]
        for r in recs:
            if r.get("limitation"):
                md.append(f"- `{r['name']}`: **{r['status']}** — {r['limitation']}")
                continue
            line = f"- `{r['name']}`: **{r['status']}** (server ready after `{r.get('server_ready_s', '-')}` s)"
            if r.get("notes"):
                line += f" — _{r['notes']}_"
            sn = r.get("sanity") or {}
            if sn.get("ok"):
                txt = (sn.get("content") or "").strip() or (sn.get("reasoning_content") or "").strip()
                line += f"\n  - sanity (greedy chat): `{txt.replace(chr(10), ' ')[-200:]}`"
            md.append(line)
        md.append("")

    (out / "SUMMARY.md").write_text(sanitize_text("\n".join(md)) + "\n")
    print(f"wrote {out/'summary.json'} and {out/'SUMMARY.md'} ({len(merged)} variant records from {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
