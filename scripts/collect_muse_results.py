#!/usr/bin/env python3
"""Collects every Muse sweep summary.json from GCS into one summary (JSON + Markdown).

Usage:
  python3 collect_muse_results.py <out_dir> [--bucket gs://YOUR_GCS_BUCKET/benchmark-results] [--prefix muse_]
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
    "v5p32_spark": "Muse-Spark-Open-816B-A42B (BF16 + DFlash2) — TPU `v5p-32` (8 hosts = 32 chips, `2x4x4`)",
    "v7x": "Muse-Glimmer-30B (+ DFlash assistant) — TPU `v7x` (1 host = 4 chips / 8 TensorCores)",
    "v5p": "Muse-Glimmer-30B (+ DFlash assistant) — TPU `v5p` (1 host = 4 chips)",
}
SECTION_ORDER = ["v7x_spark", "v5p32_spark", "v7x", "v5p"]

KEEP_METRICS = (
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


def sh(cmd: list[str]) -> str:
    env = {**os.environ, "CLOUDSDK_METRICS_ENVIRONMENT": "datacloud.jetski"}
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout


def fmt(v, nd=1) -> str:
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def sanitize_text(s: str) -> str:
    s = s.replace("YOUR_GCP_PROJECT_ID", "YOUR_GCP_PROJECT")
    s = s.replace("YOUR_GCS_BUCKET", "YOUR_GCS_BUCKET")
    s = re.sub(r"hf_[A-Za-z0-9]{16,}", "hf_REDACTED", s)
    return s


def short_reason(rec: dict) -> str:
    tail = rec.get("server_log_tail", "") or ""
    for key in (
        "ValueError:",
        "RuntimeError:",
        "VerificationError:",
        "torch._dynamo.exc.Unsupported",
        "AssertionError",
        "NotImplementedError",
        "TypeError:",
        "KeyError:",
    ):
        for line in tail.splitlines():
            if key in line:
                return line.strip()[:300]
    return rec.get("status", "")


def valid_cell_count(v: dict) -> int:
    return sum(1 for c in v.get("cells", []) if c.get("return_code") == 0 and c.get("metrics"))


def try_backfill_variant(run_dir: str, rec: dict) -> None:
    """Recover cells or sanity JSON written by post-run backfill hooks in the same GCS run directory."""
    vname = rec.get("name", "")
    for c in rec.get("cells", []):
        if c.get("return_code") == 0 and c.get("metrics"):
            continue
        inp, out_len, conc = c.get("input_len"), c.get("output_len"), c.get("concurrency")
        uri = f"{run_dir}bench_{vname}_i{inp}_o{out_len}_c{conc}.jsonl"
        try:
            raw = sh(["gcloud", "storage", "cat", uri])
            lines = [ln for ln in raw.splitlines() if ln.strip()]
            if not lines:
                continue
            full = json.loads(lines[-1])
            extracted = {k: full[k] for k in KEEP_METRICS if k in full}
            if extracted.get("output_throughput"):
                c["metrics"] = extracted
                c["return_code"] = 0
                c.pop("error_tail", None)
                if "duration" in full:
                    c["wall_s"] = round(float(full["duration"]), 1)
        except Exception:
            continue

    sanity = rec.get("sanity") or {}
    if not sanity.get("ok"):
        candidates = [f"{run_dir}backfill_sanity_{vname}.json"]
        if vname == "spark_bf16_v5p32_tp32_ep32":
            candidates.append(f"{run_dir}backfill_sanity_tp32_ep32.json")
        for cand in candidates:
            try:
                sj = json.loads(sh(["gcloud", "storage", "cat", cand]))
                if sj.get("ok"):
                    txt = sj.get("content") or sj.get("reasoning_content") or ""
                    sj["content_tail"] = txt[-240:]
                    rec["sanity"] = sj
                    break
            except Exception:
                continue


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--prefix", default="muse_")
    args = ap.parse_args()

    bucket = args.bucket.rstrip("/")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [
        l.strip()
        for l in sh(["gcloud", "storage", "ls", f"{bucket}/"]).splitlines()
        if l.strip().rstrip("/").split("/")[-1].startswith(args.prefix)
    ]
    runs = []
    for d in sorted(dirs, key=lambda x: x.rstrip("/").rsplit("_", 1)[-1]):
        try:
            s = json.loads(sh(["gcloud", "storage", "cat", f"{d}summary.json"]))
        except Exception:
            continue
        runs.append((d, s))

    latest: dict[tuple, dict] = {}
    history: dict[tuple, list] = {}
    for d, s in runs:
        hw = (s.get("hardware") or {}).get("name", "?")
        for v in s.get("variants", []):
            rec = dict(v)
            try_backfill_variant(d, rec)
            rec["source"] = d
            rec["hardware"] = hw
            rec["hardware_desc"] = (s.get("hardware") or {}).get("desc", "")
            rec["chips"] = (s.get("hardware") or {}).get("chips", 4)
            rec["model_repo"] = s.get("model_repo")
            rec["image"] = s.get("image")
            plan = s.get("plan") or {}
            pvar = next((x for x in plan.get("variants", []) if x.get("name") == v["name"]), {})
            sa = {**plan.get("server_args", {}), **pvar.get("server_args", {})}
            replicas = int(v.get("replicas") or pvar.get("replicas") or sa.get("dp_size", 1))
            tp = int(sa.get("tp_size", 1))
            devices = tp if sa.get("enable_dp_attention") else tp * replicas
            explicit_chips = v.get("chips_used") or pvar.get("chips_used")
            rec["chips_used"] = float(explicit_chips) if explicit_chips else devices / (2 if "v7x" in hw else 1)
            rec["server_args_effective"] = sa
            key = (hw, v["name"])
            history.setdefault(key, []).append(
                {
                    "source": d,
                    "status": rec.get("status"),
                    "reason": short_reason(rec) if not rec.get("cells") else "",
                }
            )
            prev = latest.get(key)
            if prev is None:
                latest[key] = rec
            else:
                new_cnt = valid_cell_count(rec)
                prev_cnt = valid_cell_count(prev)
                if new_cnt >= prev_cnt and (new_cnt > 0 or prev_cnt == 0):
                    latest[key] = rec

    for key, rec in latest.items():
        rec["attempts"] = history.get(key, [])

    def sort_key(r: dict):
        hw_idx = SECTION_ORDER.index(r["hardware"]) if r["hardware"] in SECTION_ORDER else 99
        return (hw_idx, r["hardware"], r["name"])

    merged = sorted(latest.values(), key=sort_key)
    raw_json = json.dumps(
        {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "runs": [d for d, _ in runs],
            "records": merged,
        },
        indent=2,
    )
    (out / "summary.json").write_text(sanitize_text(raw_json) + "\n")

    md = [
        "# Meta Muse (`Muse-Spark-Open-816B-A42B` & `Muse-Glimmer-30B`) on Cloud TPU (`v7x` & `v5p`) — Measured SGLang-TorchTPU Results",
        "",
        f"- **Generated (UTC):** `{dt.datetime.now(dt.timezone.utc).isoformat()}`",
        f"- **Source runs:** {len(runs)} result directories",
        "- **`Output tok/s/chip`** divides output throughput by the physical TPU chips occupied by that serving topology (`TP2` on `v5p` = 2 chips; `TP2` on `v7x` = 1 chip; `2×TP2` gateway on `v5p` = 4 chips; `4×TP2` / `2×TP4` / `TP8` on `v7x` = 4 chips; `TP32` on `v5p-32` = 32 chips).",
        "",
    ]
    seen_hws = []
    for r in merged:
        if r["hardware"] not in seen_hws:
            seen_hws.append(r["hardware"])

    for hw in seen_hws:
        recs = [r for r in merged if r["hardware"] == hw]
        title = SECTION_TITLES.get(hw, hw)
        md += [
            f"## {title}",
            "",
            f"{recs[0]['hardware_desc']}",
            "",
            "| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip "
            "| Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|",
        ]
        for r in recs:
            chips = r.get("chips_used") or r["chips"]
            if not r.get("cells"):
                md.append(
                    f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | - | - | - | - | - | - | - | - | - | "
                    f"`{r['status']}` |"
                )
                continue
            for c in r["cells"]:
                m = c.get("metrics", {})
                otps = m.get("output_throughput")
                acc = m.get("accept_length", m.get("spec_accept_length"))
                st = "PASS" if m and c.get("return_code") == 0 else f"FAIL(rc={c.get('return_code')})"
                md.append(
                    f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | `{c['input_len']}` / `{c['output_len']}` | "
                    f"{c['concurrency']} | {fmt(m.get('request_throughput'), 3)} | {fmt(otps)} | "
                    f"{fmt(otps / chips if otps else None)} | {fmt(m.get('total_throughput'))} | "
                    f"{fmt(m.get('median_ttft_ms'), 0)} / {fmt(m.get('p99_ttft_ms'), 0)} | "
                    f"{fmt(m.get('median_tpot_ms'), 2)} / {fmt(m.get('p99_tpot_ms'), 2)} | {fmt(acc, 2)} | `{st}` |"
                )
        md += ["", "### Variant status & sanity check", ""]
        for r in recs:
            line = f"- `{r['name']}`: **{r['status']}** (server ready after `{r.get('server_ready_s', '-')}` s)"
            if r.get("note"):
                line += f" — _{r['note']}_"
            if r["status"] not in ("DONE", "RUNNING") and not r.get("cells"):
                line += f"\n  - reason: `{short_reason(r)}`"
            if r.get("sanity", {}).get("ok"):
                s = r["sanity"]
                txt = (s.get("content_tail") or (s.get("content") or s.get("reasoning_content") or "")[-200:]).replace("\n", " ")
                line += f"\n  - sanity (greedy chat): `{txt[-200:]}`"
            earlier = [a for a in r.get("attempts", [])[:-1] if a.get("reason")]
            for a in earlier:
                line += f"\n  - earlier attempt: {a['status']} — `{a['reason'][:200]}`"
            md.append(line)
        md.append("")

    (out / "SUMMARY.md").write_text(sanitize_text("\n".join(md)) + "\n")
    print(f"wrote {out/'summary.json'} and {out/'SUMMARY.md'} ({len(merged)} variant records from {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
