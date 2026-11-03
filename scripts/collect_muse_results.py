#!/usr/bin/env python3
"""Collects every Muse sweep summary.json from GCS into one customer-facing summary (JSON + Markdown).

usage: python3 collect_muse_results.py <out_dir> [--prefix muse_]
Reads gs://YOUR_GCS_BUCKET/benchmark-results/<tag>_<suffix>_<ts>/summary.json (all runs),
keeps every variant record (latest run wins per (hardware, variant)), and writes:
  <out_dir>/summary.json  - merged records with source paths
  <out_dir>/SUMMARY.md    - per-hardware tables of measured numbers + failure reasons
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

BUCKET = "gs://YOUR_GCS_BUCKET/benchmark-results"


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def fmt(v, nd=1):
    return "-" if v is None else (f"{v:,.{nd}f}" if isinstance(v, (int, float)) else str(v))


def short_reason(rec: dict) -> str:
    tail = rec.get("server_log_tail", "") or ""
    for key in ("ValueError:", "RuntimeError:", "VerificationError:", "torch._dynamo.exc.Unsupported", "AssertionError",
                "NotImplementedError", "TypeError:", "KeyError:"):
        for line in tail.splitlines():
            if key in line:
                return line.strip()[:300]
    return rec.get("status", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--prefix", default="muse_")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dirs = [l.strip() for l in sh(["gcloud", "storage", "ls", f"{BUCKET}/"]).splitlines()
            if l.strip().rstrip("/").split("/")[-1].startswith(args.prefix)]
    runs = []
    for d in sorted(dirs, key=lambda x: x.rstrip("/").rsplit("_", 1)[-1]):  # by UTC timestamp suffix
        try:
            s = json.loads(sh(["gcloud", "storage", "cat", f"{d}summary.json"]))
        except Exception:
            continue
        runs.append((d, s))
    latest: dict[tuple, dict] = {}
    history: dict[tuple, list] = {}
    for d, s in runs:  # chronological
        hw = (s.get("hardware") or {}).get("name", "?")
        for v in s.get("variants", []):
            rec = dict(v)
            rec["source"] = d
            rec["hardware"] = hw
            rec["hardware_desc"] = (s.get("hardware") or {}).get("desc", "")
            rec["chips"] = (s.get("hardware") or {}).get("chips", 4)
            rec["model_repo"] = s.get("model_repo")
            rec["image"] = s.get("image")
            plan = s.get("plan") or {}
            pvar = next((x for x in plan.get("variants", []) if x.get("name") == v["name"]), {})
            sa = {**plan.get("server_args", {}), **pvar.get("server_args", {})}
            tp, dp = int(sa.get("tp_size", 1)), int(sa.get("dp_size", 1))
            devices = tp if sa.get("enable_dp_attention") else tp * dp
            rec["chips_used"] = devices / (2 if hw == "v7x" else 1)  # v7x: 2 TensorCore devices per chip
            rec["server_args_effective"] = sa
            key = (hw, v["name"])
            history.setdefault(key, []).append({"source": d, "status": v.get("status"),
                                                "reason": short_reason(rec) if not v.get("cells") else ""})
            prev = latest.get(key)
            # newest run wins, except that a run with measured cells is not replaced by a later empty/failed one
            if prev is None or v.get("cells") or not prev.get("cells"):
                latest[key] = rec
    for key, rec in latest.items():
        rec["attempts"] = history.get(key, [])
    merged = sorted(latest.values(), key=lambda r: (r["hardware"], r["name"]))
    (out / "summary.json").write_text(json.dumps(
        {"generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "runs": [d for d, _ in runs],
         "records": merged}, indent=2))
    md = [f"# Meta Muse on TPU with SGLang-TorchTPU - measured results",
          "", f"- Generated (UTC): `{dt.datetime.now(dt.timezone.utc).isoformat()}`",
          f"- Source runs: {len(runs)} result folders under `{BUCKET}/` (prefix `{args.prefix}`)",
          "- `Output tok/s/chip` divides by the chips the variant occupies (TP2 on v5p = 2 chips, TP2 on v7x = 1 chip).",
          ""]
    for hw in sorted({r["hardware"] for r in merged}):
        recs = [r for r in merged if r["hardware"] == hw]
        md += [f"## {hw}", "", f"{recs[0]['hardware_desc']}", "",
               "| Variant | Layout / precision | Chips | In / Out | Conc. | Req/s | Output tok/s | Output tok/s/chip "
               "| Total tok/s | TTFT p50 / p99 (ms) | TPOT p50 / p99 (ms) | Spec accept len | Status |",
               "|---|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
        for r in recs:
            chips = r.get("chips_used") or r["chips"]
            if not r.get("cells"):
                md.append(f"| `{r['name']}` | {r['desc']} | {fmt(chips, 0)} | - | - | - | - | - | - | - | - | - | "
                          f"`{r['status']}` |")
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
                    f"{fmt(m.get('median_tpot_ms'), 2)} / {fmt(m.get('p99_tpot_ms'), 2)} | {fmt(acc, 2)} | `{st}` |")
        md += ["", "### Variant status", ""]
        for r in recs:
            line = f"- `{r['name']}`: **{r['status']}** (server ready after `{r.get('server_ready_s', '-')}` s; `{r['source']}`)"
            if r["status"] not in ("DONE", "RUNNING"):
                line += f"\n  - reason: `{short_reason(r)}`"
            if r.get("sanity", {}).get("ok"):
                s = r["sanity"]
                txt = (s.get("content_tail") or (s.get("content") or "")[-200:]).replace("\n", " ")
                line += f"\n  - sanity (greedy chat, answer tail): `{txt[-200:]}`"
            earlier = [a for a in r.get("attempts", [])[:-1] if a.get("reason")]
            for a in earlier:
                line += f"\n  - earlier attempt `{a['source']}`: {a['status']} - `{a['reason'][:200]}`"
            md.append(line)
        md.append("")
    (out / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out/'summary.json'} and {out/'SUMMARY.md'} ({len(merged)} variant records from {len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
