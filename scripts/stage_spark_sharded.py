#!/usr/bin/env python3
"""Sharded zero-local-disk HuggingFace -> GCS model weight streamer with full-repo verification."""

import argparse
import concurrent.futures
import os
import time
from google.cloud import storage
from huggingface_hub import HfApi, hf_hub_url
import requests
from requests.adapters import HTTPAdapter


def make_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def transfer_one_file(
    session: requests.Session,
    client: storage.Client,
    repo_id: str,
    rfilename: str,
    expected_size: int,
    bucket_name: str,
    prefix: str,
    hf_token: str | None,
) -> tuple[str, str, int]:
    bucket = client.bucket(bucket_name)
    blob_path = f"{prefix.rstrip('/')}/{repo_id}/{rfilename}"
    blob = bucket.blob(blob_path)
    if blob.exists():
        blob.reload()
        if expected_size == 0 or blob.size == expected_size:
            return (rfilename, "SKIPPED_EXISTS", blob.size or 0)

    url = hf_hub_url(repo_id=repo_id, filename=rfilename)
    headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    t0 = time.time()
    for attempt in range(1, 5):
        written = 0
        try:
            with session.get(url, headers=headers, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with blob.open("wb", chunk_size=64 * 1024 * 1024) as gcs_f:
                    for chunk in resp.iter_content(chunk_size=16 * 1024 * 1024):
                        if chunk:
                            gcs_f.write(chunk)
                            written += len(chunk)
            dt = max(time.time() - t0, 1e-3)
            mbps = (written / (1024 * 1024)) / dt
            print(
                f"[stage-gcs] DONE {repo_id}/{rfilename} ({written/1e9:.2f} GB in {dt:.1f}s = {mbps:.1f} MiB/s)",
                flush=True,
            )
            return (rfilename, "UPLOADED", written)
        except Exception as exc:
            # Never log headers/tokens; sanitize URL query params if any
            msg = str(exc).split("?")[0][:240]
            print(f"[stage-gcs] WARN attempt {attempt}/4 failed for {repo_id}/{rfilename}: {type(exc).__name__}: {msg}", flush=True)
            if attempt == 4:
                raise
            time.sleep(4 * attempt)
    return (rfilename, "FAILED", 0)


def stage_repo(
    repo_id: str,
    bucket_name: str,
    prefix: str,
    workers: int,
    shard_index: int,
    num_shards: int,
    hf_token: str | None,
) -> None:
    api = HfApi(token=hf_token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    all_siblings = sorted(
        [s for s in info.siblings if not s.rfilename.startswith(".git")],
        key=lambda x: (-(x.size or 0), x.rfilename),
    )
    my_siblings = [s for i, s in enumerate(all_siblings) if i % num_shards == shard_index]
    total_bytes = sum((s.size or 0) for s in all_siblings)
    my_bytes = sum((s.size or 0) for s in my_siblings)
    print(
        f"[stage-gcs] === Shard {shard_index}/{num_shards} starting {repo_id} -> gs://{bucket_name}/{prefix}/{repo_id}/ "
        f"({len(my_siblings)}/{len(all_siblings)} files, {my_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB, workers={workers}) ===",
        flush=True,
    )
    session = make_session()
    client = storage.Client()
    t_repo = time.time()
    uploaded_bytes = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                transfer_one_file,
                session,
                client,
                repo_id,
                s.rfilename,
                s.size or 0,
                bucket_name,
                prefix,
                hf_token,
            ): s.rfilename
            for s in my_siblings
        }
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futs):
            fname, status, nbytes = fut.result()
            done_cnt += 1
            uploaded_bytes += nbytes
            if done_cnt % 5 == 0 or done_cnt == len(my_siblings):
                elapsed = max(time.time() - t_repo, 1.0)
                rate_gbps = (uploaded_bytes / 1e9) / elapsed
                print(
                    f"[stage-gcs] Shard {shard_index}/{num_shards} {repo_id}: {done_cnt}/{len(my_siblings)} files "
                    f"({uploaded_bytes/1e9:.1f}/{my_bytes/1e9:.1f} GB, {rate_gbps:.2f} GB/s, elapsed={elapsed:.0f}s)",
                    flush=True,
                )

    print(f"[stage-gcs] Shard {shard_index}/{num_shards} finished assigned files for {repo_id}.", flush=True)

    if shard_index == 0:
        bucket = client.bucket(bucket_name)
        deadline = time.time() + 7200
        while time.time() < deadline:
            existing = {
                b.name.split(f"{prefix.rstrip('/')}/{repo_id}/", 1)[-1]: b.size
                for b in client.list_blobs(bucket_name, prefix=f"{prefix.rstrip('/')}/{repo_id}/")
            }
            missing = [
                s.rfilename
                for s in all_siblings
                if s.rfilename not in existing or ((s.size or 0) > 0 and existing[s.rfilename] != s.size)
            ]
            if not missing:
                marker = bucket.blob(f"{prefix.rstrip('/')}/{repo_id}/_STAGE_COMPLETE.txt")
                marker.upload_from_string(
                    f"completed_at={time.time()}\nfiles={len(all_siblings)}\nbytes={total_bytes}\nshards={num_shards}\n"
                )
                print(
                    f"[stage-gcs] === VERIFIED ALL {len(all_siblings)} FILES & WROTE _STAGE_COMPLETE.txt FOR {repo_id} ({total_bytes/1e9:.2f} GB) ===",
                    flush=True,
                )
                return
            print(f"[stage-gcs] Shard 0 waiting on {len(missing)} remaining peer files for {repo_id}...", flush=True)
            time.sleep(15)
        raise RuntimeError(f"Timed out waiting for peer shards of {repo_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", required=True)
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--prefix", default="models")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("JOB_COMPLETION_INDEX", "0")))
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip() or None
    for repo in [r.strip() for r in args.repos.split(",") if r.strip()]:
        stage_repo(repo, args.bucket, args.prefix, args.workers, args.shard_index, args.num_shards, hf_token)


if __name__ == "__main__":
    main()
