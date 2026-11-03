#!/usr/bin/env python3
"""Generates the Muse SGLang-TorchTPU sweep Job manifests (one Job per TPU host, real values)."""
import sys
from pathlib import Path

IMAGE = "us-central1-docker.pkg.dev/YOUR_GCP_PROJECT_ID/vllm/sglang-torchtpu-diffusion:v0.2.0"

TEMPLATE = """---
apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: default
  labels: {{app: muse-sweep, muse-job: {name}}}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels: {{app: muse-sweep, muse-job: {name}}}
      annotations:
        gke-gcsfuse/volumes: "true"
        gke-gcsfuse/cpu-limit: "0"
        gke-gcsfuse/memory-limit: "0"
        gke-gcsfuse/ephemeral-storage-limit: "0"
        cluster-autoscaler.kubernetes.io/safe-to-evict: "false"
    spec:
      restartPolicy: Never
      terminationGracePeriodSeconds: 30
      serviceAccountName: default
      nodeSelector:
        cloud.google.com/gke-nodepool: {pool}
        cloud.google.com/gke-tpu-accelerator: {accel}
        cloud.google.com/gke-tpu-topology: {topo}
        kubernetes.io/hostname: {host}
      tolerations:
      - {{key: google.com/tpu, operator: Exists, effect: NoSchedule}}
{extra_tol}      containers:
      - name: sglang
        image: {image}
        imagePullPolicy: IfNotPresent
        command: ["bash", "-lc"]
        args:
        - |
          set -euo pipefail
          df -h /dev/shm | tail -1
          for m in {markers}; do
            until gcloud storage ls "$m" >/dev/null 2>&1; do echo "waiting for $m"; sleep 30; done
          done
          python3 /sweep/run_muse_sweep.py --plan /sweep/{plan} --only {only} --tag-suffix {suffix} \\
            --out-root /models/benchmark-results
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef: {{name: hf-token, key: HF_TOKEN}}
        - name: PJRT_DEVICE
          value: "TPU"
        - name: NODE_NAME
          valueFrom:
            fieldRef: {{fieldPath: spec.nodeName}}
        resources:
          requests:
            google.com/tpu: "4"
            cpu: "{cpu}"
            memory: {mem}
          limits:
            google.com/tpu: "4"
        volumeMounts:
        - {{name: models-gcs, mountPath: /models}}
        - {{name: sweep, mountPath: /sweep}}
        - {{name: dshm, mountPath: /dev/shm}}
      volumes:
      - name: models-gcs
        persistentVolumeClaim:
          claimName: tpu-models-gcs-pvc
      - name: sweep
        configMap:
          name: sglang-muse-sweep
      - name: dshm
        emptyDir:
          medium: Memory
          sizeLimit: {shm}
"""

V5P = dict(pool="tpu-v5p-32-pool", accel="tpu-v5p-slice", topo="2x4x4", cpu=96, mem="300Gi", shm="300Gi",
           extra_tol="")
V7X = dict(pool="tpu-v7x-cluster-8-pool-b", accel="tpu7x", topo="2x2x2", cpu=96, mem="400Gi", shm="400Gi",
           extra_tol="      - {key: cloud.google.com/gke-queued, operator: Exists, effect: NoSchedule}\n")

JOBS = {
    # name: (cluster kind, host, plan, variants, tag suffix)
    "muse-glimmer-v5p-tp4": ("v5p", "gke-tpu-6abd1b5a-34dk", "plan_glimmer_v5p.json", "glimmer_bf16_tp4", "_tp4"),
    "muse-glimmer-v5p-dflash": ("v5p", "gke-tpu-6abd1b5a-51dp", "plan_glimmer_v5p.json",
                                "glimmer_bf16_tp4_dflash16", "_dflash"),
    "muse-glimmer-v5p-greedy": ("v5p", "gke-tpu-6abd1b5a-99cs", "plan_glimmer_v5p.json",
                                "glimmer_bf16_tp4_dflash16_greedy", "_greedy"),
    "muse-glimmer-v5p-fp8tp2": ("v5p", "gke-tpu-6abd1b5a-gd4v", "plan_glimmer_v5p.json", "glimmer_fp8_tp2", "_fp8tp2"),
    "muse-glimmer-v5p-dp2": ("v5p", "gke-tpu-6abd1b5a-99cs", "plan_glimmer_v5p.json", "glimmer_bf16_tp2_dp2", "_dp2"),
    "muse-glimmer-v5p-dp4": ("v5p", "gke-tpu-6abd1b5a-gd4v", "plan_glimmer_v5p.json", "glimmer_bf16_tp1_dp4", "_dp4"),
    "muse-glimmer-v5p-fp8": ("v5p", "gke-tpu-6abd1b5a-gk3h", "plan_glimmer_v5p.json", "glimmer_fp8_tp4", "_fp8"),
    "muse-glimmer-v5p-dpattn2": ("v5p", "gke-tpu-6abd1b5a-99cs", "plan_glimmer_v5p.json", "glimmer_bf16_tp4_dpattn2",
                                 "_dpattn2"),
    "muse-glimmer-v5p-dpattn4": ("v5p", "gke-tpu-6abd1b5a-gd4v", "plan_glimmer_v5p.json", "glimmer_bf16_tp4_dpattn4",
                                 "_dpattn4"),
    "muse-glimmer-v5p-tp2": ("v5p", "gke-tpu-6abd1b5a-gk3h", "plan_glimmer_v5p.json", "glimmer_bf16_tp2",
                             "_tp2"),
    "muse-glimmer-v5p-dflash8": ("v5p", "gke-tpu-6abd1b5a-l0sp", "plan_glimmer_v5p.json", "glimmer_bf16_tp4_dflash8",
                                 "_dflash8"),
    "muse-glimmer-v5p-fp8kv": ("v5p", "gke-tpu-6abd1b5a-sbvp", "plan_glimmer_v5p.json", "glimmer_bf16w_fp8kv_tp4", "_fp8kv"),
    "muse-glimmer-v7x": ("v7x", "gke-tpu-771db71a-cz6l", "plan_glimmer_v7x.json",
                         "glimmer_bf16_tp8,glimmer_bf16_tp8_dflash16,glimmer_fp8_tp8,glimmer_bf16_tp4_dp2,"
                         "glimmer_fp8_tp8_dflash16", "_all"),
    "muse-glimmer-v7x-b": ("v7x", "gke-tpu-771db71a-cz6l", "plan_glimmer_v7x.json",
                           "glimmer_bf16_tp8_dflash16,glimmer_fp8_tp2,glimmer_bf16_tp2,glimmer_bf16_tp4", "_b"),
}


GCS = "gs://YOUR_GCS_BUCKET/models"
MARKERS = {
    "plan_glimmer_v5p.json": [f"{GCS}/meta-models/Muse-Glimmer-30B/_STAGE_COMPLETE.txt",
                              f"{GCS}/meta-models/Muse-Glimmer-30B-assistant/_STAGE_COMPLETE.txt"],
    "plan_glimmer_v7x.json": [f"{GCS}/meta-models/Muse-Glimmer-30B/_STAGE_COMPLETE.txt",
                              f"{GCS}/meta-models/Muse-Glimmer-30B-assistant/_STAGE_COMPLETE.txt"],
}


def render(name: str) -> str:
    kind, host, plan, only, suffix = JOBS[name]
    hw = V5P if kind == "v5p" else V7X
    return TEMPLATE.format(name=name, host=host, plan=plan, only=only, suffix=suffix, image=IMAGE,
                           markers=" ".join(MARKERS[plan]), **hw)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in JOBS.items():
        (out / f"{name}.yaml").write_text(render(name))
        print(out / f"{name}.yaml")
