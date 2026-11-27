#!/bin/bash
# Polyaxon entrypoint. uso: polyaxon.sh MODE STAGE DATASETS JOBS
set -ex

mkdir -p "$OUT_DIR/results"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
# On exit (even on failure) save the notebook and give the sidecar time to sync.
trap 'cp notebooks/02_extract_*.ipynb "$OUT_DIR/" || true; sleep 60' EXIT

pip install -q uv
export UV_CACHE_DIR=/scratch/luis_garcia/uv-cache UV_LINK_MODE=copy PYTHONUNBUFFERED=1
uv sync --frozen

# results/ lives in OUT_DIR so progress survives if the job dies. The existing
# CSVs are copied so the notebook resumes from them.
cp -n results/*.csv "$OUT_DIR/results/" 2>/dev/null || true
rm -rf results && ln -s "$OUT_DIR/results" results

if [ "$1" = real ]; then bash run.sh real "$4"; else bash run.sh "$@"; fi
