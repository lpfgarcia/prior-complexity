#!/bin/bash
# Polyaxon entrypoint: sets up the environment and calls run.sh.
# uso: polyaxon.sh MODE STAGE DATASETS JOBS
set -ex
MODE=$1 STAGE=$2 DATASETS=$3 JOBS=$4

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
echo "OUT_DIR=$OUT_DIR MODE=$MODE STAGE=$STAGE DATASETS=$DATASETS JOBS=$JOBS"

S=/scratch/luis_garcia
pip install -q uv
export UV_CACHE_DIR=$S/uv-cache UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# results/ lives in OUT_DIR so partial progress survives if the job dies;
# existing CSVs are copied first so the notebook resumes from them.
mkdir -p "$OUT_DIR/results"
cp -n results/*.csv "$OUT_DIR/results/" 2>/dev/null || true
rm -rf results
ln -sfn "$OUT_DIR/results" results

uv sync --frozen

if [ "$MODE" = "real" ]; then
    bash run.sh real "$JOBS" || echo "EXTRACAO FALHOU com código $?"
else
    bash run.sh synthetic "$STAGE" "$DATASETS" "$JOBS" || echo "EXTRACAO FALHOU com código $?"
fi

cp notebooks/02_extract_*.ipynb "$OUT_DIR/" 2>/dev/null || true
ls -laR "$OUT_DIR"
