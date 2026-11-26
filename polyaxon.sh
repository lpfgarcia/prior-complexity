#!/bin/bash
# Polyaxon entrypoint: sets up the environment and calls run.sh.
# uso: polyaxon.sh MODE STAGE DATASETS JOBS
# env: OUT_DIR (obrigatório), SMOKE=1 (não retoma dos CSVs e valida a saída),
#      SCRATCH / UV_CACHE_DIR (para rodar fora do cluster)
set -ex
MODE=$1 STAGE=$2 DATASETS=$3 JOBS=$4

mkdir -p "$OUT_DIR"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
# Give the Polyaxon sidecar time to sync logs and outputs, even on failure.
trap 'echo "exit code $?"; sleep 60' EXIT
echo "OUT_DIR=$OUT_DIR MODE=$MODE STAGE=$STAGE DATASETS=$DATASETS JOBS=$JOBS SMOKE=${SMOKE:-0}"

S=${SCRATCH:-/scratch/luis_garcia}
command -v uv >/dev/null || pip install -q uv
export UV_CACHE_DIR=${UV_CACHE_DIR:-$S/uv-cache} UV_LINK_MODE=copy PYTHONUNBUFFERED=1

if [ "$MODE" = "real" ]; then
    OUT=real.csv
    [ -d data/processed ] || { echo "data/processed ausente: o modo real precisa dos CSVs"; exit 1; }
else
    OUT=kumo_stage$STAGE.csv
fi

# results/ lives in OUT_DIR so partial progress survives if the job dies;
# existing CSVs are copied first so the notebook resumes from them.
# A smoke run starts empty, otherwise every sampled id would already be done.
mkdir -p "$OUT_DIR/results"
[ "${SMOKE:-0}" = 1 ] || cp -n results/*.csv "$OUT_DIR/results/" 2>/dev/null || true
rm -rf results
ln -sfn "$OUT_DIR/results" results

uv sync --frozen

STATUS=0
if [ "$MODE" = "real" ]; then
    bash run.sh real "$JOBS" || STATUS=$?
else
    bash run.sh synthetic "$STAGE" "$DATASETS" "$JOBS" || STATUS=$?
fi
[ $STATUS = 0 ] || echo "EXTRACAO FALHOU com código $STATUS"

cp notebooks/02_extract_*.ipynb "$OUT_DIR/" 2>/dev/null || true
ls -laR "$OUT_DIR"

if [ "${SMOKE:-0}" = 1 ] && [ $STATUS = 0 ]; then
    ROWS=$(($(wc -l < "$OUT_DIR/results/$OUT") - 1))
    echo "SMOKE: $ROWS linhas em $OUT (esperado $DATASETS)"
    [ "$ROWS" = "$DATASETS" ] || STATUS=1
fi

# Fail the run so Polyaxon reports it, after the outputs were saved.
exit $STATUS
