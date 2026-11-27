# Entrypoint do Polyaxon. uso: polyaxon.sh real JOBS | polyaxon.sh synthetic STAGE DATASETS JOBS
set -ex
mkdir -p "$OUT_DIR/results"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
# Ao sair (mesmo com erro), salva o notebook executado e dá tempo ao sidecar de sincronizar.
trap 'cp notebooks/02_extract_*.ipynb "$OUT_DIR/" || true; ls -la "$OUT_DIR" "$OUT_DIR/results"; sleep 60' EXIT

S=/scratch/luis_garcia
pip install -q uv
export UV_CACHE_DIR=$S/uv-cache UV_LINK_MODE=copy PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT=/tmp/venv   # fora de uploads/, que é sincronizado pelo sidecar
uv sync --frozen                          # inclui o grupo dev (jupyter)

# results/ passa a apontar para OUT_DIR, para o progresso sobreviver se o job cair.
# Os CSVs enviados são copiados antes, para o notebook retomar de onde parou.
cp -n results/*.csv "$OUT_DIR/results/" 2>/dev/null || true
rm -rf results && ln -s "$OUT_DIR/results" results

bash run.sh "$@"
