# Entrypoint do Polyaxon. uso: polyaxon.sh real JOBS | polyaxon.sh synthetic STAGE DATASETS JOBS
# Roda dentro do clone do GitHub feito pelo polyaxonfile.yaml.
set -ex
mkdir -p "$OUT_DIR/results"
exec > >(tee -a "$OUT_DIR/run.log") 2>&1
# Ao sair (mesmo com erro), salva o notebook executado e dá tempo ao sidecar de sincronizar.
trap 'cp notebooks/02_extract_*.ipynb "$OUT_DIR/" || true; ls -la "$OUT_DIR" "$OUT_DIR/results"; sleep 60' EXIT

S=/scratch/luis_garcia
pip install -q uv
export UV_CACHE_DIR=$S/uv-cache UV_LINK_MODE=copy PYTHONUNBUFFERED=1
export UV_PROJECT_ENVIRONMENT=/tmp/venv
uv sync --frozen                          # inclui o grupo dev (jupyter)

# Os CSVs reais estão no Git LFS: baixa só quando vai usá-los.
if [ "$1" = real ]; then
    apt-get update -qq && apt-get install -y -qq git-lfs
    git lfs install --local && git lfs pull
fi

# results/ passa a apontar para OUT_DIR, para o progresso sobreviver se o job cair.
# Os CSVs do repositório são copiados antes, para o notebook retomar de onde parou.
cp -n results/*.csv "$OUT_DIR/results/" 2>/dev/null || true
rm -rf results && ln -s "$OUT_DIR/results" results

bash run.sh "$@"
