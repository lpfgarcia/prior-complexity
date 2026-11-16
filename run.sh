set -ex

pip install uv
uv pip install --system -r pyproject.toml

jupyter nbconvert --to notebook --execute analysis.ipynb --output-dir "$OUT_DIR" --ExecutePreprocessor.timeout=-1
cp complexity.csv "$OUT_DIR"