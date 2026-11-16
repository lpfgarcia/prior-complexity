#!/bin/bash
set -ex

pip install uv
uv sync

uv run jupyter nbconvert --to notebook --execute analysis.ipynb --output-dir "$OUT_DIR"
cp complexity.csv "$OUT_DIR"