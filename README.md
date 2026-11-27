# prior-complexity

How complex are the synthetic datasets used to pre-train tabular foundation models, compared with real-world data?

This project samples classification datasets from the **Kumo Tabular synthetic prior** ([NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models)) at each of its three curriculum stages. It extracts data **complexity measures** from them ([pymfe](https://github.com/ealcobaca/pymfe)) and compares the results with the same measures computed on real classification benchmarks (**OpenML-CC18**, **TabArena**, **Grinsztajn** and **TALENT**).

## Real datasets

All real datasets are stored in a single folder, `data/processed/`, as `<source>__<name>.csv`, with the target in the last column (`target`). After removing duplicates, the folder holds 251 classification datasets:

| Source | Origin | Datasets |
|---|---|---|
| `cc18` | [OpenML-CC18](https://www.openml.org/s/99) (OpenML suite 99) | 72 |
| `tabarena` | [TabArena v0.1](https://tabarena.ai) (OpenML suite 457), classification tasks | 29 |
| `grinsztajn` | [Grinsztajn et al. (2022)](https://github.com/LeoGrin/tabular-benchmark), classification on numerical and categorical features (OpenML suite 334) | 6 |
| `talent` | [TALENT](https://github.com/LAMDA-Tabular/TALENT), binary and multiclass datasets | 144 |

The same dataset often appears in more than one benchmark. Duplicates are found by comparing names in lowercase with every non-alphanumeric character removed, and only the first file in alphabetical order is kept. Datasets with more than 1M rows are dropped.

## Repository structure

```
data/processed/     # real datasets as <source>__<name>.csv (Git LFS)
notebooks/          # 01_data → 02_extract → 03_compare
results/            # complexity measures, one row per dataset
src/kumo_prior/     # standalone copy of the Kumo Tabular prior
run.sh              # SLURM job that runs 02_extract.ipynb
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python ≥ 3.12.

```bash
git lfs install
git clone https://github.com/lpfgarcia/prior-complexity
cd prior-complexity
uv sync
```

The package `pymfe` is installed from a fork pinned to a specific commit.

## Pipeline
 
1. **`01_data.ipynb`:** downloads CC18, TabArena and Grinsztajn from OpenML, converts TALENT, and writes everything to `data/processed/`; then removes duplicates across sources and datasets with more than 1M rows. The processed CSVs are already in the repository; to regenerate TALENT, download the raw data from the [TALENT benchmark](https://github.com/LAMDA-Tabular/TALENT) into `data/raw/`.
2. **`02_extract.ipynb`:** extracts the complexity measures from the real datasets or from datasets sampled from the Kumo prior, appending them to `results/` as it goes. Run it with `run.sh`.
3. **`03_compare.ipynb`:** plots and compares the measures.

## Results

The complexity measures are saved in `results/`, one row per dataset: `real.csv` for the real datasets (the `id` column is the file name, so the source is its prefix) and `kumo_stage{1,2,3}.csv` for the synthetic datasets of each Kumo stage.

## License

`src/kumo_prior` is derived from [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (commit `6d4a128`) and is licensed under Apache-2.0; see `src/kumo_prior/LICENSE`.