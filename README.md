# prior-complexity

How complex are the synthetic datasets used to pre-train tabular foundation models, compared with real-world data?

This project samples classification datasets from the **Kumo Tabular synthetic prior** ([NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models)) at each of its three curriculum stages. It extracts data **complexity measures** from them ([pymfe](https://github.com/ealcobaca/pymfe)) and compares the results with the same measures computed on real classification benchmarks (**TabArena** and **TALENT**).

## Repository structure

```
data/processed/     # real datasets as CSVs (Git LFS)
  tabarena/         #  38 classification tasks from TabArena v0.1
  talent/           #  183 classification datasets from TALENT (duplicates removed)
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
 
1. **`01_data.ipynb`:** downloads TabArena from OpenML and converts TALENT into `data/processed/`, dropping TALENT datasets that duplicate TabArena ones. The processed CSVs are already in the repository; to regenerate TALENT, download the raw data from the [TALENT benchmark](https://github.com/LAMDA-Tabular/TALENT) into `data/raw/`.
2. **`02_extract.ipynb`:** extracts the complexity measures from the real datasets or from datasets sampled from the Kumo prior, appending them to `results/` as it goes. Run it with `run.sh`.
3. **`03_compare.ipynb`:** plots and compares the measures.

## Results

The complexity measures are saved in `results/`, one row per dataset: `real.csv` for the real datasets (TabArena and TALENT) and `kumo_stage{1,2,3}.csv` for the synthetic datasets of each Kumo stage.

## License

`src/kumo_prior` is derived from [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (commit `6d4a128`) and is licensed under Apache-2.0; see `src/kumo_prior/LICENSE`.