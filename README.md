# prior-complexity

How complex are the synthetic datasets used to pre-train tabular foundation models, compared with real-world data?

This project samples classification datasets from the **Kumo Tabular synthetic prior** ([NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models)) at each of its three curriculum stages. It extracts data **complexity measures** from them ([pymfe](https://github.com/ealcobaca/pymfe)) and compares the results with the same measures computed on real classification benchmarks (**TabArena** and **TALENT**).

## Repository structure

```
data/processed/     # real datasets as CSVs (Git LFS)
  tabarena/         #   38 classification tasks from TabArena v0.1
  talent/           #   183 classification datasets from TALENT (TabArena duplicates removed)
notebooks/          # 01_data → 02_extract → 03_compare
results/            # complexity measures, one row per dataset
src/kumo_prior/     # standalone copy of the Kumo Tabular prior
run.sh              # SLURM job that runs 02_extract.ipynb
```

Every processed CSV has the features in the first columns and the label-encoded target in the last column (`target`).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python ≥ 3.12.

```bash
git lfs install
git clone https://github.com/lpfgarcia/prior-complexity
cd prior-complexity
uv sync
```

`pymfe` is installed from a fork pinned to a specific commit (see `pyproject.toml`).

## The Kumo prior

`src/kumo_prior` is a standalone copy of the Kumo Tabular prior, with the presets of each pre-training stage ([arXiv:2611.23001](https://arxiv.org/abs/2611.23001)):

| Stage | Rows per dataset          | Train fraction | Notes                                                        |
|-------|---------------------------|----------------|--------------------------------------------------------------|
| 1     | 1,024 (fixed)             | 30%–90%        | most augmentations off, trivial datasets always rejected     |
| 2     | 400–10,240 (log-uniform)  | 79%–81%        | every augmentation at its default probability                |
| 3     | 400–60,000 (log-uniform)  | 79%–81%        | replay of small datasets and column caps per row count       |

```python
from kumo_prior import sample_datasets

for ds in sample_datasets('clf', stage=1, n_datasets=10, seed=0):
    print(ds['X'].shape, ds['y'].shape)
```

## Pipeline
 
1. **`01_data.ipynb`:** downloads TabArena from OpenML and converts TALENT into `data/processed/`, dropping TALENT datasets that duplicate TabArena ones. The processed CSVs are already in the repository; to regenerate TALENT, download the raw data from the [TALENT benchmark](https://github.com/LAMDA-Tabular/TALENT) into `data/raw/`.
2. **`02_extract.ipynb`:** extracts the complexity measures (`f1`–`f4`, `l1`–`l3`, `n1`–`n3`, `t1`–`t4`, `c1`, `c2`) from the real datasets or from datasets sampled from the Kumo prior, appending them to `results/` as it goes. Run it with `run.sh`:
```bash
sbatch run.sh real JOBS                             # → results/real.csv
sbatch run.sh synthetic STAGE DATASETS JOBS PRIOR   # → results/kumo_stage{STAGE}.csv
```
 
   `JOBS` and `PRIOR` are the worker processes for the extraction and for the prior sampler.
3. **`03_compare.ipynb`:** plots and compares the measures.

## Results

| File                          | Content                                    |
|-------------------------------|--------------------------------------------|
| `results/real.csv`            | TabArena and TALENT datasets               |
| `results/kumo_stage{1,2,3}.csv` | synthetic datasets of each Kumo stage    |

## License

`src/kumo_prior` is derived from [NVIDIA/structured-data-models](https://github.com/NVIDIA/structured-data-models) (commit `6d4a128`) and is licensed under Apache-2.0; see `src/kumo_prior/LICENSE`.