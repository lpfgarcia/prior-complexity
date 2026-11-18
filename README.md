# prior-complexity

How complex are the synthetic datasets used to pre-train tabular foundation models, compared with real-world data?

This project samples classification datasets from the **TabICLv2 synthetic prior** at each of its three curriculum stages and extracts **data complexity measures** ([pymfe](https://github.com/ealcobaca/pymfe), `complexity` group) from them. It compares the results with the same measures computed on real classification benchmarks (**TabArena** and **TALENT**).

## Repository structure

```
data/        # real datasets (processed CSVs in Git LFS)
notebooks/   # 01_preprocessing → 02_analysis → 03_evaluate
results/     # complexity measures
src/         # TabICLv2 prior
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

You can run the notebooks from the `notebooks/` folder without installing anything else:

```bash
uv run jupyter lab
```

## Pipeline

1. **`01_preprocessing.ipynb`:** downloads TabArena from OpenML and converts TALENT into `data/processed/`. The processed CSVs are already in the repository. The TALENT raw data is not versioned; to regenerate it, download it from the [TALENT benchmark](https://github.com/LAMDA-Tabular/TALENT) into `data/raw/`.
2. **`02_analysis.ipynb`:** samples synthetic datasets from the TabICLv2 prior and extracts the complexity measures into `results/`.
3. **`03_evaluate.ipynb`:** plots and compares the measures.

## License

The code in `src/tabicl_prior` is derived from [TabICL](https://github.com/soda-inria/tabicl); see that repository for its license.