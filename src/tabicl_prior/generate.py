"""Generate TabICLv2 prior data for one curriculum stage and save it to disk.

Examples
--------
Classifier, stage 1, 1000 batches of 64 datasets, 8 worker processes::

    python -m tabicl_prior.generate --task clf --stage 1 \
        --save_dir data/clf_stage1 --num_batches 1000 --n_jobs 8

Regressor, stage 3, continue an interrupted run at batch 250::

    python -m tabicl_prior.generate --task reg --stage 3 \
        --save_dir data/reg_stage3 --num_batches 750 --resume_from 250

Any preset value can be overridden on the command line (e.g. ``--max_seq_len 2048``).
Run with ``--show_config`` to print the effective configuration without generating.

Output format
-------------
One file ``batch_XXXXXX.pt`` per batch plus ``metadata.json``. The format is the one
written by ``python -m tabicl.prior`` and read by ``tabicl.prior._genload.LoadPriorDataset``,
so a directory can be given directly to ``python -m tabicl.train --prior_dir ...``
(or read back with :func:`tabicl_prior.io.iter_datasets`).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from tabicl_prior._dataset import PriorDataset
from tabicl_prior._genload import dense2sparse, seed_worker
from tabicl_prior._prior_config import DEFAULT_FIXED_HP, DEFAULT_SAMPLED_HP
from tabicl_prior.graph_lib._config import PriorConfig
from tabicl_prior.stages import STAGES, TASKS, get_stage_config

warnings.filterwarnings("ignore", message=".*nested tensors is in prototype stage.*", category=UserWarning)

SOURCE = "https://github.com/soda-inria/tabicl @ a626a1b67049b0b1748ebe8c92f7df40aaebd334"


def _str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected True/False, got {value!r}")


def _optional_int(value: str):
    return None if value.lower() == "none" else int(value)


def _train_size(value: str):
    value = float(value)
    if 0 < value < 1:
        return value
    if value.is_integer():
        return int(value)
    raise argparse.ArgumentTypeError("train size must be an int (absolute) or a float in (0, 1) (ratio)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tabicl_prior.generate",
        description="Generate TabICLv2 synthetic prior datasets for a given pre-training stage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("stage selection")
    g.add_argument("--task", choices=TASKS, required=True, help="clf = classifier prior, reg = regressor prior")
    g.add_argument("--stage", type=int, choices=STAGES, required=True, help="pre-training curriculum stage")

    g = p.add_argument_group("output")
    g.add_argument("--save_dir", type=str, default=None, help="output directory (default: data/<task>_stage<stage>)")
    g.add_argument("--num_batches", type=int, default=100, help="number of batches (files) to generate")
    g.add_argument("--resume_from", type=int, default=0, help="index of the first batch file to write")
    g.add_argument("--seed", type=int, default=None, help="base seed (preset: 42 clf, 43 reg); combined with resume_from")
    g.add_argument("--n_jobs", type=int, default=os.cpu_count() or 1, help="DataLoader worker processes (0 = main process)")
    g.add_argument("--num_threads_per_generate", type=int, default=1)
    g.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    g.add_argument("--show_config", action="store_true", help="print the effective configuration and exit")

    g = p.add_argument_group("dataset shape (defaults come from the stage preset)")
    g.add_argument("--prior_type", choices=["mlp_scm", "tree_scm", "mix_scm", "graph_scm"])
    g.add_argument("--regression", type=_str2bool)
    g.add_argument("--batch_size", type=int, help="datasets per batch file (= per optimizer step in training)")
    g.add_argument("--batch_size_per_gp", type=int, help="datasets sharing seq_len / train_size")
    g.add_argument("--min_features", type=int)
    g.add_argument("--max_features", type=int)
    g.add_argument("--max_classes", type=int)
    g.add_argument("--min_seq_len", type=_optional_int, help="'none' = always use max_seq_len")
    g.add_argument("--max_seq_len", type=int)
    g.add_argument("--log_seq_len", type=_str2bool)
    g.add_argument("--log_n_features", type=_str2bool, default=False)
    g.add_argument("--seq_len_per_gp", type=_str2bool)
    g.add_argument("--min_train_size", type=_train_size)
    g.add_argument("--max_train_size", type=_train_size)
    g.add_argument("--replay_small", type=_str2bool, default=False)

    # graph_scm options (same flags and defaults as tabicl.train / python -m tabicl.prior)
    PriorConfig.add_args_to_parser(p.add_argument_group("graph_scm prior options"))
    return p


def parse_args(argv=None) -> argparse.Namespace:
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.task is None or pre.stage is None:
        parser.parse_args(argv)  # prints the usage error
    preset = get_stage_config(pre.task, pre.stage)
    parser.set_defaults(**{k: v for k, v in preset.items() if k not in ("task", "stage", "train_steps")})
    args = parser.parse_args(argv)
    args.train_steps = preset["train_steps"]
    if args.save_dir is None:
        args.save_dir = f"data/{args.task}_stage{args.stage}"
    return args


def effective_config(args: argparse.Namespace) -> dict:
    cfg = {k: v for k, v in vars(args).items() if k != "show_config"}
    cfg["prior_config"] = vars(PriorConfig.from_args(args))
    cfg["source"] = SOURCE
    return cfg


def make_prior(args: argparse.Namespace) -> PriorDataset:
    return PriorDataset(
        regression=args.regression,
        batch_size=args.batch_size,
        batch_size_per_gp=args.batch_size_per_gp,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=args.log_seq_len,
        log_n_features=args.log_n_features,
        seq_len_per_gp=args.seq_len_per_gp,
        min_train_size=args.min_train_size,
        max_train_size=args.max_train_size,
        replay_small=args.replay_small,
        prior_type=args.prior_type,
        config=PriorConfig.from_args(args),
        scm_fixed_hp=DEFAULT_FIXED_HP,
        scm_sampled_hp=DEFAULT_SAMPLED_HP,
        n_jobs=1,  # parallelism is across batches (DataLoader workers), not within a batch
        num_threads_per_generate=args.num_threads_per_generate,
        device=args.device,
    )


def _save_batch(save_dir: Path, idx: int, X, y, d, seq_lens, train_sizes, nested: bool) -> int:
    if nested:
        B = len(d)
    else:
        B, T, H = X.shape
        X = dense2sparse(X.reshape(-1, H), d.repeat_interleave(T), dtype=torch.float32)
    final = save_dir / f"batch_{idx:06d}.pt"
    tmp = save_dir / f"batch_{idx:06d}.pt.tmp"
    torch.save({"X": X, "y": y, "d": d, "seq_lens": seq_lens, "train_sizes": train_sizes, "batch_size": B}, tmp)
    tmp.replace(final)  # atomic: a reader never sees a half-written file
    return final.stat().st_size


def generate(args: argparse.Namespace) -> None:
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Metadata: the keys read by LoadPriorDataset plus the full effective configuration.
    meta = effective_config(args)
    meta_file = save_dir / "metadata.json"
    if meta_file.exists():
        old = json.loads(meta_file.read_text())
        keep = ("task", "stage", "regression", "batch_size", "min_seq_len", "max_seq_len", "prior_type")
        diff = {k: (old.get(k), meta.get(k)) for k in keep if old.get(k) != meta.get(k)}
        if diff:
            sys.exit(f"{save_dir} already holds data generated with a different config: {diff}")
    meta_file.write_text(json.dumps(meta, indent=2, default=str))

    # Seed the main process; DataLoader derives each worker's seed from it, and
    # seed_worker propagates that seed to numpy / random inside the worker.
    # (seed, resume_from) are mixed with SeedSequence so that chunks generated in
    # parallel with different --resume_from never share a random stream.
    seed = int(np.random.SeedSequence([args.seed, args.resume_from]).generate_state(1)[0])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    prior = make_prior(args)
    print(prior, flush=True)
    loader = DataLoader(
        prior,
        batch_size=None,
        shuffle=False,
        num_workers=args.n_jobs,
        prefetch_factor=2 if args.n_jobs > 0 else None,
        worker_init_fn=seed_worker if args.n_jobs > 0 else None,
        persistent_workers=args.n_jobs > 0,
    )
    it = iter(loader)

    first, last = args.resume_from, args.resume_from + args.num_batches
    print(f"[{args.task} stage {args.stage}] writing batches {first}..{last - 1} to {save_dir}", flush=True)
    t0, total_bytes = time.time(), 0
    for i, idx in enumerate(range(first, last), start=1):
        X, y, d, seq_lens, train_sizes = next(it)
        nested = bool(getattr(X, "is_nested", False))
        total_bytes += _save_batch(
            save_dir, idx, X.cpu(), y.cpu(), d.cpu(), seq_lens.cpu(), train_sizes.cpu(), nested
        )
        if i == 1 or i % 10 == 0 or i == args.num_batches:
            el = time.time() - t0
            print(
                f"  {i}/{args.num_batches} batches | {el:.0f}s | {el / i:.2f}s/batch | "
                f"{total_bytes / i / 2**20:.1f} MiB/batch",
                flush=True,
            )
    print("done.", flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.show_config:
        print(json.dumps(effective_config(args), indent=2, default=str))
        return
    generate(args)


if __name__ == "__main__":
    main()
