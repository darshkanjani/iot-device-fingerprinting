#!/usr/bin/env python3
# scripts/run_permutation_importance.py

"""
Run permutation-importance analysis on a train/test pair.

Permutation importance measures the contribution of each individual feature by
shuffling one feature at a time in the test set and observing the resulting
change in model score.

Typical uses:

    - train on UNSW-DI and test on UNSW-DI to inspect within-dataset feature
      importance;
    - train on UNSW-DI and test on UNSW-AD to inspect which features remain
      useful under environment shift;
    - train on UNSW-DI and test on YourThings to inspect cross-dataset feature
      behaviour.

The output CSV contains:

    - feature;
    - importance_mean;
    - importance_std.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from run_baseline_models import (
    build_preprocessor,
    cap_high_cardinality_columns,
    make_classical_models,
)


def format_time(seconds: float) -> str:
    """Return a compact elapsed-time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}min"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run permutation importance for a selected model and train/test pair."
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Training CSV, usually the UNSW-DI clean dataset.",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Test CSV for the target evaluation setting.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV for per-feature importance scores.",
    )
    parser.add_argument(
        "--model",
        default="rf",
        help="Model key to evaluate. RF or GB are recommended for this analysis.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Label column to predict. Default: device.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature.",
    )
    parser.add_argument(
        "--scoring",
        default="f1_macro",
        help="Scoring metric passed to sklearn.inspection.permutation_importance.",
    )
    parser.add_argument(
        "--max_test_rows",
        type=int,
        default=None,
        help="Optional stratified subsample size for the test set.",
    )
    parser.add_argument(
        "--use_sparse_preprocessor",
        action="store_true",
        help="Use sparse one-hot preprocessing to reduce memory usage.",
    )
    parser.add_argument(
        "--top_n_requested_server_name",
        type=int,
        default=None,
        help="If set, cap requested_server_name to the top-N most frequent values.",
    )

    args = parser.parse_args()

    print(f"Loading training data: {args.train_csv}")
    start = time.time()
    train_df = cap_high_cardinality_columns(
        pd.read_csv(args.train_csv, low_memory=False),
        args.top_n_requested_server_name,
    )
    print(f"  {len(train_df):,} rows loaded in {format_time(time.time() - start)}")

    print(f"Loading test data: {args.test_csv}")
    start = time.time()
    test_df = cap_high_cardinality_columns(
        pd.read_csv(args.test_csv, low_memory=False),
        args.top_n_requested_server_name,
    )
    print(f"  {len(test_df):,} rows loaded in {format_time(time.time() - start)}")

    if args.label_col == "device" and "device_canonical" in test_df.columns:
        print("  Found device_canonical in test CSV; using it as the device label.")
        test_df["device"] = test_df["device_canonical"]

    if args.label_col not in train_df.columns:
        raise ValueError(f"Label column not found in training data: {args.label_col}")

    if args.label_col not in test_df.columns:
        raise ValueError(f"Label column not found in test data: {args.label_col}")

    common = sorted(set(train_df.columns) & set(test_df.columns))
    common = [col for col in common if col != args.label_col]

    X_train = train_df[common].copy()
    y_train = train_df[args.label_col].astype(str).copy()

    X_test = test_df[common].copy()
    y_test = test_df[args.label_col].astype(str).copy()

    metadata_cols = ["source_file", "device_raw", "category", "device_canonical"]

    if args.label_col != "device":
        metadata_cols.append("device")

    for frame in (X_train, X_test):
        for col in metadata_cols:
            if col in frame.columns:
                frame.drop(columns=[col], inplace=True)

    seen_labels = set(y_train.unique())
    mask = y_test.isin(seen_labels)
    dropped = int((~mask).sum())

    X_test = X_test.loc[mask].copy()
    y_test = y_test.loc[mask].copy()

    if dropped:
        print(f"  Dropped {dropped:,} test rows with labels unseen during training.")

    if args.max_test_rows is not None and len(X_test) > args.max_test_rows:
        before = len(X_test)

        tmp = X_test.copy()
        tmp[args.label_col] = y_test

        tmp = tmp.groupby(args.label_col, group_keys=False).apply(
            lambda group: group.sample(
                n=min(
                    len(group),
                    max(1, int(args.max_test_rows * len(group) / before)),
                ),
                random_state=42,
            )
        )

        y_test = tmp[args.label_col]
        X_test = tmp.drop(columns=[args.label_col])

        print(f"  Subsampled test set: {before:,} -> {len(X_test):,} rows")

    n_features = X_test.shape[1]
    total_calls = n_features * args.n_repeats

    print(f"\n  Model              : {args.model}")
    print(f"  Training rows      : {len(X_train):,}")
    print(f"  Test rows          : {len(X_test):,}")
    print(f"  Features           : {n_features}")
    print(f"  Repeats            : {args.n_repeats}")
    print(f"  Total predict calls: {total_calls}")
    print(f"  Scoring            : {args.scoring}\n")

    registry = make_classical_models(random_state=42)

    if args.model not in registry:
        raise ValueError(f"Unknown model key: {args.model}")

    print(f"Training {args.model} ...", end=" ", flush=True)
    fit_start = time.time()

    model = registry[args.model]
    prep = build_preprocessor(
        X_train,
        sparse_output=args.use_sparse_preprocessor,
    )
    pipe = Pipeline([
        ("prep", prep),
        ("model", model),
    ])

    pipe.fit(X_train, y_train)

    print(f"done in {format_time(time.time() - fit_start)}")

    print(f"Running permutation importance ({total_calls} predict calls) ...")
    perm_start = time.time()

    result = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=args.n_repeats,
        random_state=42,
        scoring=args.scoring,
        n_jobs=1,
    )

    print(f"  Complete in {format_time(time.time() - perm_start)}")

    out_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print("\n  Top 15 features by importance:")
    for _, row in out_df.head(15).iterrows():
        print(
            f"    {row['feature']:40s}  "
            f"{row['importance_mean']:.4f} ± {row['importance_std']:.4f}"
        )

    print(f"\n{'=' * 60}")
    print(f"Saved permutation importance to: {args.out_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()