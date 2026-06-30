#!/usr/bin/env python3
# scripts/run_temporal_decay.py

"""
Run a temporal-decay experiment using source_file day groups.

The experiment trains each model on the earliest capture groups and evaluates it
on progressively later non-overlapping test windows. This measures how model
performance changes as the test data moves further away from the training
window.

The output CSV contains one row per model and test chunk. It can be used by
plot_temporal_decay.py to generate the temporal-decay figure.
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
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
        description=(
            "Run temporal decay by training on early groups and testing on "
            "progressively later chunks."
        )
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Clean CSV containing a temporal/group column such as source_file.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output CSV for per-model, per-chunk metrics.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "gb"],
        help="Model keys from the run_baseline_models registry.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Label column to predict. Default: device.",
    )
    parser.add_argument(
        "--group_col",
        default="source_file",
        help="Column whose unique values define temporal groups.",
    )
    parser.add_argument(
        "--train_groups",
        type=int,
        default=10,
        help="Number of earliest groups to use for training.",
    )
    parser.add_argument(
        "--test_window",
        type=int,
        default=2,
        help="Number of groups per test chunk.",
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

    print(f"Loading input data: {args.input_csv}")
    start = time.time()

    df = cap_high_cardinality_columns(
        pd.read_csv(args.input_csv, low_memory=False),
        args.top_n_requested_server_name,
    )

    print(f"  Loaded {len(df):,} rows in {format_time(time.time() - start)}")

    if args.group_col not in df.columns:
        raise ValueError(f"Temporal group column not found: {args.group_col}")

    if args.label_col not in df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    ordered_groups = sorted(df[args.group_col].astype(str).unique().tolist())

    train_group_names = ordered_groups[: args.train_groups]
    later_groups = ordered_groups[args.train_groups :]

    if not later_groups:
        raise ValueError(
            f"No test groups remain after using the first {args.train_groups} "
            f"of {len(ordered_groups)} groups for training."
        )

    test_chunks = [
        later_groups[i : i + args.test_window]
        for i in range(0, len(later_groups), args.test_window)
    ]

    print(f"\n  Total groups    : {len(ordered_groups)}")
    print(
        f"  Training groups : {len(train_group_names)} "
        f"({train_group_names[0]} to {train_group_names[-1]})"
    )
    print(f"  Test chunks     : {len(test_chunks)}")
    print(f"  Window size     : {args.test_window}")
    print(f"  Models          : {', '.join(args.models)}\n")

    train_df = df[df[args.group_col].astype(str).isin(train_group_names)].copy()

    feature_cols = [col for col in df.columns if col != args.label_col]

    X_train = train_df[feature_cols].copy()
    y_train = train_df[args.label_col].astype(str).copy()

    if args.group_col in X_train.columns:
        X_train.drop(columns=[args.group_col], inplace=True)

    print(f"  Training set: {len(X_train):,} rows, {X_train.shape[1]} features\n")

    registry = make_classical_models(random_state=42)
    rows = []

    for model_name in args.models:
        if model_name not in registry:
            raise ValueError(f"Unknown model key: {model_name}")

        print(f"{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        prep = build_preprocessor(
            X_train,
            sparse_output=args.use_sparse_preprocessor,
        )
        pipe = Pipeline([
            ("prep", prep),
            ("model", registry[model_name]),
        ])

        fit_start = time.time()
        pipe.fit(X_train, y_train)
        fit_elapsed = time.time() - fit_start

        print(f"  Fit complete in {format_time(fit_elapsed)}\n")

        for chunk_index, chunk in enumerate(test_chunks, start=1):
            test_df = df[df[args.group_col].astype(str).isin(chunk)].copy()

            X_test = test_df[feature_cols].copy()
            y_test = test_df[args.label_col].astype(str).copy()

            if args.group_col in X_test.columns:
                X_test.drop(columns=[args.group_col], inplace=True)

            print(
                f"  Chunk {chunk_index}/{len(test_chunks)} "
                f"groups=[{', '.join(chunk)}] "
                f"rows={len(X_test):,} ... ",
                end="",
                flush=True,
            )

            pred_start = time.time()
            pred = pipe.predict(X_test)
            pred_elapsed = time.time() - pred_start

            macro_f1 = f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
            weighted_f1 = f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0,
            )
            accuracy = accuracy_score(y_test, pred)

            print(
                f"macro_f1={macro_f1:.4f} "
                f"weighted_f1={weighted_f1:.4f} "
                f"accuracy={accuracy:.4f} "
                f"({format_time(pred_elapsed)})"
            )

            rows.append({
                "model": model_name,
                "chunk_index": chunk_index,
                "test_groups": "|".join(chunk),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "test_rows": len(X_test),
            })

        print()

    out_df = pd.DataFrame(rows)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"{'=' * 60}")
    print(f"Saved temporal decay results to: {args.out_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()