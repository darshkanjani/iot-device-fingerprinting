#!/usr/bin/env python3
# scripts/run_transfer_unsw_to_ad.py

"""
Train on UNSW-DI and evaluate on the external UNSW-AD benign subset.

This script implements the Tier 2 cross-environment evaluation. Models are
trained on the full cleaned UNSW-DI dataset and evaluated on the prepared
UNSW-AD intersection set containing devices common to both datasets.

The evaluation measures how well models trained on one UNSW capture setting
generalise to a later/different UNSW capture setting with an overlapping device
label space.

Outputs per model:
    - predictions CSV;
    - full confusion matrix CSV;
    - test-only confusion matrix CSV;
    - classification report JSON;
    - all-seen classification report JSON.

Aggregate outputs:
    - transfer_summary.csv;
    - run_metadata.json.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
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


def save_json(path: Path, obj) -> None:
    """Write an object as formatted JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def stratified_cap(
    df: pd.DataFrame,
    label_col: str,
    max_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cap each class to at most max_per_class rows using stratified sampling.

    Classes with fewer rows than the cap are retained in full.
    """
    return df.groupby(label_col, group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), max_per_class),
            random_state=random_state,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on UNSW-DI and evaluate on the UNSW-AD intersection set."
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Clean UNSW-DI training CSV.",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Prepared UNSW-AD intersection CSV.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Directory for output artefacts.",
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
        "--max_test_per_class",
        type=int,
        default=None,
        help="Optional per-class cap for the test set.",
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    common = sorted(set(train_df.columns) & set(test_df.columns))
    required = {args.label_col}
    missing = required - set(common)

    if missing:
        raise ValueError(f"Missing required columns across train/test: {missing}")

    common = [col for col in common if col != args.label_col]

    X_train = train_df[common].copy()
    y_train = train_df[args.label_col].astype(str).copy()

    X_test = test_df[common].copy()
    y_test = test_df[args.label_col].astype(str).copy()

    for frame in (X_train, X_test):
        if "source_file" in frame.columns:
            frame.drop(columns=["source_file"], inplace=True)

    seen_labels = set(y_train.unique())
    mask = y_test.isin(seen_labels)
    dropped = int((~mask).sum())

    X_test = X_test.loc[mask].copy()
    y_test = y_test.loc[mask].copy()

    if args.max_test_per_class is not None:
        before_cap = len(X_test)

        capped = X_test.copy()
        capped[args.label_col] = y_test
        capped = stratified_cap(capped, args.label_col, args.max_test_per_class)

        y_test = capped[args.label_col]
        X_test = capped.drop(columns=[args.label_col])

        print(f"\n  Per-class cap      : {args.max_test_per_class:,}")
        print(f"  Test rows before   : {before_cap:,}")
        print(f"  Test rows after    : {len(X_test):,}")

    print(f"\n  Common features    : {X_train.shape[1]}")
    print(f"  Training rows      : {len(X_train):,}")
    print(f"  Test rows kept     : {len(X_test):,}")
    print(f"  Test rows dropped  : {dropped:,} unseen-label rows")
    print(f"  Train classes      : {len(seen_labels)}")
    print(f"  Test classes       : {y_test.nunique()}")
    print(f"  Models             : {', '.join(args.models)}\n")

    preprocessor = build_preprocessor(
        X_train,
        sparse_output=args.use_sparse_preprocessor,
    )
    model_registry = make_classical_models(random_state=42)

    summary = []
    labels = sorted(y_train.unique().tolist())

    for model_name in args.models:
        if model_name not in model_registry:
            raise ValueError(f"Unknown model key: {model_name}")

        print(f"{'=' * 60}")
        print(f"  DI-to-AD transfer | {model_name}")
        print(f"{'=' * 60}")

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model_registry[model_name]),
        ])

        fit_start = time.time()
        pipe.fit(X_train, y_train)
        fit_elapsed = time.time() - fit_start
        print(f"  Fit complete in {format_time(fit_elapsed)}")

        print(f"  Predicting on {len(X_test):,} AD rows ... ", end="", flush=True)
        pred_start = time.time()
        y_pred = pipe.predict(X_test)
        pred_elapsed = time.time() - pred_start

        test_labels = sorted(y_test.unique().tolist())

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            labels=test_labels,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(
            y_test,
            y_pred,
            labels=test_labels,
            average="weighted",
            zero_division=0,
        )

        print(f"done in {format_time(pred_elapsed)}")
        print(
            f"  Results: macro_f1={macro_f1:.4f} "
            f"weighted_f1={weighted_f1:.4f} "
            f"accuracy={accuracy:.4f}\n"
        )

        pd.DataFrame({
            "y_true": y_test.values,
            "y_pred": y_pred,
        }).to_csv(out_dir / f"{model_name}_predictions.csv", index=False)

        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=labels,
            columns=labels,
        ).to_csv(out_dir / f"{model_name}_confusion_matrix.csv")

        pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=test_labels),
            index=test_labels,
            columns=test_labels,
        ).to_csv(out_dir / f"{model_name}_confusion_matrix_test_only.csv")

        save_json(
            out_dir / f"{model_name}_classification_report.json",
            classification_report(
                y_test,
                y_pred,
                labels=test_labels,
                output_dict=True,
                zero_division=0,
            ),
        )

        save_json(
            out_dir / f"{model_name}_classification_report_all_seen.json",
            classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0,
            ),
        )

        summary.append({
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "dropped_unseen_test_rows": dropped,
            "max_test_per_class": args.max_test_per_class,
        })

    pd.DataFrame(summary).sort_values("macro_f1", ascending=False).to_csv(
        out_dir / "transfer_summary.csv",
        index=False,
    )

    save_json(
        out_dir / "run_metadata.json",
        {
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "common_feature_count": len(X_train.columns),
            "dropped_unseen_test_rows": dropped,
            "max_test_per_class": args.max_test_per_class,
            "models": args.models,
        },
    )

    print(f"{'=' * 60}")
    print(f"Saved results to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()