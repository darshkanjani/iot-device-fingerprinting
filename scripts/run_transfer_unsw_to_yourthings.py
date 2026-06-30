#!/usr/bin/env python3
# scripts/run_transfer_unsw_to_yourthings.py

"""
Train on UNSW-DI and evaluate on YourThings at device or category level.

This script implements the Tier 3 cross-dataset evaluation. Models are trained
on the cleaned UNSW-DI dataset and evaluated on the prepared YourThings dataset,
which comes from a different network environment and different device instances.

Supported evaluation levels:

    device:
        Predict canonical device labels. The prepared YourThings file should
        contain device_canonical labels aligned with the UNSW-DI device names.

    category:
        Predict broader functional categories such as camera, plug, sensor, hub,
        speaker, and light.

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


CATEGORY_MAP_UNSW = {
    "Smart Things": "hub",
    "Amazon Echo": "speaker",
    "Triby Speaker": "speaker",
    "iHome": "speaker",
    "Netatmo Welcome": "camera",
    "TP-Link Day Night Cloud camera": "camera",
    "Samsung SmartCam": "camera",
    "Dropcam": "camera",
    "Insteon Camera": "camera",
    "Withings Smart Baby Monitor": "camera",
    "Nest Dropcam": "camera",
    "Belkin Wemo switch": "plug",
    "TP-Link Smart plug": "plug",
    "Belkin wemo motion sensor": "sensor",
    "NEST Protect smoke alarm": "sensor",
    "Netatmo weather station": "sensor",
    "Withings Aura smart sleep sensor": "sensor",
    "Light Bulbs LiFX Smart Bulb": "light",
}


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
        description="Train on UNSW-DI and evaluate on YourThings."
    )
    parser.add_argument(
        "--train_csv",
        required=True,
        help="Clean UNSW-DI training CSV.",
    )
    parser.add_argument(
        "--test_csv",
        required=True,
        help="Prepared YourThings CSV.",
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
        "--level",
        choices=["device", "category"],
        default="device",
        help="Evaluation level.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Device-label column in the training data. Default: device.",
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

    if "device_canonical" in test_df.columns:
        print("\n  Found device_canonical in test CSV; using it as the device label.")

        if "device_raw" not in test_df.columns:
            test_df["device_raw"] = test_df["device"]

        test_df["device"] = test_df["device_canonical"]

        di_devices = set(train_df[args.label_col].astype(str).unique())
        yt_devices = set(test_df["device"].astype(str).unique())
        overlap = di_devices & yt_devices

        print(f"  Device overlap: {len(overlap)} devices")
        for device in sorted(overlap):
            print(f"    {device}")

    if args.level == "category":
        train_df = train_df.copy()
        test_df = test_df.copy()

        train_df["category"] = train_df[args.label_col].astype(str).map(
            CATEGORY_MAP_UNSW
        )

        if "category" not in test_df.columns:
            raise ValueError(
                "YourThings prepared CSV must contain a 'category' column for "
                "category-level evaluation."
            )

        label_col = "category"

        before_train = len(train_df)
        before_test = len(test_df)

        train_df = train_df[train_df[label_col].notna()].copy()
        test_df = test_df[test_df[label_col].notna()].copy()

        print("\n  Category-level mode:")
        print(f"    Train rows: {before_train:,} -> {len(train_df):,}")
        print(f"    Test rows : {before_test:,} -> {len(test_df):,}")

    else:
        label_col = args.label_col

    if label_col not in train_df.columns:
        raise ValueError(f"Label column not found in training data: {label_col}")

    if label_col not in test_df.columns:
        raise ValueError(f"Label column not found in test data: {label_col}")

    common = sorted(set(train_df.columns) & set(test_df.columns))
    common = [col for col in common if col != label_col]

    X_train = train_df[common].copy()
    y_train = train_df[label_col].astype(str).copy()

    X_test = test_df[common].copy()
    y_test = test_df[label_col].astype(str).copy()

    metadata_cols = [
        "source_file",
        "device_raw",
        "device_raw_original",
        "device_canonical",
        "is_device_overlap",
        "is_category_overlap",
        "category",
    ]

    if args.level == "category":
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

    if args.max_test_per_class is not None:
        before_cap = len(X_test)

        capped = X_test.copy()
        capped[label_col] = y_test
        capped = stratified_cap(capped, label_col, args.max_test_per_class)

        y_test = capped[label_col]
        X_test = capped.drop(columns=[label_col])

        print(f"\n  Per-class cap      : {args.max_test_per_class:,}")
        print(f"  Test rows before   : {before_cap:,}")
        print(f"  Test rows after    : {len(X_test):,}")

    print(f"\n  Evaluation level   : {args.level}")
    print(f"  Common features    : {X_train.shape[1]}")
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

    labels = sorted(y_train.unique().tolist())
    summary = []

    for model_name in args.models:
        if model_name not in model_registry:
            raise ValueError(f"Unknown model key: {model_name}")

        print(f"{'=' * 60}")
        print(f"  DI-to-YourThings transfer ({args.level}) | {model_name}")
        print(f"{'=' * 60}")

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model_registry[model_name]),
        ])

        fit_start = time.time()
        pipe.fit(X_train, y_train)
        fit_elapsed = time.time() - fit_start
        print(f"  Fit complete in {format_time(fit_elapsed)}")

        print(
            f"  Predicting on {len(X_test):,} YourThings rows ... ",
            end="",
            flush=True,
        )
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
            "level": args.level,
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
            "level": args.level,
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