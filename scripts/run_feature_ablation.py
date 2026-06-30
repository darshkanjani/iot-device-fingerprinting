#!/usr/bin/env python3
# scripts/run_feature_ablation.py

"""
Run feature-family ablation experiments on a train/test pair.

The experiment measures the contribution of each feature family by removing one
family at a time, retraining the model, and measuring the change in macro-F1.

Interpretation:
    delta_macro_f1 = ablated_macro_f1 - baseline_macro_f1

    Negative delta:
        Removing the family reduced performance, suggesting that the family was
        useful.

    Near-zero or positive delta:
        Removing the family had little effect or slightly improved performance,
        suggesting limited stable contribution in that setting.

This analysis complements permutation importance by evaluating grouped feature
families rather than individual features.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from run_baseline_models import (
    build_preprocessor,
    cap_high_cardinality_columns,
    make_classical_models,
)


FAMILIES = {
    "size_volume": [
        "bidirectional_bytes",
        "bidirectional_packets",
        "src2dst_bytes",
        "dst2src_bytes",
        "src2dst_packets",
        "dst2src_packets",
        "bidirectional_min_ps",
        "bidirectional_mean_ps",
        "bidirectional_stddev_ps",
        "bidirectional_max_ps",
        "src2dst_min_ps",
        "src2dst_mean_ps",
        "src2dst_stddev_ps",
        "src2dst_max_ps",
        "dst2src_min_ps",
        "dst2src_mean_ps",
        "dst2src_stddev_ps",
        "dst2src_max_ps",
    ],
    "timing": [
        "bidirectional_duration_ms",
        "bidirectional_min_piat_ms",
        "bidirectional_mean_piat_ms",
        "bidirectional_stddev_piat_ms",
        "bidirectional_max_piat_ms",
        "src2dst_duration_ms",
        "src2dst_min_piat_ms",
        "src2dst_mean_piat_ms",
        "src2dst_stddev_piat_ms",
        "src2dst_max_piat_ms",
        "dst2src_duration_ms",
        "dst2src_min_piat_ms",
        "dst2src_mean_piat_ms",
        "dst2src_stddev_piat_ms",
        "dst2src_max_piat_ms",
    ],
    "tcp_flags": [
        "bidirectional_syn_packets",
        "bidirectional_cwr_packets",
        "bidirectional_ece_packets",
        "bidirectional_urg_packets",
        "bidirectional_ack_packets",
        "bidirectional_psh_packets",
        "bidirectional_rst_packets",
        "bidirectional_fin_packets",
        "src2dst_syn_packets",
        "src2dst_cwr_packets",
        "src2dst_ece_packets",
        "src2dst_urg_packets",
        "src2dst_ack_packets",
        "src2dst_psh_packets",
        "src2dst_rst_packets",
        "src2dst_fin_packets",
        "dst2src_syn_packets",
        "dst2src_cwr_packets",
        "dst2src_ece_packets",
        "dst2src_urg_packets",
        "dst2src_ack_packets",
        "dst2src_psh_packets",
        "dst2src_rst_packets",
        "dst2src_fin_packets",
    ],
    "protocol_application": [
        "dst_port",
        "protocol",
        "ip_version",
        "application_name",
        "application_category_name",
        "application_is_guessed",
        "application_confidence",
    ],
    "dpi_metadata": [
        "requested_server_name",
        "client_fingerprint",
        "server_fingerprint",
        "user_agent",
        "content_type",
    ],
    "rates": [
        "bidirectional_bytes_s",
        "bidirectional_packets_s",
        "src2dst_bytes_s",
        "src2dst_packets_s",
        "dst2src_bytes_s",
        "dst2src_packets_s",
    ],
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
    """Cap each class to at most max_per_class rows."""
    return df.groupby(label_col, group_keys=False).apply(
        lambda group: group.sample(
            n=min(len(group), max_per_class),
            random_state=random_state,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature-family ablation by removing one feature group at a time."
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
        "--out_dir",
        required=True,
        help="Directory for ablation outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "gb"],
        help="Model keys to evaluate.",
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
        help="Optional cap on test rows per class.",
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

    if args.label_col == "device" and "device_canonical" in test_df.columns:
        print("  Found device_canonical in test CSV; using it as the device label.")
        test_df["device"] = test_df["device_canonical"]

    if args.label_col not in train_df.columns:
        raise ValueError(f"Label column not found in training data: {args.label_col}")

    if args.label_col not in test_df.columns:
        raise ValueError(f"Label column not found in test data: {args.label_col}")

    common = sorted(set(train_df.columns) & set(test_df.columns))
    common = [col for col in common if col != args.label_col]

    X_train_full = train_df[common].copy()
    y_train = train_df[args.label_col].astype(str).copy()

    X_test_full = test_df[common].copy()
    y_test = test_df[args.label_col].astype(str).copy()

    metadata_cols = ["source_file", "device_raw", "category", "device_canonical"]

    if args.label_col != "device":
        metadata_cols.append("device")

    for frame in (X_train_full, X_test_full):
        for col in metadata_cols:
            if col in frame.columns:
                frame.drop(columns=[col], inplace=True)

    seen_labels = set(y_train.unique())
    mask = y_test.isin(seen_labels)
    dropped = int((~mask).sum())

    X_test_full = X_test_full.loc[mask].copy()
    y_test = y_test.loc[mask].copy()

    if dropped:
        print(f"  Dropped {dropped:,} test rows with labels unseen during training.")

    if args.max_test_per_class is not None:
        before_cap = len(X_test_full)

        capped = X_test_full.copy()
        capped[args.label_col] = y_test
        capped = stratified_cap(capped, args.label_col, args.max_test_per_class)

        y_test = capped[args.label_col]
        X_test_full = capped.drop(columns=[args.label_col])

        print(f"  Capped test set: {before_cap:,} -> {len(X_test_full):,} rows")

    actual_features = set(X_train_full.columns)

    print(f"\n  Features in data: {len(actual_features)}")
    for family_name, family_cols in FAMILIES.items():
        present = [col for col in family_cols if col in actual_features]
        missing = [col for col in family_cols if col not in actual_features]

        message = f"    {family_name:25s}: {len(present)} present"
        if missing:
            message += f", {len(missing)} missing ({', '.join(missing)})"
        print(message)

    n_ablations = len(args.models) * (1 + len(FAMILIES))

    print(f"\n  Models           : {', '.join(args.models)}")
    print(f"  Total train/eval : {n_ablations} runs\n")

    registry = make_classical_models(random_state=42)
    rows = []
    run_count = 0

    for model_name in args.models:
        if model_name not in registry:
            raise ValueError(f"Unknown model key: {model_name}")

        print(f"{'=' * 60}")
        print(f"  Ablation | {model_name}")
        print(f"{'=' * 60}")

        run_count += 1
        print(
            f"  [{run_count}/{n_ablations}] Baseline (all features) ... ",
            end="",
            flush=True,
        )

        start = time.time()

        base_prep = build_preprocessor(
            X_train_full,
            sparse_output=args.use_sparse_preprocessor,
        )
        base_pipe = Pipeline([
            ("prep", base_prep),
            ("model", registry[model_name]),
        ])

        base_pipe.fit(X_train_full, y_train)
        base_pred = base_pipe.predict(X_test_full)

        test_labels = sorted(y_test.unique().tolist())
        base_macro = f1_score(
            y_test,
            base_pred,
            labels=test_labels,
            average="macro",
            zero_division=0,
        )

        print(f"macro_f1={base_macro:.4f}  ({format_time(time.time() - start)})")

        rows.append({
            "model": model_name,
            "family": "none",
            "macro_f1": base_macro,
            "delta_macro_f1": 0.0,
            "removed_feature_count": 0,
        })

        for family_name, family_cols in FAMILIES.items():
            run_count += 1

            actually_removed = [
                col for col in family_cols
                if col in actual_features
            ]
            keep_cols = [
                col for col in X_train_full.columns
                if col not in set(family_cols)
            ]

            print(
                f"  [{run_count}/{n_ablations}] Remove {family_name} "
                f"({len(actually_removed)} features) ... ",
                end="",
                flush=True,
            )

            start = time.time()

            X_train = X_train_full[keep_cols].copy()
            X_test = X_test_full[keep_cols].copy()

            prep = build_preprocessor(
                X_train,
                sparse_output=args.use_sparse_preprocessor,
            )
            pipe = Pipeline([
                ("prep", prep),
                ("model", registry[model_name]),
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            macro = f1_score(
                y_test,
                pred,
                labels=test_labels,
                average="macro",
                zero_division=0,
            )
            delta = macro - base_macro

            print(
                f"macro_f1={macro:.4f}  "
                f"delta={delta:+.4f}  "
                f"({format_time(time.time() - start)})"
            )

            rows.append({
                "model": model_name,
                "family": family_name,
                "macro_f1": macro,
                "delta_macro_f1": delta,
                "removed_feature_count": len(actually_removed),
            })

        print()

    result_df = pd.DataFrame(rows).sort_values(["model", "family"])
    result_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    save_json(out_dir / "families_used.json", FAMILIES)

    print(f"{'=' * 60}")
    print(f"Saved ablation results to: {out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()