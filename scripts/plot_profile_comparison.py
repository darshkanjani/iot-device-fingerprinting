#!/usr/bin/env python3
# scripts/plot_profile_comparison.py

"""
Plot feature-profile comparison from experiment summary CSV files.

This script is intended for profile-sensitivity comparisons where not every
feature profile has necessarily been evaluated with the same set of models.

By default, the script keeps only models that are present in every input summary.
A specific subset of models can also be selected explicitly, for example:

    --models rf

Typical use:
    Compare flow_only, flow_plus_app, and extended profiles using a shared model
    such as Random Forest.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_tagged(paths, tags):
    """
    Load summary CSV files and attach a feature-profile label to each row.

    Args:
        paths: Summary CSV paths.
        tags: Feature-profile labels corresponding to the CSV paths.

    Returns:
        Combined dataframe containing all tagged summaries.

    Raises:
        ValueError: If an input summary does not contain a model column.
    """
    frames = []

    for path, tag in zip(paths, tags):
        df = pd.read_csv(path).copy()

        if "model" not in df.columns:
            raise ValueError(f"'model' column missing in {path}")

        frames.append(df.assign(profile=tag))

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature-profile comparisons from summary CSV files."
    )
    parser.add_argument(
        "--summary_csvs",
        nargs="+",
        required=True,
        help="One or more summary CSV files.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Feature-profile labels, one per summary CSV.",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "accuracy"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional explicit model subset to keep, e.g. rf.",
    )
    parser.add_argument(
        "--out_png",
        required=True,
        help="Output path for the PNG figure.",
    )
    parser.add_argument(
        "--out_pdf",
        default="",
        help="Optional output path for the PDF figure.",
    )
    parser.add_argument(
        "--out_csv",
        default="",
        help="Optional output path for the filtered comparison CSV.",
    )

    args = parser.parse_args()

    if len(args.summary_csvs) != len(args.tags):
        raise ValueError("--summary_csvs and --tags must have the same length.")

    df = load_tagged(args.summary_csvs, args.tags)

    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in summaries.")

    if args.models:
        keep_models = set(args.models)
    else:
        profile_model_sets = [set(group["model"]) for _, group in df.groupby("profile")]
        keep_models = set.intersection(*profile_model_sets)

    df = df[df["model"].isin(keep_models)].copy()

    if df.empty:
        raise ValueError("No overlapping models left after filtering.")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)

    pivot = df.pivot_table(
        index="model",
        columns="profile",
        values=args.metric,
        aggfunc="mean",
    )
    pivot = pivot.sort_index()

    metric_label = args.metric.replace("_", " ").title()

    ax = pivot.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Feature-profile comparison by {metric_label}")
    ax.set_ylim(0, min(1.0, max(0.05, pivot.max().max() * 1.15)))

    plt.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=220, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")

    plt.close()


if __name__ == "__main__":
    main()