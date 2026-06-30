#!/usr/bin/env python3
# scripts/plot_baseline_results.py

"""
Combine and plot experiment summary CSV files.

This script creates a grouped bar chart from one or more experiment summary
files. It is intended for model-level comparisons where each input CSV contains
the same general summary structure.

Typical uses include:

    - random-stratified vs day-held-out baseline comparison;
    - UNSW-DI to UNSW-AD transfer summaries;
    - UNSW-DI to YourThings device/category transfer summaries.

Input requirements:
    Each summary CSV must contain at least:
        - model
        - macro_f1

    The script can also plot:
        - weighted_f1
        - accuracy

Outputs:
    - PNG chart;
    - optional PDF chart;
    - optional combined CSV;
    - optional LaTeX table.
"""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def load_tagged_summaries(paths: List[str], tags: List[str]) -> pd.DataFrame:
    """
    Load summary CSV files and attach an experiment label to each row.

    Args:
        paths: Summary CSV paths.
        tags: Experiment labels corresponding to the CSV paths.

    Returns:
        Combined dataframe containing all tagged summaries.

    Raises:
        ValueError: If a summary CSV does not contain the required columns.
    """
    frames = []

    for path, tag in zip(paths, tags):
        df = pd.read_csv(path)

        if "model" not in df.columns or "macro_f1" not in df.columns:
            raise ValueError(
                f"Summary CSV must contain at least 'model' and 'macro_f1': {path}"
            )

        df = df.copy()
        df["experiment"] = tag
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def write_latex_table(df: pd.DataFrame, out_path: str) -> None:
    """
    Write a compact LaTeX table from the combined summary dataframe.

    Only columns commonly used in dissertation results tables are retained.
    """
    keep_cols = [
        col for col in [
            "experiment",
            "model",
            "macro_f1",
            "weighted_f1",
            "accuracy",
        ]
        if col in df.columns
    ]

    tex_df = df[keep_cols].copy()

    for col in ["macro_f1", "weighted_f1", "accuracy"]:
        if col in tex_df.columns:
            tex_df[col] = tex_df[col].map(lambda x: f"{x:.4f}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tex_df.to_latex(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot grouped model-comparison charts from summary CSV files."
    )
    parser.add_argument(
        "--summary_csvs",
        nargs="+",
        required=True,
        help="One or more summary CSV paths.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Experiment labels, one per summary CSV.",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "accuracy"],
        help="Metric to plot.",
    )
    parser.add_argument(
        "--out_png",
        required=True,
        help="Output path for the PNG chart.",
    )
    parser.add_argument(
        "--out_pdf",
        default="",
        help="Optional output path for the PDF chart.",
    )
    parser.add_argument(
        "--out_combined_csv",
        default="",
        help="Optional output path for the combined CSV table.",
    )
    parser.add_argument(
        "--out_latex",
        default="",
        help="Optional output path for the LaTeX table.",
    )

    args = parser.parse_args()

    if len(args.summary_csvs) != len(args.tags):
        raise ValueError("--summary_csvs and --tags must have the same length.")

    df = load_tagged_summaries(args.summary_csvs, args.tags)

    if args.out_combined_csv:
        Path(args.out_combined_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_combined_csv, index=False)

    if args.out_latex:
        write_latex_table(df, args.out_latex)

    pivot = df.pivot_table(
        index="model",
        columns="experiment",
        values=args.metric,
        aggfunc="mean",
    )
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", figsize=(11, 6), rot=0)
    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_xlabel("Model")
    ax.set_title(f"Model comparison by {args.metric.replace('_', ' ').title()}")
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