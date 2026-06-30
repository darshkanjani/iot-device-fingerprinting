#!/usr/bin/env python3
# scripts/plot_class_distribution.py

"""
Plot class-label distribution from a cleaned dataset CSV.

The script counts labels in a selected column, usually the device label, and
creates a horizontal bar chart. It can also export the class counts as a CSV for
use in dissertation tables or appendices.

Inputs:
    - cleaned dataset CSV containing a label column.

Outputs:
    - PNG class-distribution figure;
    - optional PDF figure;
    - optional counts CSV.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot class-label distribution from a cleaned dataset CSV."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Label column to count. Default: device.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=0,
        help="If greater than 0, plot only the top N classes.",
    )
    parser.add_argument(
        "--sort",
        choices=["count_desc", "alpha"],
        default="count_desc",
        help="Class ordering for the plot.",
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
        "--out_counts_csv",
        default="",
        help="Optional output path for the class-count CSV.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.label_col not in df.columns:
        raise ValueError(f"Column '{args.label_col}' not found in {args.input_csv}")

    counts = (
        df[args.label_col]
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis(args.label_col)
        .reset_index(name="count")
    )

    if args.sort == "alpha":
        counts = counts.sort_values(args.label_col)

    if args.top_n and args.top_n > 0:
        counts = counts.head(args.top_n)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, max(5, 0.35 * len(counts))))
    plt.barh(counts[args.label_col], counts["count"])
    plt.xlabel("Number of flows")
    plt.ylabel(args.label_col.capitalize())
    plt.title(f"Class distribution ({args.label_col})")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig(out_png, dpi=220, bbox_inches="tight")

    if args.out_pdf:
        out_pdf = Path(args.out_pdf)
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_pdf, bbox_inches="tight")

    plt.close()

    if args.out_counts_csv:
        out_counts = Path(args.out_counts_csv)
        out_counts.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(out_counts, index=False)


if __name__ == "__main__":
    main()