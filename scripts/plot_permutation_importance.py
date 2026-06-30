#!/usr/bin/env python3
# scripts/plot_permutation_importance.py

"""
Plot permutation-importance results as a horizontal bar chart.

The input CSV is expected to contain one row per feature and an importance score
column produced by the permutation-importance experiment script.

Supported score-column names:
    - importance_mean
    - importance
    - mean_importance

The script sorts features by importance, keeps the top N features, and renders
them as a horizontal bar chart.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a permutation-importance bar chart from a CSV file."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the permutation-importance CSV.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of highest-importance features to plot.",
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

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if "feature" not in df.columns:
        raise ValueError("Expected a 'feature' column in permutation importance CSV.")

    score_col = None
    for candidate in ["importance_mean", "importance", "mean_importance"]:
        if candidate in df.columns:
            score_col = candidate
            break

    if score_col is None:
        raise ValueError("Could not find importance score column.")

    df = (
        df.sort_values(score_col, ascending=False)
        .head(args.top_n)
        .sort_values(score_col, ascending=True)
    )

    plt.figure(figsize=(11, max(5, 0.35 * len(df))))
    plt.barh(df["feature"], df[score_col])
    plt.xlabel(score_col.replace("_", " ").title())
    plt.ylabel("Feature")
    plt.title(f"Top {len(df)} permutation-importance features")
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