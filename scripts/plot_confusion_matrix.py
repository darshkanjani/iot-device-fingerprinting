#!/usr/bin/env python3
# scripts/plot_confusion_matrix.py

"""
Render a confusion-matrix CSV as a heatmap figure.

The input CSV is expected to contain true labels as the row index and predicted
labels as columns. This is the format produced by the experiment scripts in this
project.

Features:
    - optional row normalisation;
    - optional cell annotations for smaller matrices;
    - PNG output;
    - optional PDF output.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def maybe_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-normalise a confusion matrix.

    Each row is divided by its row sum so that cell values represent the
    fraction of samples from each true class assigned to each predicted class.
    Rows with zero total support are filled with zeros.
    """
    row_sums = df.sum(axis=1).replace(0, np.nan)
    return df.div(row_sums, axis=0).fillna(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a confusion-matrix heatmap from a CSV file."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the confusion-matrix CSV.",
    )
    parser.add_argument(
        "--normalize_rows",
        action="store_true",
        help="Normalise each row to fractions.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write values in cells. Best suited to smaller matrices.",
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

    df = pd.read_csv(args.input_csv, index_col=0)

    if args.normalize_rows:
        df = maybe_normalize(df)

    fig_w = max(8, 0.35 * len(df.columns))
    fig_h = max(7, 0.35 * len(df.index))

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(df.values, aspect="auto")
    plt.colorbar()

    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.index)), df.index)

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

    if args.annotate and len(df.index) <= 20 and len(df.columns) <= 20:
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.iat[i, j]
                text = f"{val:.2f}" if args.normalize_rows else str(int(val))
                plt.text(j, i, text, ha="center", va="center", fontsize=7)

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