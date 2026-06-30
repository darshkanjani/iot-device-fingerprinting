#!/usr/bin/env python3
# scripts/plot_temporal_decay.py

"""
Plot temporal-decay results from one or more decay CSV files.

The input CSV is expected to contain a temporal/window index column and at least
one performance metric column. If a CSV contains multiple models, each model is
plotted as a separate line.

Supported x-axis columns:
    - temporal_distance
    - chunk_index
    - test_window_index
    - window_index

Supported y-axis columns, in priority order:
    - macro_f1
    - weighted_f1
    - accuracy
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def infer_x_col(df: pd.DataFrame) -> str:
    """
    Infer the temporal x-axis column from a decay-results dataframe.

    Raises:
        ValueError: If no supported temporal column is present.
    """
    for candidate in [
        "temporal_distance",
        "chunk_index",
        "test_window_index",
        "window_index",
    ]:
        if candidate in df.columns:
            return candidate

    raise ValueError("Could not infer temporal x-axis column.")


def infer_y_col(df: pd.DataFrame) -> str:
    """
    Infer the score column from a decay-results dataframe.

    Raises:
        ValueError: If no supported score column is present.
    """
    for candidate in ["macro_f1", "weighted_f1", "accuracy"]:
        if candidate in df.columns:
            return candidate

    raise ValueError("Could not infer score column.")


def pretty_metric_name(col: str) -> str:
    """Return a display label for a metric column."""
    mapping = {
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "accuracy": "Accuracy",
    }
    return mapping.get(col, col)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot temporal-decay line charts from result CSV files."
    )
    parser.add_argument(
        "--decay_csvs",
        nargs="+",
        required=True,
        help="One or more temporal-decay CSV files.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Plot labels, one per decay CSV.",
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

    if len(args.decay_csvs) != len(args.labels):
        raise ValueError("--decay_csvs and --labels must have the same length.")

    plt.figure(figsize=(10, 6))

    first_y_col = None

    for csv_path, label in zip(args.decay_csvs, args.labels):
        df = pd.read_csv(csv_path)

        x_col = infer_x_col(df)
        y_col = infer_y_col(df)

        if first_y_col is None:
            first_y_col = y_col

        if "model" in df.columns and df["model"].nunique() > 1:
            for model_name, sub in df.groupby("model", sort=True):
                sub = sub.sort_values(x_col)
                plt.plot(
                    sub[x_col],
                    sub[y_col],
                    marker="o",
                    label=f"{label} ({str(model_name).upper()})",
                )
        else:
            df = df.sort_values(x_col)
            plt.plot(df[x_col], df[y_col], marker="o", label=label)

    plt.xlabel("Temporal distance / test window")
    plt.ylabel(pretty_metric_name(first_y_col or "macro_f1"))
    plt.title("Temporal decay of device-identification performance")
    plt.legend()
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