#!/usr/bin/env python3
# scripts/plot_feature_ablation.py

"""
Plot feature-family ablation results as a bar chart.

The input CSV is expected to come from the feature-ablation experiment script.
Each row should describe the effect of removing one feature family from the
model input.

The script supports both:

    - single-model ablation CSVs;
    - multi-model ablation CSVs containing a model column.

Rows representing the unmodified baseline are excluded from the plot. Feature
families with zero removed features are also excluded when the
removed_feature_count column is present.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feature-family ablation results."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to the ablation summary CSV.",
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

    family_col = None
    for candidate in ["family", "feature_family", "ablation_family"]:
        if candidate in df.columns:
            family_col = candidate
            break

    if family_col is None:
        raise ValueError("Could not find feature-family column.")

    value_col = None
    for candidate in ["delta_macro_f1", "delta_f1", "macro_f1_drop", "macro_f1"]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        raise ValueError("Could not find ablation score column.")

    df = df[df[family_col] != "none"].copy()

    if "removed_feature_count" in df.columns:
        df = df[df["removed_feature_count"] > 0].copy()

    if df.empty:
        raise ValueError("No ablation rows left after filtering.")

    df = df.sort_values(value_col, ascending=True)

    ylabel_map = {
        "delta_macro_f1": "Δ Macro-F1 after removing family",
        "delta_f1": "Δ F1 after removing family",
        "macro_f1_drop": "Macro-F1 drop after removing family",
        "macro_f1": "Macro F1",
    }
    ylabel = ylabel_map.get(value_col, value_col)

    if "model" in df.columns and df["model"].nunique() > 1:
        pivot = df.pivot_table(
            index=family_col,
            columns="model",
            values=value_col,
            aggfunc="mean",
        )
        pivot = pivot.sort_values(by=list(pivot.columns)[0], ascending=True)

        ax = pivot.plot(kind="bar", figsize=(10, 6), rot=25)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Feature family")
        ax.set_title("Feature-family ablation")
        plt.xticks(rotation=25, ha="right")

    else:
        plt.figure(figsize=(10, 6))
        plt.bar(df[family_col], df[value_col])
        plt.ylabel(ylabel)
        plt.xlabel("Feature family")
        plt.title("Feature-family ablation")
        plt.xticks(rotation=25, ha="right")

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