#!/usr/bin/env python3
# scripts/make_results_tables.py

"""
Create compact CSV and LaTeX tables from experiment summary files.

This script combines one or more summary CSV files into a single table suitable
for inclusion in the dissertation or appendices. Each input summary is assigned
an experiment tag so that results from different settings can be compared in one
output table.

Typical inputs include:

    - baseline_summary.csv files from within-dataset experiments;
    - transfer_summary.csv files from cross-environment or cross-dataset runs.

Outputs:
    - a cleaned CSV table;
    - optionally, a LaTeX table generated from the same data.
"""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create compact dissertation tables from summary CSV files."
    )
    parser.add_argument(
        "--summary_csvs",
        nargs="+",
        required=True,
        help="One or more experiment summary CSV files.",
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Experiment labels corresponding to the summary CSV files.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output path for the combined CSV table.",
    )
    parser.add_argument(
        "--out_latex",
        default="",
        help="Optional output path for the LaTeX table.",
    )
    args = parser.parse_args()

    if len(args.summary_csvs) != len(args.tags):
        raise ValueError("--summary_csvs and --tags must have the same length.")

    frames = []

    for path, tag in zip(args.summary_csvs, args.tags):
        df = pd.read_csv(path).copy()
        df.insert(0, "experiment", tag)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    keep_cols = [
        col for col in [
            "experiment",
            "model",
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "fit_seconds",
            "predict_seconds",
        ]
        if col in out.columns
    ]

    out = out[keep_cols]

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    if args.out_latex:
        tex = out.copy()

        for col in [
            "macro_f1",
            "weighted_f1",
            "accuracy",
            "fit_seconds",
            "predict_seconds",
        ]:
            if col in tex.columns:
                tex[col] = tex[col].map(
                    lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
                )

        Path(args.out_latex).parent.mkdir(parents=True, exist_ok=True)
        tex.to_latex(args.out_latex, index=False)


if __name__ == "__main__":
    main()