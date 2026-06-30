#!/usr/bin/env python3
# scripts/prepare_ad_benign_dataset.py

"""
Prepare the UNSW-AD benign subset for UNSW-DI to UNSW-AD evaluation.

This script filters a cleaned and labelled UNSW-AD flow CSV to the device labels
that are also present in UNSW-DI. The resulting dataset is used as the external
test set for cross-environment DI-to-AD transfer experiments.

Inputs:
    - cleaned/labeled UNSW-AD CSV;
    - CSV containing the UNSW-DI / UNSW-AD intersection device set.

Output:
    - filtered UNSW-AD CSV containing only intersection devices.
"""

import argparse
from pathlib import Path

import pandas as pd


def canonicalize_name(name: str) -> str:
    """
    Return a canonical device name for DI-to-AD matching.

    Only known naming variants are normalised. This keeps labels close to their
    source naming while making the intersection filter consistent.
    """
    s = str(name).strip()
    replacements = {
        "MacBook/Iphone": "MacBook-Iphone",
        "TPLink Router Bridge LAN": "TPLink Router Bridge LAN (Gateway)",
    }
    return replacements.get(s, s)


def load_intersection_devices(path: str) -> set[str]:
    """
    Load the device set used for DI-to-AD evaluation.

    The input CSV may contain either a canonical_device column or a device
    column. Values are canonicalised before being returned.
    """
    df = pd.read_csv(path)

    if "canonical_device" in df.columns:
        return set(df["canonical_device"].astype(str).map(canonicalize_name))

    if "device" in df.columns:
        return set(df["device"].astype(str).map(canonicalize_name))

    raise ValueError("Intersection CSV must contain 'device' or 'canonical_device'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter UNSW-AD to the DI∩AD device intersection."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Cleaned and labelled UNSW-AD CSV from the extractor.",
    )
    parser.add_argument(
        "--intersection_csv",
        required=True,
        help="CSV containing the DI∩AD intersection device set.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output path for the filtered UNSW-AD CSV.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Device-label column to filter on. Default: device.",
    )
    parser.add_argument(
        "--keep_source_file",
        action="store_true",
        help="Keep the source_file column in the output if present.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    intersection = load_intersection_devices(args.intersection_csv)
    df[args.label_col] = df[args.label_col].astype(str).map(canonicalize_name)

    before_rows = len(df)
    before_devices = df[args.label_col].nunique(dropna=True)

    out = df[df[args.label_col].isin(intersection)].copy()

    if not args.keep_source_file and "source_file" in out.columns:
        out = out.drop(columns=["source_file"])

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Input rows: {before_rows}")
    print(f"Input device labels: {before_devices}")
    print(f"Intersection device set size: {len(intersection)}")
    print(f"Output rows: {len(out)}")
    print(f"Output device labels: {out[args.label_col].nunique(dropna=True)}")
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()