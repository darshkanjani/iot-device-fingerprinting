#!/usr/bin/env python3
# scripts/compute_unsw_overlap_from_extracted.py

"""
Compute the device-label overlap between extracted UNSW-DI and UNSW-AD clean CSVs.

This script reads two labelled clean flow CSVs, extracts the set of device labels
from each dataset, canonicalises a small number of known naming variants, and
writes three device-set CSVs:

    - devices present in both UNSW-DI and UNSW-AD;
    - devices present only in UNSW-DI;
    - devices present only in UNSW-AD.

The resulting CSV files are used to define the DI-to-AD intersection label space
for cross-environment evaluation.
"""

import argparse
from pathlib import Path

import pandas as pd


def canonicalize_name(name: str) -> str:
    """
    Return a canonical device name for overlap comparison.

    Only a small set of known naming variants is normalised. This avoids
    aggressive relabelling while ensuring that obvious formatting differences
    do not prevent correct device-set matching.
    """
    s = str(name).strip()
    replacements = {
        "MacBook/Iphone": "MacBook-Iphone",
        "TPLink Router Bridge LAN (Gateway)": "TPLink Router Bridge LAN",
        "Belkin Wemo switch": "Belkin Wemo switch",
    }
    return replacements.get(s, s)


def get_device_set(df: pd.DataFrame, label_col: str) -> set[str]:
    """
    Extract a canonicalised set of device labels from a dataframe.

    Args:
        df: Input dataframe containing a device-label column.
        label_col: Name of the label column to use.

    Returns:
        Set of non-empty canonical device names.

    Raises:
        ValueError: If the requested label column is not present.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")

    return {
        canonicalize_name(x)
        for x in df[label_col].dropna().astype(str)
        if str(x).strip()
    }


def write_csv(path: str, devices: list[str], set_name: str) -> None:
    """
    Write a device-set CSV.

    Columns:
        device, canonical_device, set_name
    """
    out = pd.DataFrame({
        "device": devices,
        "canonical_device": devices,
        "set_name": [set_name] * len(devices),
    })

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute UNSW-DI / UNSW-AD device-label overlap."
    )
    parser.add_argument("--di_csv", required=True, help="Path to the UNSW-DI clean CSV.")
    parser.add_argument("--ad_csv", required=True, help="Path to the UNSW-AD clean CSV.")
    parser.add_argument(
        "--label_col",
        default="device",
        help="Device-label column to compare. Default: device.",
    )
    parser.add_argument(
        "--out_intersection_csv",
        default="mappings/unsw_intersection_devices_computed.csv",
        help="Output CSV for devices present in both datasets.",
    )
    parser.add_argument(
        "--out_di_only_csv",
        default="mappings/unsw_di_only_devices_computed.csv",
        help="Output CSV for devices present only in UNSW-DI.",
    )
    parser.add_argument(
        "--out_ad_only_csv",
        default="mappings/unsw_ad_only_devices_computed.csv",
        help="Output CSV for devices present only in UNSW-AD.",
    )
    args = parser.parse_args()

    di = pd.read_csv(args.di_csv, low_memory=False)
    ad = pd.read_csv(args.ad_csv, low_memory=False)

    di_set = get_device_set(di, args.label_col)
    ad_set = get_device_set(ad, args.label_col)

    intersection = sorted(di_set & ad_set)
    di_only = sorted(di_set - ad_set)
    ad_only = sorted(ad_set - di_set)

    write_csv(args.out_intersection_csv, intersection, "intersection")
    write_csv(args.out_di_only_csv, di_only, "di_only")
    write_csv(args.out_ad_only_csv, ad_only, "ad_only")

    print(f"DI devices: {len(di_set)}")
    print(f"AD devices: {len(ad_set)}")
    print(f"Intersection: {len(intersection)}")
    print(f"DI-only: {len(di_only)}")
    print(f"AD-only: {len(ad_only)}")

    print("\nIntersection devices:")
    for device in intersection:
        print(f"  {device}")

    print("\nSaved:")
    print(f"  {args.out_intersection_csv}")
    print(f"  {args.out_di_only_csv}")
    print(f"  {args.out_ad_only_csv}")


if __name__ == "__main__":
    main()