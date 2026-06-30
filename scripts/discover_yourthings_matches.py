#!/usr/bin/env python3
# scripts/discover_yourthings_matches.py

"""
Discover possible YourThings-to-UNSW-DI device-label overlaps.

This script is an exploratory mapping aid. It does not create final labels or
force any mapping decisions. Instead, it compares raw YourThings device labels
against UNSW-DI device labels and exports files for manual review.

The script identifies:

    - exact or normalised label matches;
    - likely fuzzy-match candidates;
    - broad category hints based on device-name keywords.

Inputs:
    - a cleaned YourThings flow CSV;
    - the UNSW-DI device-label text file.

Outputs:
    - CSV of all raw YourThings device labels observed;
    - CSV of exact/normalised matches;
    - CSV of fuzzy candidate matches for manual review.
"""

import argparse
import re
from pathlib import Path
from difflib import get_close_matches

import pandas as pd


def load_di_labels(di_label_file: str) -> list[str]:
    """
    Load device labels from the UNSW-DI label text file.

    Header lines are skipped. Duplicate labels are removed while preserving
    the original order.
    """
    labels = []

    with open(di_label_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or "MAC ADDRESS" in line or line.startswith("List of Devices"):
                continue

            parts = re.split(r"\t+", line)
            name = parts[0].strip() if parts else line.strip()
            if name:
                labels.append(name)

    seen = set()
    out = []
    for label in labels:
        if label not in seen:
            seen.add(label)
            out.append(label)

    return out


def norm(s: str) -> str:
    """
    Return a normalised string for approximate device-name comparison.

    The normalisation is intentionally lightweight. It removes punctuation,
    standardises spacing, handles common spelling variants, and normalises a
    small number of device-name patterns.
    """
    s = str(s).strip().lower()
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())

    replacements = {
        "gen1": "",
        "wifi": "",
        "smartthings": "smart things",
        "wemo": "wemo",
        "tplink": "tp link",
        "tp link wifi plug": "tp link smart plug",
        "lifxvirtualbulb": "lifx bulb",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = " ".join(s.split())
    return s


def category_hint(raw_name: str) -> str | None:
    """
    Infer a broad device category from a raw device name.

    These category hints are used only to support manual review. They are not
    final labels.
    """
    n = norm(raw_name)

    if any(x in n for x in ["camera", "cam", "dropcam", "doorbell", "welcome"]):
        return "camera"
    if any(x in n for x in ["plug", "wemo switch", "switch"]):
        return "plug"
    if any(x in n for x in ["motion", "sensor", "protect", "smoke"]):
        return "sensor"
    if any(x in n for x in ["echo", "speaker", "triby"]):
        return "speaker"
    if "smart things" in n or "hub" in n:
        return "hub"
    if any(x in n for x in ["bulb", "light", "lifx", "hue"]):
        return "light"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover possible YourThings-to-UNSW-DI device-label matches."
    )
    parser.add_argument(
        "--yourthings_csv",
        required=True,
        help="Path to the cleaned YourThings flow CSV.",
    )
    parser.add_argument(
        "--di_label_file",
        required=True,
        help="Path to the UNSW-DI device-label text file.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Name of the raw device-label column in the YourThings CSV.",
    )
    parser.add_argument(
        "--out_raw_csv",
        default="outputs/prepared/yourthings_raw_devices_seen.csv",
        help="Output CSV containing raw YourThings device labels.",
    )
    parser.add_argument(
        "--out_matches_csv",
        default="outputs/prepared/yourthings_device_matches_discovered.csv",
        help="Output CSV containing exact or normalised matches.",
    )
    parser.add_argument(
        "--out_candidates_csv",
        default="outputs/prepared/yourthings_device_match_candidates.csv",
        help="Output CSV containing fuzzy candidates for manual review.",
    )
    args = parser.parse_args()

    yt = pd.read_csv(args.yourthings_csv, low_memory=False)

    if args.label_col not in yt.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    di_labels = load_di_labels(args.di_label_file)
    di_norm_map = {norm(label): label for label in di_labels}

    raw_devices = sorted(set(yt[args.label_col].dropna().astype(str).str.strip()))

    Path(args.out_raw_csv).parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.DataFrame({"device_raw": raw_devices})
    raw_df["normalized"] = raw_df["device_raw"].map(norm)
    raw_df["category_hint"] = raw_df["device_raw"].map(category_hint)
    raw_df.to_csv(args.out_raw_csv, index=False)

    exact_rows = []
    candidate_rows = []

    for raw in raw_devices:
        raw_n = norm(raw)

        if raw_n in di_norm_map:
            exact_rows.append({
                "device_raw": raw,
                "normalized": raw_n,
                "match_type": "normalized_exact",
                "matched_di_device": di_norm_map[raw_n],
                "category_hint": category_hint(raw),
            })
            continue

        close = get_close_matches(raw_n, list(di_norm_map.keys()), n=5, cutoff=0.55)
        if close:
            for candidate in close:
                candidate_rows.append({
                    "device_raw": raw,
                    "normalized": raw_n,
                    "candidate_di_device": di_norm_map[candidate],
                    "candidate_normalized": candidate,
                    "category_hint": category_hint(raw),
                })
        else:
            candidate_rows.append({
                "device_raw": raw,
                "normalized": raw_n,
                "candidate_di_device": "",
                "candidate_normalized": "",
                "category_hint": category_hint(raw),
            })

    match_df = (
        pd.DataFrame(exact_rows).sort_values(["matched_di_device", "device_raw"])
        if exact_rows
        else pd.DataFrame(
            columns=[
                "device_raw",
                "normalized",
                "match_type",
                "matched_di_device",
                "category_hint",
            ]
        )
    )

    cand_df = (
        pd.DataFrame(candidate_rows).sort_values(["device_raw", "candidate_di_device"])
        if candidate_rows
        else pd.DataFrame(
            columns=[
                "device_raw",
                "normalized",
                "candidate_di_device",
                "candidate_normalized",
                "category_hint",
            ]
        )
    )

    match_df.to_csv(args.out_matches_csv, index=False)
    cand_df.to_csv(args.out_candidates_csv, index=False)

    print(f"DI labels loaded: {len(di_labels)}")
    print(f"Unique raw YourThings devices seen: {len(raw_devices)}")
    print(f"Exact/normalized matches found: {len(match_df)}")
    print(f"Candidate rows for review: {len(cand_df)}")
    print("\nSaved:")
    print(f"  {args.out_raw_csv}")
    print(f"  {args.out_matches_csv}")
    print(f"  {args.out_candidates_csv}")


if __name__ == "__main__":
    main()