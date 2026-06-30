#!/usr/bin/env python3
# scripts/prepare_yourthings_dataset.py

"""
Prepare the YourThings dataset for device-level and category-level transfer.

This script maps raw YourThings device labels into:

    - canonical device labels aligned with the UNSW-DI label space where a
      device-level match is supported;
    - broader device categories for category-level transfer experiments.

The script produces:

    - a prepared YourThings CSV with device_raw, device_canonical, category,
      is_device_overlap, and is_category_overlap columns;
    - a compact mapping CSV for review and appendix use;
    - a device-overlap subset for device-level transfer;
    - a category-overlap subset for category-level transfer.

Device-level mappings are intentionally conservative. Broader category mappings
are used where exact device-level alignment is not appropriate but the device
belongs to a category also represented in UNSW-DI.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


# Conservative raw YourThings label to UNSW-DI canonical device mapping.
DEVICE_TO_CANONICAL = {
    "SamsungSmartThingsHub": "Smart Things",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
    "BelkinWeMoSwitch": "Belkin Wemo switch",
    "AmazonEchoGen1": "Amazon Echo",
    "NestProtect": "NEST Protect smoke alarm",
    "TP-LinkWiFiPlug": "TP-Link Smart plug",
    "iPhone": "IPhone",

    # Additional manually reviewed device-level matches.
    "RingDoorbell": "Ring Door Bell",
    "Canary": "Canary Camera",
}


# Mapping from raw or canonical device labels to broad device categories.
RAW_OR_CANONICAL_TO_CATEGORY = {
    # Canonical device labels.
    "Smart Things": "hub",
    "Belkin wemo motion sensor": "sensor",
    "Light Bulbs LiFX Smart Bulb": "light",
    "Belkin Wemo switch": "plug",
    "Amazon Echo": "speaker",
    "NEST Protect smoke alarm": "sensor",
    "TP-Link Smart plug": "plug",
    "IPhone": "phone",
    "Ring Door Bell": "camera",
    "Canary Camera": "camera",

    # Raw YourThings labels used for category-level alignment.
    "GoogleOnHub": "hub",
    "PhilipsHUEHub": "hub",
    "InsteonHub": "hub",
    "WinkHub": "hub",
    "Wink2Hub": "hub",
    "CasetaWirelessHub": "hub",
    "MiCasaVerdeVeraLite": "hub",
    "SecurifiAlmond": "hub",
    "LogitechHarmonyHub": "hub",
    "GoogleHomeHub": "hub",

    "NestCamera": "camera",
    "NestCamIQ": "camera",
    "BelkinNetcam": "camera",
    "NetgearArloCamera": "camera",
    "D-LinkDCS-5009LCamera": "camera",
    "LogitechLogiCircle": "camera",
    "PiperNV": "camera",
    "WithingsHome": "camera",
    "ChineseWebcam": "camera",
    "AugustDoorbellCam": "camera",
    "AxisNetworkCamera": "camera",
    "AVTechIPCam": "camera",
    "NestBell": "camera",

    "GoogleHomeMini": "speaker",
    "GoogleHome": "speaker",
    "BoseSoundTouch10": "speaker",
    "HarmonKardonInvoke": "speaker",
    "AppleHomePod": "speaker",
    "Sonos": "speaker",
    "SonosBeam": "speaker",
    "AmazonEchoDotGen3": "speaker",

    "KoogeekLightbulb": "light",
    "TP-LinkSmartWiFiLEDBulb": "light",

    "AndroidTablet": "tablet",
    "iPad": "tablet",
    "SamsungSmartTV": "tv",
    "LGWebOSTV": "tv",
    "RokuTV": "tv",
    "Roku4": "tv",
    "AppleTV(4thGen)": "tv",
    "AmazonFireTV": "tv",
    "nVidiaShield": "tv",

    "AndroidPhone": "phone",
    "iPhone": "phone",

    "UbuntuDesktop": "computer",
    "MyCloudEX2Ultra": "storage",
    "Roomba": "robot",
    "NestThermostat": "thermostat",
    "NestGuard": "security",
    "Rachio3": "iot_controller",
    "ChamberlainmyQGarageOpener": "garage",
    "NintendoSwitch": "console",
    "PlayStation4": "console",
    "XboxOneX": "console",
}


# UNSW-DI device labels mapped to the category universe used for transfer.
DI_DEVICE_TO_CATEGORY = {
    "Smart Things": "hub",
    "Amazon Echo": "speaker",
    "Netatmo Welcome": "camera",
    "TP-Link Day Night Cloud camera": "camera",
    "Samsung SmartCam": "camera",
    "Dropcam": "camera",
    "Insteon Camera": "camera",
    "Withings Smart Baby Monitor": "camera",
    "Belkin Wemo switch": "plug",
    "TP-Link Smart plug": "plug",
    "iHome": "speaker",
    "Belkin wemo motion sensor": "sensor",
    "NEST Protect smoke alarm": "sensor",
    "Light Bulbs LiFX Smart Bulb": "light",
    "Nest Dropcam": "camera",
    "IPhone": "phone",
    "Samsung Galaxy Tab": "tablet",
}


def load_di_labels(di_label_file: str) -> set[str]:
    """
    Load device labels from the UNSW-DI label text file.

    Header lines and MAC-only continuation lines are ignored.
    """
    labels = set()
    mac_only = re.compile(r"^[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}$")

    with open(di_label_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or "MAC ADDRESS" in line or line.startswith("List of Devices"):
                continue

            parts = re.split(r"\t+", line)
            name = parts[0].strip() if parts else line.strip()

            if mac_only.fullmatch(name):
                continue

            if name:
                labels.add(name)

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare YourThings for device-level and category-level transfer."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Cleaned and labelled YourThings CSV from the extractor.",
    )
    parser.add_argument(
        "--di_label_file",
        required=True,
        help="Path to the UNSW-DI device-label text file.",
    )
    parser.add_argument(
        "--out_csv",
        required=True,
        help="Output path for the prepared YourThings CSV.",
    )
    parser.add_argument(
        "--out_mapping_csv",
        required=True,
        help="Output path for the compact mapping summary CSV.",
    )
    parser.add_argument(
        "--out_device_overlap_csv",
        required=True,
        help="Output path for the device-overlap subset.",
    )
    parser.add_argument(
        "--out_category_overlap_csv",
        required=True,
        help="Output path for the category-overlap subset.",
    )
    parser.add_argument(
        "--label_col",
        default="device",
        help="Raw device-label column in the input CSV. Default: device.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, low_memory=False)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column not found: {args.label_col}")

    di_labels = load_di_labels(args.di_label_file)
    di_categories = set(DI_DEVICE_TO_CATEGORY.values())

    df["device_raw"] = df[args.label_col].astype(str).str.strip()
    df["device_canonical"] = df["device_raw"].map(
        lambda x: DEVICE_TO_CANONICAL.get(x, x)
    )
    df["category"] = df["device_canonical"].map(RAW_OR_CANONICAL_TO_CATEGORY)

    raw_category = df["device_raw"].map(RAW_OR_CANONICAL_TO_CATEGORY)
    df["category"] = df["category"].fillna(raw_category)

    df["is_device_overlap"] = df["device_canonical"].isin(di_labels)
    df["is_category_overlap"] = df["category"].isin(di_categories)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_mapping_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_device_overlap_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_category_overlap_csv).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(args.out_csv, index=False)

    mapping_df = (
        df[
            [
                "device_raw",
                "device_canonical",
                "category",
                "is_device_overlap",
                "is_category_overlap",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "is_device_overlap",
                "is_category_overlap",
                "device_canonical",
                "device_raw",
            ],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    mapping_df.to_csv(args.out_mapping_csv, index=False)

    df[df["is_device_overlap"]].to_csv(args.out_device_overlap_csv, index=False)
    df[df["is_category_overlap"]].to_csv(args.out_category_overlap_csv, index=False)

    print(f"DI labels loaded: {len(di_labels)}")
    print(f"Rows total: {len(df)}")
    print(f"Rows usable for device-level transfer: {int(df['is_device_overlap'].sum())}")
    print(f"Rows usable for category-level transfer: {int(df['is_category_overlap'].sum())}")

    print("\nDevice-level overlaps seen:")
    for device in sorted(
        df.loc[df["is_device_overlap"], "device_canonical"]
        .dropna()
        .astype(str)
        .unique()
    ):
        print(f"  {device}")

    print("\nCategory overlaps seen:")
    for category in sorted(
        df.loc[df["is_category_overlap"], "category"]
        .dropna()
        .astype(str)
        .unique()
    ):
        print(f"  {category}")


if __name__ == "__main__":
    main()