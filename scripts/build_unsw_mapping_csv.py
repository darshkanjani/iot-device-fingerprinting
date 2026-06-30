#!/usr/bin/env python3
# scripts/build_unsw_mapping_csv.py

"""
Build UNSW mapping and device-set CSVs for flow labelling and DI-to-AD
evaluation.

This script serves two purposes:

1. Flow labelling
   It creates MAC-to-device mapping files used to assign device labels to
   extracted UNSW flows. The combined mapping uses the official UNSW-DI device
   list as the primary source and supplements it with additional UNSW-related
   MAC addresses from literature/code-derived mappings.

2. Evaluation filtering
   It writes explicit device-set CSVs for the UNSW-DI and UNSW-AD comparison:
   intersection devices, DI-only devices, and AD-only devices. These files are
   used later to restrict DI-to-AD evaluation to devices present in both
   datasets.

Scope:
    This script is only for UNSW-related mappings and device sets.

It does not create:
    - MAC-to-IP mappings;
    - YourThings mapping files;
    - mappings for other IoT capture datasets.

Outputs:
    Mapping files:
        mappings/unsw_ad_only_mac_map.csv
        mappings/unsw_combined_mac_map.csv
        mappings/unsw_mapping_conflicts.csv

    Device-set files:
        mappings/unsw_intersection_devices.csv
        mappings/unsw_di_only_devices.csv
        mappings/unsw_ad_only_devices.csv
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set


# MAC-address pattern, e.g. aa:bb:cc:dd:ee:ff.
MAC_RE = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")


# Literature/code-derived UNSW MAC-to-device additions.
#
# These entries are UNSW-related only and are used to supplement the official
# UNSW-DI mapping where useful for UNSW-AD labelling and DI-to-AD comparison.
# They are MAC-to-device entries, not MAC-to-IP entries.
KOSTAS_UNSW_MACS: Dict[str, str] = {
    "d0:52:a8:00:67:5e": "Smart Things",
    "44:65:0d:56:cc:d3": "Amazon Echo",
    "70:ee:50:18:34:43": "Netatmo Welcome",
    "f4:f2:6d:93:51:f1": "TP-Link Day Night Cloud camera",
    "00:16:6c:ab:6b:88": "Samsung SmartCam",
    "30:8c:fb:2f:e4:b2": "Dropcam",
    "00:62:6e:51:27:2e": "Insteon Camera",
    "00:24:e4:11:18:a8": "Withings Smart Baby Monitor",
    "ec:1a:59:79:f4:89": "Belkin Wemo switch",
    "50:c7:bf:00:56:39": "TP-Link Smart plug",
    "74:c6:3b:29:d7:1d": "iHome",
    "ec:1a:59:83:28:11": "Belkin wemo motion sensor",
    "18:b4:30:25:be:e4": "NEST Protect smoke alarm",
    "70:ee:50:03:b8:ac": "Netatmo weather station",
    "00:24:e4:1b:6f:96": "Withings Smart scale",
    "74:6a:89:00:2e:25": "Blipcare Blood Pressure meter",
    "00:24:e4:20:28:c6": "Withings Aura smart sleep sensor",
    "d0:73:d5:01:83:08": "Light Bulbs LiFX Smart Bulb",
    "18:b7:9e:02:20:44": "Triby Speaker",
    "e0:76:d0:33:bb:85": "PIX-STAR Photo-frame",
    "70:5a:0f:e4:9b:c0": "HP Printer",
    "08:21:ef:3b:fc:e3": "Samsung Galaxy Tab",
    "e8:ab:fa:19:de:4f": "unknown maybe cam",
    "30:8c:fb:b6:ea:45": "Nest Dropcam",
    "40:f3:08:ff:1e:da": "Android Phone 1",
    "74:2f:68:81:69:42": "Laptop",
    "ac:bc:32:d4:6f:2f": "MacBook",
    "b4:ce:f6:a7:a3:c2": "Android Phone 2",
    "d0:a6:37:df:a1:e1": "IPhone",
    "f4:5c:89:93:cc:85": "MacBook-Iphone",
    "14:cc:20:51:33:ea": "TPLink Router Bridge LAN",
    "00:24:e4:10:ee:4c": "Withings Baby Monitor 2",

    # Additional UNSW-AD / extra testbed devices.
    "88:4a:ea:31:66:9d": "Ring Door Bell",
    "00:17:88:2b:9a:25": "Phillip Hue Lightbulb",
    "7c:70:bc:5d:5e:dc": "Canary Camera",
    "6c:ad:f8:5e:e4:61": "Google Chromecast",
    "28:c2:dd:ff:a5:2d": "Hello Barbie",
    "70:88:6b:10:0f:c6": "Awair air quality monitor",
    "b4:75:0e:ec:e5:a9": "Belkin Camera",
    "e0:76:d0:3f:00:ae": "August Doorbell Cam",
}


# Device sets used for UNSW-DI vs UNSW-AD evaluation filtering.
#
# These sets are not used directly for flow labelling. They define the expected
# DI-only, AD-only, and intersection label groups for later analysis.
UNSW_AD_ONLY_DEVICES = {
    "Hello Barbie",
    "Belkin Camera",
    "August Doorbell Cam",
    "Ring Door Bell",
}

UNSW_INTERSECTION_DEVICES = {
    "MacBook",
    "HP Printer",
    "Smart Things",
    "Triby Speaker",
    "Canary Camera",
    "Netatmo Welcome",
    "TP-Link Smart plug",
    "Netatmo weather station",
    "NEST Protect smoke alarm",
    "Belkin wemo motion sensor",
    "TP-Link Day Night Cloud camera",
    "Light Bulbs LiFX Smart Bulb",
    "Awair air quality monitor",
    "TPLink Router Bridge LAN",
    "Phillip Hue Lightbulb",
    "PIX-STAR Photo-frame",
    "Samsung Galaxy Tab",
    "Belkin Wemo switch",
    "Samsung SmartCam",
    "Amazon Echo",
    "Dropcam",
    "iHome",
}

UNSW_DI_ONLY_DEVICES = {
    "IPhone",
    "unknown maybe cam",
    "Withings Aura smart sleep sensor",
    "Smart Sleep Snsr",  # Preserved from source naming.
    "Withings Smart Baby Monitor",
    "Withings Smart scale",
    "Google Chromecast",
    "Blipcare Blood Pressure meter",
    "Prssr meter",  # Preserved from source naming.
    "Withings Baby Monitor 2",
    "MacBook-Iphone",
    "Laptop",
    "Insteon Camera",
    "Android Phone 1",
    "Nest Dropcam",
    "Android Phone 2",
}


def normalize_mac(mac: str) -> str:
    """Return a lowercase colon-separated MAC address."""
    return mac.strip().lower().replace("-", ":")


def normalize_device_name(name: str) -> str:
    """Return a lightly normalised device name for output storage."""
    return " ".join(str(name).strip().split())


def canonical_device_name(name: str) -> str:
    """
    Return the canonical device-name form used for comparison and grouping.

    This function is separate from normalize_device_name() because the stored
    output labels should remain close to their source naming, while set
    membership checks need a more consistent representation.
    """
    s = normalize_device_name(name)
    s = s.replace("/", "-")
    s = s.replace("(Gateway)", "").strip()
    s = " ".join(s.split())
    return s


def load_unsw_di_list(path: str) -> Dict[str, str]:
    """
    Parse the official UNSW-DI device list.

    Each relevant line is assumed to contain a device name before a MAC address.
    """
    mapping: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = MAC_RE.search(line)
            if not m:
                continue

            mac = normalize_mac(m.group(1))
            device = normalize_device_name(line[:m.start()])

            if device:
                mapping[mac] = device

    return mapping


def find_name_conflicts(
    official: Dict[str, str],
    kostas_unsw: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """
    Return MAC addresses present in both sources with different device names.

    These entries are saved for manual review. In most cases they represent
    naming differences rather than incorrect MAC-to-device associations.
    """
    conflicts: List[Tuple[str, str, str]] = []

    shared_macs = set(official.keys()) & set(kostas_unsw.keys())
    for mac in sorted(shared_macs):
        off_name = normalize_device_name(official[mac])
        kost_name = normalize_device_name(kostas_unsw[mac])

        if off_name != kost_name:
            conflicts.append((mac, off_name, kost_name))

    return conflicts


def write_mapping_csv(path: str, mapping: Dict[str, str], source_name: str) -> None:
    """
    Write a MAC-to-device mapping CSV.

    Columns:
        mac, device, source
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mac", "device", "source"])
        for mac, device in sorted(mapping.items(), key=lambda x: (x[1].lower(), x[0])):
            writer.writerow([mac, device, source_name])


def write_combined_csv(
    path: str,
    official: Dict[str, str],
    kostas_unsw: Dict[str, str],
) -> None:
    """
    Write the combined UNSW MAC-to-device mapping.

    The official UNSW-DI mapping is treated as authoritative when the same MAC
    appears in both sources. Kostas-only MAC entries are added as supplemental
    UNSW entries.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    combined = dict(official)
    for mac, device in kostas_unsw.items():
        combined.setdefault(mac, normalize_device_name(device))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mac", "device", "source"])
        for mac, device in sorted(combined.items(), key=lambda x: (x[1].lower(), x[0])):
            source = "official_unsw_di" if mac in official else "kostas_unsw"
            writer.writerow([mac, device, source])


def write_conflicts_csv(path: str, conflicts: List[Tuple[str, str, str]]) -> None:
    """
    Write mapping name conflicts for manual review.

    Columns:
        mac, official_device, kostas_device
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mac", "official_device", "kostas_device"])
        for mac, official_name, kostas_name in conflicts:
            writer.writerow([mac, official_name, kostas_name])


def write_device_set_csv(path: str, devices: Set[str], set_name: str) -> None:
    """
    Write a device-set CSV for later evaluation filtering.

    Columns:
        device, canonical_device, set_name
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device", "canonical_device", "set_name"])
        for device in sorted(devices):
            writer.writerow([device, canonical_device_name(device), set_name])


def main() -> None:
    di_list_path = "mappings/List_Of_Devices_UNSW_DI_MAC.txt"

    official = load_unsw_di_list(di_list_path)

    kostas_unsw = {
        normalize_mac(mac): normalize_device_name(device)
        for mac, device in KOSTAS_UNSW_MACS.items()
    }

    ad_only_mac_entries = {
        mac: dev
        for mac, dev in kostas_unsw.items()
        if mac not in official
    }

    conflicts = find_name_conflicts(official, kostas_unsw)

    write_mapping_csv(
        "mappings/unsw_ad_only_mac_map.csv",
        ad_only_mac_entries,
        "kostas_unsw",
    )

    write_combined_csv(
        "mappings/unsw_combined_mac_map.csv",
        official,
        kostas_unsw,
    )

    write_conflicts_csv(
        "mappings/unsw_mapping_conflicts.csv",
        conflicts,
    )

    write_device_set_csv(
        "mappings/unsw_intersection_devices.csv",
        UNSW_INTERSECTION_DEVICES,
        "intersection",
    )

    write_device_set_csv(
        "mappings/unsw_di_only_devices.csv",
        UNSW_DI_ONLY_DEVICES,
        "di_only",
    )

    write_device_set_csv(
        "mappings/unsw_ad_only_devices.csv",
        UNSW_AD_ONLY_DEVICES,
        "ad_only",
    )

    print(f"Loaded official DI mappings: {len(official)}")
    print(f"Supplemental UNSW mappings from Kostas: {len(kostas_unsw)}")
    print(f"AD-only / extra MAC entries by MAC absence from official DI: {len(ad_only_mac_entries)}")
    print(f"Name conflicts between sources: {len(conflicts)}")
    print("Wrote mapping files:")
    print("  mappings/unsw_ad_only_mac_map.csv")
    print("  mappings/unsw_combined_mac_map.csv")
    print("  mappings/unsw_mapping_conflicts.csv")
    print("Wrote device-set files:")
    print("  mappings/unsw_intersection_devices.csv")
    print("  mappings/unsw_di_only_devices.csv")
    print("  mappings/unsw_ad_only_devices.csv")


if __name__ == "__main__":
    main()