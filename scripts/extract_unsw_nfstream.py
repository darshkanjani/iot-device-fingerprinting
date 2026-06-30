#!/usr/bin/env python3
# scripts/extract_unsw_nfstream.py

"""
Extract NFStream flow features and apply dataset-specific device labelling.

This script supports the three datasets used in the project:

    - UNSW-DI, using MAC-based labelling;
    - UNSW-AD, using MAC-based labelling;
    - YourThings, using IP-based labelling.

The extraction pipeline performs the following stages:

    1. Load a MAC-to-device or IP-to-device mapping file.
    2. Extract bidirectional flow features from one or more PCAP files using
       NFStream.
    3. Label each flow by matching either source/destination MAC addresses or
       source/destination IP addresses to the mapping file.
    4. Save a raw labelled flow CSV.
    5. Remove direct identity and temporal leakage columns.
    6. Apply one of the configured feature profiles:
           - flow_only
           - flow_plus_app
           - extended
    7. Drop constant or unused metadata columns.
    8. Save a clean labelled flow CSV and optional feature/correlation metadata.

The source_file column is preserved so that temporal and group-based
experiments can later split flows by capture file.
"""

import os
import re
import glob
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from nfstream import NFStreamer


MAC_RE = re.compile(r"([0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5})")

DEFAULT_ALWAYS_DROP = {
    "label_side",
    "bidirectional_urg_packets",
    "src2dst_urg_packets",
    "dst2src_urg_packets",
}

OUI_COLS = {"src_oui", "dst_oui"}

DPI_COLS = {
    "requested_server_name",
    "client_fingerprint",
    "server_fingerprint",
    "user_agent",
    "content_type",
}

APP_NAME_COLS = {"application_name", "application_category_name"}


def normalize_mac(mac: str) -> str:
    """Return a lowercase colon-separated MAC address."""
    return str(mac).strip().lower().replace("-", ":")


def normalize_ip(ip: str) -> str:
    """Return a stripped IP-address string."""
    return str(ip).strip()


def _guess_mapping_columns(df: pd.DataFrame, label_kind: str) -> Tuple[str, str]:
    """
    Infer key and label columns from a mapping dataframe.

    Args:
        df: Mapping dataframe.
        label_kind: Either "mac" or "ip".

    Returns:
        Tuple of (key_column, label_column).

    Raises:
        ValueError: If the mapping type is unsupported or columns cannot be
        inferred.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    if label_kind == "mac":
        key_candidates = ["mac", "mac_address", "macaddress", "mac addr", "macaddr"]
    elif label_kind == "ip":
        key_candidates = ["ip", "ip_address", "ipaddress", "device_ip", "host_ip"]
    else:
        raise ValueError(f"Unsupported label_kind for guessing columns: {label_kind}")

    label_candidates = ["device", "device_name", "name", "label"]

    key_col = next((cols_lower[c] for c in key_candidates if c in cols_lower), None)
    label_col = next((cols_lower[c] for c in label_candidates if c in cols_lower), None)

    if key_col and label_col:
        return key_col, label_col

    # Headerless two-column fallback.
    if df.shape[1] == 2:
        return df.columns[1], df.columns[0]

    raise ValueError(
        f"Could not infer mapping columns for {label_kind}. "
        f"Available columns: {list(df.columns)}"
    )


def load_mapping(
    mapping_path: str,
    label_kind: str,
    key_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> Dict[str, str]:
    """
    Load a device mapping file.

    Supported formats:
        - TXT MAC list, used for the official UNSW-DI device list;
        - CSV MAC-to-device mapping;
        - CSV IP-to-device mapping.

    Args:
        mapping_path: Path to the mapping file.
        label_kind: Either "mac" or "ip".
        key_col: Optional explicit key column for CSV mappings.
        label_col: Optional explicit device-label column for CSV mappings.

    Returns:
        Dictionary mapping normalised MAC/IP values to device labels.
    """
    mapping: Dict[str, str] = {}

    if mapping_path.lower().endswith(".txt"):
        if label_kind != "mac":
            raise ValueError("TXT mapping is only supported for MAC-based labelling.")

        with open(mapping_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = MAC_RE.search(line)
                if not m:
                    continue

                mac = normalize_mac(m.group(1))
                device_name = line[: m.start()].strip()

                if device_name:
                    mapping[mac] = device_name

        return mapping

    try:
        df = pd.read_csv(mapping_path)
    except pd.errors.ParserError:
        df = pd.read_csv(mapping_path, header=None)

    if key_col is None or label_col is None:
        key_col, label_col = _guess_mapping_columns(df, label_kind)

    for _, row in df.iterrows():
        raw_key = row[key_col]
        raw_label = row[label_col]

        if pd.isna(raw_key) or pd.isna(raw_label):
            continue

        key = normalize_mac(raw_key) if label_kind == "mac" else normalize_ip(raw_key)
        label = str(raw_label).strip()

        if key and label and key != "nan" and label != "nan":
            mapping[key] = label

    return mapping


def normalize_endpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise endpoint identifier columns in a flow dataframe.

    MAC addresses are lowercased and colon-separated. IP addresses are stripped
    of surrounding whitespace.
    """
    for col in ("src_mac", "dst_mac"):
        if col in df.columns:
            df[col] = df[col].astype(str).map(normalize_mac)

    for col in ("src_ip", "dst_ip"):
        if col in df.columns:
            df[col] = df[col].astype(str).map(normalize_ip)

    return df


def label_flows(
    df: pd.DataFrame,
    endpoint_to_device: Dict[str, str],
    label_kind: str,
) -> pd.DataFrame:
    """
    Assign device labels to flows using source/destination MAC or IP columns.

    Source-side matches are preferred when both source and destination endpoints
    match the mapping. The label_side column records which endpoint supplied the
    label before it is later removed from the clean feature set.
    """
    df = normalize_endpoint_columns(df)

    if label_kind == "mac":
        src_col, dst_col = "src_mac", "dst_mac"
    elif label_kind == "ip":
        src_col, dst_col = "src_ip", "dst_ip"
    else:
        raise ValueError(f"Unsupported label_kind: {label_kind}")

    src_label = df[src_col].map(endpoint_to_device) if src_col in df.columns else None
    dst_label = df[dst_col].map(endpoint_to_device) if dst_col in df.columns else None

    if src_label is not None and dst_label is not None:
        df["device"] = src_label.combine_first(dst_label)
        df["label_side"] = pd.NA
        df.loc[src_label.notna(), "label_side"] = "src"
        df.loc[src_label.isna() & dst_label.notna(), "label_side"] = "dst"
    elif src_label is not None:
        df["device"] = src_label
        df["label_side"] = "src"
    elif dst_label is not None:
        df["device"] = dst_label
        df["label_side"] = "dst"
    else:
        df["device"] = pd.NA
        df["label_side"] = pd.NA

    return df


def drop_identity_leakage_columns(df: pd.DataFrame, keep_dst_port: bool) -> pd.DataFrame:
    """
    Remove direct identity and timestamp-derived leakage columns.

    IP addresses, MAC addresses, source ports, flow identifiers, and explicit
    timestamp columns are removed. Destination port is retained by default
    because it can be interpreted as a behavioural service feature, but it can
    also be dropped using the corresponding command-line option.
    """
    explicit_drop = {
        "src_ip",
        "dst_ip",
        "src_mac",
        "dst_mac",
        "src_port",
        "flow_id",
        "first_seen_ms",
        "last_seen_ms",
        "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms",
    }

    if not keep_dst_port:
        explicit_drop.add("dst_port")

    pattern_drop = re.compile(
        r"(?:^|_)(?:ip|mac)(?:$|_)|"
        r"(?:^|_)(?:first_seen|last_seen|timestamp|time|tstamp)(?:$|_)|"
        r"(?:^|_)(?:flow_id|id)(?:$|_)",
        flags=re.IGNORECASE,
    )

    cols_to_drop = set()
    for col in df.columns:
        if col in explicit_drop or pattern_drop.search(col):
            cols_to_drop.add(col)

    cols_to_drop.discard("device")

    return df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")


def drop_constant_and_meta_columns(
    df: pd.DataFrame,
    always_drop: Optional[set] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop configured metadata columns and columns with a single unique value.

    Args:
        df: Input dataframe.
        always_drop: Optional set of columns to drop when present.

    Returns:
        Tuple containing the cleaned dataframe and the list of dropped columns.
    """
    dropped: List[str] = []
    always_drop = always_drop or set()

    present_always = [c for c in always_drop if c in df.columns and c != "device"]
    if present_always:
        df = df.drop(columns=present_always, errors="ignore")
        dropped.extend(present_always)

    const_cols = [
        c for c in df.columns
        if c != "device" and df[c].nunique(dropna=False) <= 1
    ]
    if const_cols:
        df = df.drop(columns=const_cols, errors="ignore")
        dropped.extend(const_cols)

    dropped_unique = []
    seen = set()
    for col in dropped:
        if col not in seen:
            dropped_unique.append(col)
            seen.add(col)

    return df, dropped_unique


def apply_feature_profile(
    df: pd.DataFrame,
    profile: str,
    keep_dst_port: bool,
    keep_oui: bool,
    keep_dpi: bool,
    keep_app_names: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply a predefined feature profile.

    Profiles:
        flow_only:
            Retains statistical flow features and optionally destination port.
            Drops OUI, DPI metadata, and application-name columns.

        flow_plus_app:
            Retains flow features plus application-name/category features.
            DPI metadata and OUI fields are dropped unless explicitly retained.

        extended:
            Retains flow features plus application and DPI metadata by default.
            OUI fields are still dropped unless explicitly retained.
    """
    dropped = []
    drop_cols = set()

    if profile == "flow_only":
        if not keep_oui:
            drop_cols |= OUI_COLS
        if not keep_dpi:
            drop_cols |= DPI_COLS
        drop_cols |= APP_NAME_COLS

    elif profile == "flow_plus_app":
        if not keep_oui:
            drop_cols |= OUI_COLS
        if not keep_dpi:
            drop_cols |= DPI_COLS
        if not keep_app_names:
            drop_cols |= APP_NAME_COLS

    elif profile == "extended":
        if not keep_oui:
            drop_cols |= OUI_COLS
        if not keep_dpi:
            drop_cols |= DPI_COLS
        if not keep_app_names:
            drop_cols |= APP_NAME_COLS

    else:
        raise ValueError(f"Unknown profile: {profile}")

    if not keep_dst_port:
        drop_cols.add("dst_port")

    present = [c for c in drop_cols if c in df.columns and c != "device"]
    if present:
        df = df.drop(columns=present, errors="ignore")
        dropped.extend(present)

    return df, dropped


def write_feature_list(df_clean: pd.DataFrame, out_path: str) -> None:
    """Write a feature inventory CSV for the cleaned flow dataframe."""
    rows = [
        {"feature": c, "dtype": str(df_clean[c].dtype), "is_label": c == "device"}
        for c in df_clean.columns
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def high_correlation_pairs(
    df_clean: pd.DataFrame,
    threshold: float = 0.95,
    topk: int = 50,
) -> pd.DataFrame:
    """
    Compute highly correlated numeric feature pairs.

    The device label is excluded before correlation analysis. Only numeric
    columns are considered.
    """
    df = df_clean.copy()

    if "device" in df.columns:
        df = df.drop(columns=["device"])

    X = df.select_dtypes(include=[np.number])
    if X.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "corr_abs"])

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = []
    for col in upper.columns:
        series = upper[col].dropna()
        high = series[series >= threshold]
        for row, val in high.items():
            pairs.append((row, col, float(val)))

    out = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr_abs"])
    return out.sort_values("corr_abs", ascending=False).head(topk).reset_index(drop=True)


def process_one_pcap(
    pcap_path: str,
    endpoint_to_device: Dict[str, str],
    label_kind: str,
    statistical_analysis: bool,
    idle_timeout: int,
    active_timeout: int,
    max_flows: Optional[int],
) -> Tuple[pd.DataFrame, int]:
    """
    Extract, label, and return flows for a single PCAP file.

    Returns:
        Tuple of (flow dataframe, number of labelled flows).
    """
    kwargs = dict(
        source=pcap_path,
        statistical_analysis=statistical_analysis,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
    )

    if max_flows is not None:
        kwargs["max_nflows"] = max_flows

    streamer = NFStreamer(**kwargs)
    df = streamer.to_pandas()

    df["source_file"] = os.path.basename(pcap_path)
    df = label_flows(df, endpoint_to_device=endpoint_to_device, label_kind=label_kind)

    labeled_count = int(df["device"].notna().sum())
    return df, labeled_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract labelled NFStream flow CSVs from PCAP files."
    )

    parser.add_argument(
        "--pcap_glob",
        default="*.pcap",
        help="Glob pattern for input PCAP files.",
    )
    parser.add_argument(
        "--device_list",
        required=True,
        help="Path to the TXT or CSV endpoint-to-device mapping file.",
    )
    parser.add_argument(
        "--label_kind",
        choices=["mac", "ip"],
        default="mac",
        help="Endpoint type used for labelling: MAC for UNSW or IP for YourThings.",
    )
    parser.add_argument(
        "--mapping_key_col",
        default=None,
        help="Optional explicit mapping key column, e.g. mac or ip.",
    )
    parser.add_argument(
        "--mapping_label_col",
        default=None,
        help="Optional explicit mapping label column, e.g. device.",
    )

    parser.add_argument("--out_raw", default="flows_labeled_raw.csv")
    parser.add_argument("--out_clean", default="flows_labeled_clean.csv")

    parser.add_argument("--max_flows", type=int, default=0)
    parser.add_argument("--idle_timeout", type=int, default=120)
    parser.add_argument("--active_timeout", type=int, default=1800)

    parser.add_argument("--drop_dst_port", action="store_true")
    parser.add_argument("--keep_unlabeled", action="store_true")

    parser.add_argument("--feature_list_out", default="")
    parser.add_argument("--corr_threshold", type=float, default=0.95)
    parser.add_argument("--corr_topk", type=int, default=50)
    parser.add_argument("--corr_out", default="")
    parser.add_argument("--no_corr_scan", action="store_true")

    parser.add_argument(
        "--profile",
        choices=["flow_only", "flow_plus_app", "extended"],
        default="flow_plus_app",
    )
    parser.add_argument("--keep_oui", action="store_true")
    parser.add_argument("--keep_dpi", action="store_true")
    parser.add_argument("--drop_app_names", action="store_true")

    args = parser.parse_args()

    pcaps = sorted(glob.glob(args.pcap_glob, recursive=True))

    if not pcaps:
        raise SystemExit(f"No PCAPs matched: {args.pcap_glob}")

    if not os.path.exists(args.device_list):
        raise SystemExit(f"Mapping file not found: {args.device_list}")

    endpoint_to_device = load_mapping(
        mapping_path=args.device_list,
        label_kind=args.label_kind,
        key_col=args.mapping_key_col,
        label_col=args.mapping_label_col,
    )

    print(f"Loaded {len(endpoint_to_device)} {args.label_kind}->device entries from {args.device_list}")

    keep_dst_port = not args.drop_dst_port
    statistical_analysis = True
    max_flows = None if args.max_flows == 0 else args.max_flows

    all_frames: List[pd.DataFrame] = []
    total_flows = 0
    total_labeled = 0

    for pcap_path in pcaps:
        print(f"\n==> Processing: {pcap_path}")

        df, labeled = process_one_pcap(
            pcap_path=pcap_path,
            endpoint_to_device=endpoint_to_device,
            label_kind=args.label_kind,
            statistical_analysis=statistical_analysis,
            idle_timeout=args.idle_timeout,
            active_timeout=args.active_timeout,
            max_flows=max_flows,
        )

        n = len(df)
        total_flows += n
        total_labeled += labeled

        print(f"Flows: {n} | Labeled: {labeled} ({(labeled / n if n else 0):.1%})")
        all_frames.append(df)

    big = pd.concat(all_frames, ignore_index=True)

    Path(args.out_raw).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_clean).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nTOTAL flows: {total_flows}")
    print(f"TOTAL labeled: {total_labeled} ({(total_labeled / total_flows if total_flows else 0):.1%})")

    big.to_csv(args.out_raw, index=False)
    print(f"Saved RAW labeled CSV: {args.out_raw} shape={big.shape}")

    clean = drop_identity_leakage_columns(big, keep_dst_port=keep_dst_port)

    if not args.keep_unlabeled and "device" in clean.columns:
        before = len(clean)
        clean = clean[clean["device"].notna()].copy()
        print(f"Dropped unlabeled rows: {before - len(clean)}")

    keep_app_names = not args.drop_app_names

    clean, profile_dropped = apply_feature_profile(
        clean,
        profile=args.profile,
        keep_dst_port=keep_dst_port,
        keep_oui=args.keep_oui,
        keep_dpi=args.keep_dpi or args.profile == "extended",
        keep_app_names=keep_app_names,
    )

    if profile_dropped:
        print(f"Profile dropped {len(profile_dropped)} columns")

    clean, dropped_cols = drop_constant_and_meta_columns(
        clean,
        always_drop=DEFAULT_ALWAYS_DROP,
    )

    if dropped_cols:
        print(f"Dropped {len(dropped_cols)} constant/meta columns")

    clean.to_csv(args.out_clean, index=False)
    print(f"Saved CLEAN labeled CSV: {args.out_clean} shape={clean.shape}")

    if args.feature_list_out:
        Path(args.feature_list_out).parent.mkdir(parents=True, exist_ok=True)
        write_feature_list(clean, args.feature_list_out)
        print(f"Saved feature list: {args.feature_list_out}")

    if not args.no_corr_scan:
        corr_df = high_correlation_pairs(
            clean,
            threshold=args.corr_threshold,
            topk=args.corr_topk,
        )

        if args.corr_out:
            Path(args.corr_out).parent.mkdir(parents=True, exist_ok=True)
            corr_df.to_csv(args.corr_out, index=False)
            print(f"Saved high-correlation pairs: {args.corr_out}")
        else:
            print("\nTop high-correlation pairs:")
            print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()