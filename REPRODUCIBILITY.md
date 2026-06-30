# Reproducibility Commands and Supporting Material Guide

**Project:** Cross-Dataset Generalisation of Flow-Based IoT Device Identification  
**Module:** CM3203 One Semester Individual Project  
**Student:** Darsh Kanjani — C23013638

This file consolidates the useful command-log, plotting, table-helper, and support-generation notes into one clean reference file. It is intended as optional supporting material alongside the submitted report PDF and source-code archive.

The raw public PCAPs, full raw flow CSVs, full cleaned/prepared datasets, and per-flow prediction CSVs are intentionally excluded from the submitted support archive because they are large generated artefacts. The report, scripts, mappings, summary outputs, tables, figures, classification reports, and confusion matrices provide the reproducibility evidence.

Run commands from the project root, i.e. the folder containing `scripts/`, `mappings/`, `outputs/`, `figures/`, and `dissertation/`.

```bash
mkdir -p outputs/tables figures dissertation/figures
```

---

## 1. Data preparation commands

### 1.1 NFStream extraction

The full NFStream extraction was run for three datasets and three feature profiles. The generated clean CSVs were placed under `outputs/clean/`.

Expected clean outputs:

```text
outputs/clean/unsw_di_all_flow_only.csv
outputs/clean/unsw_di_all_flow_plus_app.csv
outputs/clean/unsw_di_all_extended.csv
outputs/clean/unsw_ad_all_flow_only.csv
outputs/clean/unsw_ad_all_flow_plus_app.csv
outputs/clean/unsw_ad_all_extended.csv
outputs/clean/yourthings_all_flow_only.csv
outputs/clean/yourthings_all_flow_plus_app.csv
outputs/clean/yourthings_all_extended.csv
```

Approximate produced flow counts used in the report:

```text
UNSW-DI:     962,447 flows
UNSW-AD:   6,146,781 flows
YourThings: 3,978,467 flows
```

### 1.2 Compute UNSW-DI / UNSW-AD overlap

```bash
python3 scripts/compute_unsw_overlap_from_extracted.py \
  --di_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --ad_csv "outputs/clean/unsw_ad_all_flow_plus_app.csv"
```

Expected summary:

```text
UNSW-DI devices: 29
UNSW-AD devices: 26
DI∩AD devices: 19
```

### 1.3 Prepare UNSW-AD intersection set

```bash
python3 scripts/prepare_ad_benign_dataset.py \
  --input_csv "outputs/clean/unsw_ad_all_flow_plus_app.csv" \
  --intersection_csv "mappings/unsw_intersection_devices_computed.csv" \
  --out_csv "outputs/prepared/unsw_ad_intersection_flow_plus_app.csv"
```

Expected summary:

```text
6,146,781 rows -> 5,574,697 rows
19 overlapping devices retained
```

### 1.4 Cap UNSW-AD intersection to 50,000 flows per device

```bash
python3 - <<'PY'
import pandas as pd

inp = "outputs/prepared/unsw_ad_intersection_flow_plus_app.csv"
out = "outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv"

df = pd.read_csv(inp, low_memory=False)
capped = df.groupby("device", group_keys=False).apply(
    lambda g: g.sample(n=min(len(g), 50000), random_state=42)
)
capped.to_csv(out, index=False)
print(f"{len(df):,} -> {len(capped):,}")
PY
```

Expected summary:

```text
5,574,697 rows -> 405,832 rows
```

### 1.5 Prepare YourThings mapping and overlap files

```bash
python3 scripts/prepare_yourthings_dataset.py \
  --input_csv "outputs/clean/yourthings_all_flow_plus_app.csv" \
  --di_label_file "mappings/List_Of_Devices_UNSW_DI_MAC.txt" \
  --out_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_mapping_csv "outputs/prepared/yourthings_seen_mapping_flow_plus_app.csv" \
  --out_device_overlap_csv "outputs/prepared/yourthings_device_overlap_flow_plus_app.csv" \
  --out_category_overlap_csv "outputs/prepared/yourthings_category_overlap_flow_plus_app.csv"
```

Expected summary:

```text
YourThings total rows:              3,978,467
Device-level overlap rows:            602,433
Category-level overlap rows:        2,923,751
```

---

## 2. Main experiment commands

### 2.1 Tier 1: random-stratified baseline on UNSW-DI

```bash
python3 scripts/run_baseline_models.py \
  --input_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --out_dir "outputs/baselines_random_flow_plus_app_final" \
  --split_mode random_stratified \
  --models knn rf dt gb logreg gnb cnn1d \
  --save_predictions
```

Reported macro-F1 summary:

```text
KNN 0.704, RF 0.686, DT 0.666, GB 0.647, LR 0.398, GNB 0.181, CNN 0.007
```

### 2.2 Tier 1: day-held-out baseline on UNSW-DI

```bash
python3 scripts/run_baseline_models.py \
  --input_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --out_dir "outputs/baselines_dayholdout_flow_plus_app_final" \
  --split_mode day_holdout \
  --models knn rf dt gb logreg gnb cnn1d \
  --save_predictions
```

Reported macro-F1 summary:

```text
KNN 0.716, DT 0.682, RF 0.675, GB 0.662, LR 0.402, GNB 0.191, CNN 0.052
```

### 2.3 Temporal decay on UNSW-DI

```bash
python3 scripts/run_temporal_decay.py \
  --input_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --out_csv "outputs/temporal/temporal_decay_flow_plus_app.csv" \
  --models knn rf gb
```

Reported RF macro-F1 sequence:

```text
0.651 -> 0.659 -> 0.563 -> 0.466 -> 0.524
```

### 2.4 Tier 2: UNSW-DI to UNSW-AD transfer

```bash
python3 scripts/run_transfer_unsw_to_ad.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv" \
  --out_dir "outputs/transfer/unsw_di_to_ad_flow_plus_app" \
  --models knn rf gb
```

Reported macro-F1 summary:

```text
RF 0.538, GB 0.531, KNN 0.507
```

### 2.5 Tier 3: UNSW-DI to YourThings device-level transfer

```bash
python3 scripts/run_transfer_unsw_to_yourthings.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_dir "outputs/transfer/unsw_di_to_yourthings_device_flow_plus_app" \
  --models knn rf gb \
  --level device \
  --max_test_per_class 50000
```

Reported summary:

```text
Macro-F1: GB 0.129, RF 0.113, KNN 0.098
Rows after cap: 228,867
```

### 2.6 Tier 3: UNSW-DI to YourThings category-level transfer

```bash
python3 scripts/run_transfer_unsw_to_yourthings.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_dir "outputs/transfer/unsw_di_to_yourthings_category_flow_plus_app" \
  --models knn rf gb \
  --level category \
  --max_test_per_class 50000
```

Reported summary:

```text
Macro-F1: RF 0.141, KNN 0.124, GB 0.112
Rows after cap: 280,112
```

---

## 3. Feature analysis commands

### 3.1 Permutation importance: within-dataset

```bash
python3 scripts/run_permutation_importance.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --out_csv "outputs/importance/importance_within.csv" \
  --model rf \
  --n_repeats 10 \
  --max_test_rows 100000
```

Expected top feature:

```text
application_name, mean Δ macro-F1 ≈ 0.0287
```

### 3.2 Permutation importance: UNSW-DI to UNSW-AD

```bash
python3 scripts/run_permutation_importance.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv" \
  --out_csv "outputs/importance/importance_cross_ad.csv" \
  --model rf \
  --n_repeats 10 \
  --max_test_rows 100000
```

Expected top feature:

```text
application_name, mean Δ macro-F1 ≈ 0.0366
```

### 3.3 Permutation importance: UNSW-DI to YourThings

```bash
python3 scripts/run_permutation_importance.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_csv "outputs/importance/importance_cross_yourthings.csv" \
  --model rf \
  --n_repeats 10 \
  --max_test_rows 100000
```

### 3.4 Feature-family ablation: within-dataset

```bash
python3 scripts/run_feature_ablation.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --out_dir "outputs/ablation/ablation_within_flow_plus_app_rf_only" \
  --models rf \
  --max_test_per_class 50000
```

Reported RF summary:

```text
RF baseline macro-F1 ≈ 0.7401
Largest drops: size_volume ≈ -0.1728, timing ≈ -0.0750
```

### 3.5 Feature-family ablation: UNSW-DI to UNSW-AD

```bash
python3 scripts/run_feature_ablation.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv" \
  --out_dir "outputs/ablation/ablation_ad_flow_plus_app_rf_only" \
  --models rf
```

### 3.6 Feature-family ablation: UNSW-DI to YourThings device-level

```bash
python3 scripts/run_feature_ablation.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_dir "outputs/ablation/ablation_yourthings_device_flow_plus_app_rf_only" \
  --models rf \
  --max_test_per_class 50000
```

### 3.7 Feature-family ablation: UNSW-DI to YourThings category-level

Only use this if the category-level ablation was run and the output folder exists.

```bash
python3 scripts/run_feature_ablation.py \
  --train_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --test_csv "outputs/prepared/yourthings_prepared_flow_plus_app.csv" \
  --out_dir "outputs/ablation/ablation_yourthings_category_flow_plus_app_rf_only" \
  --models rf \
  --label_col category \
  --max_test_per_class 50000
```

---

## 4. Profile-sensitivity commands

These commands produce the RF-only feature-profile comparison used as an auxiliary analysis.

### 4.1 Tier 1 flow-only RF baseline

```bash
python3 scripts/run_baseline_models.py \
  --input_csv "outputs/clean/unsw_di_all_flow_only.csv" \
  --out_dir "outputs/baselines_dayholdout_flow_only_rf_only" \
  --split_mode day_holdout \
  --models rf \
  --save_predictions
```

### 4.2 Tier 1 extended-profile RF baseline

```bash
python3 scripts/run_baseline_models.py \
  --input_csv "outputs/clean/unsw_di_all_extended.csv" \
  --out_dir "outputs/baselines_dayholdout_extended_rf_only" \
  --split_mode day_holdout \
  --models rf \
  --save_predictions \
  --use_sparse_preprocessor \
  --top_n_requested_server_name 50
```

### 4.3 Tier 2 flow-only RF transfer

```bash
python3 scripts/run_transfer_unsw_to_ad.py \
  --train_csv "outputs/clean/unsw_di_all_flow_only.csv" \
  --test_csv "outputs/prepared/unsw_ad_intersection_flow_only.csv" \
  --out_dir "outputs/transfer/unsw_di_to_ad_flow_only_rf_only" \
  --models rf \
  --max_test_per_class 50000
```

### 4.4 Tier 2 extended-profile RF transfer

```bash
python3 scripts/run_transfer_unsw_to_ad.py \
  --train_csv "outputs/clean/unsw_di_all_extended.csv" \
  --test_csv "outputs/prepared/unsw_ad_intersection_extended.csv" \
  --out_dir "outputs/transfer/unsw_di_to_ad_extended_rf_only" \
  --models rf \
  --max_test_per_class 50000 \
  --use_sparse_preprocessor \
  --top_n_requested_server_name 50
```

---

## 5. Plotting commands

### 5.1 Pipeline diagram conversion

```bash
python3 - <<'PY'
from PIL import Image
Image.open("pipelinediag3.png").convert("RGB").save("dissertation/figures/pipeline_diagram.pdf")
print("saved dissertation/figures/pipeline_diagram.pdf")
PY
```

### 5.2 Class distribution

```bash
python3 scripts/plot_class_distribution.py \
  --input_csv "outputs/clean/unsw_di_all_flow_plus_app.csv" \
  --label_col device \
  --out_png "figures/class_distribution.png" \
  --out_pdf "dissertation/figures/class_distribution.pdf" \
  --out_counts_csv "outputs/tables/class_distribution_counts.csv"
```

### 5.3 Tier 1 baseline comparison

```bash
python3 scripts/plot_baseline_results.py \
  --summary_csvs \
    "outputs/baselines_random_flow_plus_app_final/random_stratified/baseline_summary.csv" \
    "outputs/baselines_dayholdout_flow_plus_app_final/day_holdout/baseline_summary.csv" \
  --tags random_stratified day_holdout \
  --metric macro_f1 \
  --out_png "figures/baseline_comparison.png" \
  --out_pdf "dissertation/figures/baseline_comparison.pdf" \
  --out_combined_csv "outputs/tables/tier1_combined_summary.csv" \
  --out_latex "outputs/tables/tier1_combined_summary.tex"
```

### 5.4 Temporal decay plot

```bash
python3 scripts/plot_temporal_decay.py \
  --decay_csvs "outputs/temporal/temporal_decay_flow_plus_app.csv" \
  --labels "flow_plus_app" \
  --out_png "figures/temporal_decay.png" \
  --out_pdf "dissertation/figures/temporal_decay.pdf"
```

### 5.5 RF confusion matrices

Using RF throughout the main-body confusion analysis gives a consistent across-tier comparison and aligns with the RF-based importance/ablation analyses.

```bash
python3 scripts/plot_confusion_matrix.py \
  --input_csv "outputs/baselines_dayholdout_flow_plus_app_final/day_holdout/rf_confusion_matrix.csv" \
  --normalize_rows \
  --out_png "figures/confusion_tier1_rf.png" \
  --out_pdf "dissertation/figures/confusion_tier1_rf.pdf"

python3 scripts/plot_confusion_matrix.py \
  --input_csv "outputs/transfer/unsw_di_to_ad_flow_plus_app/rf_confusion_matrix.csv" \
  --normalize_rows \
  --out_png "figures/confusion_tier2_rf.png" \
  --out_pdf "dissertation/figures/confusion_tier2_rf.pdf"

python3 scripts/plot_confusion_matrix.py \
  --input_csv "outputs/transfer/unsw_di_to_yourthings_device_flow_plus_app/rf_confusion_matrix.csv" \
  --normalize_rows \
  --out_png "figures/confusion_tier3_device_rf.png" \
  --out_pdf "dissertation/figures/confusion_tier3_device_rf.pdf"

python3 scripts/plot_confusion_matrix.py \
  --input_csv "outputs/transfer/unsw_di_to_yourthings_category_flow_plus_app/rf_confusion_matrix.csv" \
  --normalize_rows \
  --out_png "figures/confusion_tier3_category_rf.png" \
  --out_pdf "dissertation/figures/confusion_tier3_category_rf.pdf"
```

### 5.6 Permutation-importance plots

```bash
python3 scripts/plot_permutation_importance.py \
  --input_csv "outputs/importance/importance_within.csv" \
  --top_n 20 \
  --out_png "figures/importance_within.png" \
  --out_pdf "dissertation/figures/importance_within.pdf"

python3 scripts/plot_permutation_importance.py \
  --input_csv "outputs/importance/importance_cross_ad.csv" \
  --top_n 20 \
  --out_png "figures/importance_cross_ad.png" \
  --out_pdf "dissertation/figures/importance_cross_ad.pdf"

python3 scripts/plot_permutation_importance.py \
  --input_csv "outputs/importance/importance_cross_yourthings.csv" \
  --top_n 20 \
  --out_png "figures/importance_cross_yourthings.png" \
  --out_pdf "dissertation/figures/importance_cross_yourthings.pdf"
```

### 5.7 Feature-family ablation plots

```bash
python3 scripts/plot_feature_ablation.py \
  --input_csv "outputs/ablation/ablation_within_flow_plus_app_rf_only/ablation_summary.csv" \
  --out_png "figures/ablation_within_rf.png" \
  --out_pdf "dissertation/figures/ablation_within_rf.pdf"

python3 scripts/plot_feature_ablation.py \
  --input_csv "outputs/ablation/ablation_ad_flow_plus_app_rf_only/ablation_summary.csv" \
  --out_png "figures/ablation_ad_rf.png" \
  --out_pdf "dissertation/figures/ablation_ad_rf.pdf"

python3 scripts/plot_feature_ablation.py \
  --input_csv "outputs/ablation/ablation_yourthings_device_flow_plus_app_rf_only/ablation_summary.csv" \
  --out_png "figures/ablation_yourthings_device_rf.png" \
  --out_pdf "dissertation/figures/ablation_yourthings_device_rf.pdf"
```

Optional category-level ablation plot:

```bash
python3 scripts/plot_feature_ablation.py \
  --input_csv "outputs/ablation/ablation_yourthings_category_flow_plus_app_rf_only/ablation_summary.csv" \
  --out_png "figures/ablation_yourthings_category_rf.png" \
  --out_pdf "dissertation/figures/ablation_yourthings_category_rf.pdf"
```

### 5.8 RF-only profile comparison plots

```bash
python3 scripts/plot_profile_comparison.py \
  --summary_csvs \
    "outputs/baselines_dayholdout_flow_only_rf_only/day_holdout/baseline_summary.csv" \
    "outputs/baselines_dayholdout_flow_plus_app_final/day_holdout/baseline_summary.csv" \
    "outputs/baselines_dayholdout_extended_rf_only/day_holdout/baseline_summary.csv" \
  --tags flow_only flow_plus_app extended \
  --metric macro_f1 \
  --models rf \
  --out_png "figures/profile_compare_tier1_rf_only.png" \
  --out_pdf "dissertation/figures/profile_compare_tier1_rf_only.pdf" \
  --out_csv "outputs/tables/profile_compare_tier1_rf_only.csv"

python3 scripts/plot_profile_comparison.py \
  --summary_csvs \
    "outputs/transfer/unsw_di_to_ad_flow_only_rf_only/transfer_summary.csv" \
    "outputs/transfer/unsw_di_to_ad_flow_plus_app/transfer_summary.csv" \
    "outputs/transfer/unsw_di_to_ad_extended_rf_only/transfer_summary.csv" \
  --tags flow_only flow_plus_app extended \
  --metric macro_f1 \
  --models rf \
  --out_png "figures/profile_compare_tier2_ad_rf_only.png" \
  --out_pdf "dissertation/figures/profile_compare_tier2_ad_rf_only.pdf" \
  --out_csv "outputs/tables/profile_compare_tier2_ad_rf_only.csv"
```

---

## 6. Results-table commands

```bash
python3 scripts/make_results_tables.py \
  --summary_csvs \
    "outputs/baselines_random_flow_plus_app_final/random_stratified/baseline_summary.csv" \
    "outputs/baselines_dayholdout_flow_plus_app_final/day_holdout/baseline_summary.csv" \
  --tags random_stratified day_holdout \
  --out_csv "outputs/tables/tier1_results_table.csv" \
  --out_latex "outputs/tables/tier1_results_table.tex"

python3 scripts/make_results_tables.py \
  --summary_csvs "outputs/transfer/unsw_di_to_ad_flow_plus_app/transfer_summary.csv" \
  --tags tier2_ad \
  --out_csv "outputs/tables/tier2_results_table.csv" \
  --out_latex "outputs/tables/tier2_results_table.tex"

python3 scripts/make_results_tables.py \
  --summary_csvs "outputs/transfer/unsw_di_to_yourthings_device_flow_plus_app/transfer_summary.csv" \
  --tags tier3_device \
  --out_csv "outputs/tables/tier3_device_results_table.csv" \
  --out_latex "outputs/tables/tier3_device_results_table.tex"

python3 scripts/make_results_tables.py \
  --summary_csvs "outputs/transfer/unsw_di_to_yourthings_category_flow_plus_app/transfer_summary.csv" \
  --tags tier3_category \
  --out_csv "outputs/tables/tier3_category_results_table.csv" \
  --out_latex "outputs/tables/tier3_category_results_table.tex"
```

Profile-comparison tables:

```bash
python3 scripts/make_results_tables.py \
  --summary_csvs \
    "outputs/baselines_dayholdout_flow_only_rf_only/day_holdout/baseline_summary.csv" \
    "outputs/baselines_dayholdout_flow_plus_app_final/day_holdout/baseline_summary.csv" \
    "outputs/baselines_dayholdout_extended_rf_only/day_holdout/baseline_summary.csv" \
  --tags flow_only flow_plus_app extended \
  --out_csv "outputs/tables/profile_tier1_table.csv" \
  --out_latex "outputs/tables/profile_tier1_table.tex"

python3 scripts/make_results_tables.py \
  --summary_csvs \
    "outputs/transfer/unsw_di_to_ad_flow_only_rf_only/transfer_summary.csv" \
    "outputs/transfer/unsw_di_to_ad_flow_plus_app/transfer_summary.csv" \
    "outputs/transfer/unsw_di_to_ad_extended_rf_only/transfer_summary.csv" \
  --tags flow_only flow_plus_app extended \
  --out_csv "outputs/tables/profile_tier2_table.csv" \
  --out_latex "outputs/tables/profile_tier2_table.tex"
```

---

## 7. Helper commands for manual report tables and appendices

### 7.1 Dataset preparation summary helper

Creates a compact dataset-construction table.

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

spec = [
    ("UNSW-DI (clean, flow_plus_app)", "outputs/clean/unsw_di_all_flow_plus_app.csv", "device", "Tier 1 source / train"),
    ("UNSW-AD (clean, flow_plus_app)", "outputs/clean/unsw_ad_all_flow_plus_app.csv", "device", "Raw external source"),
    ("UNSW-AD intersection (flow_plus_app)", "outputs/prepared/unsw_ad_intersection_flow_plus_app.csv", "device", "DI∩AD overlap before cap"),
    ("UNSW-AD capped intersection (flow_plus_app)", "outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv", "device", "Tier 2 evaluation set"),
    ("YourThings (clean, flow_plus_app)", "outputs/clean/yourthings_all_flow_plus_app.csv", "device", "Raw external source"),
    ("YourThings prepared (flow_plus_app)", "outputs/prepared/yourthings_prepared_flow_plus_app.csv", "device", "Mapped external source"),
    ("YourThings device-overlap (flow_plus_app)", "outputs/prepared/yourthings_device_overlap_flow_plus_app.csv", "device", "Device-overlap subset before cap"),
    ("YourThings category-overlap (flow_plus_app)", "outputs/prepared/yourthings_category_overlap_flow_plus_app.csv", "category", "Category-overlap subset before cap"),
]

rows = []
for name, path, label_col, role in spec:
    df = pd.read_csv(path, low_memory=False)
    rows.append({
        "dataset": name,
        "path": path,
        "role": role,
        "rows": len(df),
        "label_col": label_col,
        "n_labels": df[label_col].nunique() if label_col in df.columns else None,
        "n_columns": df.shape[1],
    })

out = pd.DataFrame(rows)
out.to_csv("outputs/tables/dataset_preparation_summary_helper.csv", index=False)
print(out)
PY
```

### 7.2 Effective evaluation subset helper

Mirrors the actual transfer-script filtering logic for the evaluated subsets.

```bash
python3 - <<'PY'
import sys
import pandas as pd
from pathlib import Path

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

sys.path.append("scripts")
from run_transfer_unsw_to_yourthings import CATEGORY_MAP_UNSW

MAX_PER_CLASS = 50000

di = pd.read_csv("outputs/clean/unsw_di_all_flow_plus_app.csv", low_memory=False)
ad_capped = pd.read_csv("outputs/prepared/unsw_ad_intersection_flow_plus_app_capped.csv", low_memory=False)
yt = pd.read_csv("outputs/prepared/yourthings_prepared_flow_plus_app.csv", low_memory=False)

if "device_canonical" in yt.columns:
    if "device_raw" not in yt.columns:
        yt["device_raw"] = yt["device"]
    yt["device"] = yt["device_canonical"]

# Device-level logic: keep only labels seen in DI.
di_device_labels = set(di["device"].astype(str).unique())
yt_device = yt[yt["device"].astype(str).isin(di_device_labels)].copy()
yt_device_rows_before_cap = len(yt_device)
yt_device_rows_after_cap = int(yt_device.groupby("device").size().clip(upper=MAX_PER_CLASS).sum())

# Category-level logic: map DI devices to categories, drop unmapped/unseen categories.
train_cat = di.copy()
test_cat = yt.copy()
train_cat["category"] = train_cat["device"].astype(str).map(CATEGORY_MAP_UNSW)
train_cat = train_cat[train_cat["category"].notna()].copy()
test_cat = test_cat[test_cat["category"].notna()].copy()
seen_categories = set(train_cat["category"].astype(str).unique())
test_cat = test_cat[test_cat["category"].astype(str).isin(seen_categories)].copy()
yt_cat_rows_before_cap = len(test_cat)
yt_cat_rows_after_cap = int(test_cat.groupby("category").size().clip(upper=MAX_PER_CLASS).sum())

summary = pd.DataFrame([
    {
        "setting": "Tier 1 / UNSW-DI",
        "rows_used": len(di),
        "label_col": "device",
        "labels_used": di["device"].nunique(),
        "notes": "Random stratified and day-held-out splits drawn from same CSV",
    },
    {
        "setting": "Tier 2 / DI→AD",
        "rows_used": len(ad_capped),
        "label_col": "device",
        "labels_used": ad_capped["device"].nunique(),
        "notes": "Capped AD DI∩AD overlap set",
    },
    {
        "setting": "Tier 3 / DI→YourThings device",
        "rows_used": yt_device_rows_after_cap,
        "label_col": "device",
        "labels_used": yt_device["device"].nunique(),
        "notes": f"Canonicalised labels, filtered to DI-seen devices, capped at {MAX_PER_CLASS}/class (before cap: {yt_device_rows_before_cap:,})",
    },
    {
        "setting": "Tier 3 / DI→YourThings category",
        "rows_used": yt_cat_rows_after_cap,
        "label_col": "category",
        "labels_used": test_cat["category"].nunique(),
        "notes": f"Mapped with CATEGORY_MAP_UNSW, dropped unmapped/unseen categories, capped at {MAX_PER_CLASS}/class (before cap: {yt_cat_rows_before_cap:,})",
    },
])

summary.to_csv("outputs/tables/evaluation_subset_summary_helper.csv", index=False)
print(summary)
PY
```

### 7.3 UNSW-DI / UNSW-AD overlap helper

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

di = pd.read_csv("outputs/clean/unsw_di_all_flow_plus_app.csv", usecols=["device"])
ad = pd.read_csv("outputs/clean/unsw_ad_all_flow_plus_app.csv", usecols=["device"])
inter = pd.read_csv("mappings/unsw_intersection_devices_computed.csv")
di_only = pd.read_csv("mappings/unsw_di_only_devices_computed.csv")
ad_only = pd.read_csv("mappings/unsw_ad_only_devices_computed.csv")

summary = pd.DataFrame([
    {"set": "UNSW-DI all devices", "count": di["device"].nunique()},
    {"set": "UNSW-AD all devices", "count": ad["device"].nunique()},
    {"set": "DI∩AD intersection", "count": len(inter)},
    {"set": "DI only", "count": len(di_only)},
    {"set": "AD only", "count": len(ad_only)},
])

summary.to_csv("outputs/tables/unsw_overlap_summary.csv", index=False)
inter.to_csv("outputs/tables/unsw_intersection_devices_for_appendix.csv", index=False)
di_only.to_csv("outputs/tables/unsw_di_only_devices_for_appendix.csv", index=False)
ad_only.to_csv("outputs/tables/unsw_ad_only_devices_for_appendix.csv", index=False)
print(summary)
PY
```

### 7.4 YourThings mapping helper

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

mapping = pd.read_csv("outputs/prepared/yourthings_seen_mapping_flow_plus_app.csv")
prepared = pd.read_csv("outputs/prepared/yourthings_prepared_flow_plus_app.csv", low_memory=False)
device_overlap = pd.read_csv("outputs/prepared/yourthings_device_overlap_flow_plus_app.csv", low_memory=False)
category_overlap = pd.read_csv("outputs/prepared/yourthings_category_overlap_flow_plus_app.csv", low_memory=False)

mapping = mapping.sort_values(
    ["is_device_overlap", "is_category_overlap", "device_canonical", "device_raw"],
    ascending=[False, False, True, True]
)

mapping.to_csv("outputs/tables/yourthings_mapping_for_appendix.csv", index=False)

summary = pd.DataFrame([
    {"metric": "unique_raw_devices", "value": mapping["device_raw"].nunique()},
    {"metric": "unique_canonical_devices", "value": mapping["device_canonical"].nunique()},
    {"metric": "device_overlap_count", "value": int(mapping["is_device_overlap"].sum())},
    {"metric": "category_overlap_count", "value": int(mapping["is_category_overlap"].sum())},
    {"metric": "unique_categories", "value": mapping["category"].dropna().nunique()},
    {"metric": "prepared_rows", "value": len(prepared)},
    {"metric": "device_overlap_rows", "value": len(device_overlap)},
    {"metric": "category_overlap_rows", "value": len(category_overlap)},
])

summary.to_csv("outputs/tables/yourthings_mapping_summary.csv", index=False)
print(summary)
PY
```

### 7.5 Full feature inventory helper

```bash
python3 - <<'PY'
import sys
from pathlib import Path
import pandas as pd

Path("outputs/tables").mkdir(parents=True, exist_ok=True)

sys.path.append("scripts")
from run_feature_ablation import FAMILIES

profiles = {
    "flow_only": "outputs/clean/unsw_di_all_flow_only.csv",
    "flow_plus_app": "outputs/clean/unsw_di_all_flow_plus_app.csv",
    "extended": "outputs/clean/unsw_di_all_extended.csv",
}

summary_rows = []

for profile_name, path in profiles.items():
    df = pd.read_csv(path, nrows=5, low_memory=False)

    rows = []
    for col in df.columns:
        fam = "metadata_or_label"
        for k, cols in FAMILIES.items():
            if col in cols:
                fam = k
                break
        rows.append({
            "profile": profile_name,
            "feature": col,
            "dtype": str(df[col].dtype),
            "family": fam,
            "is_model_feature": fam != "metadata_or_label",
        })

    out = pd.DataFrame(rows)
    out.to_csv(f"outputs/tables/feature_inventory_{profile_name}.csv", index=False)

    family_counts = (
        out.groupby("family")
        .size()
        .reset_index(name="count")
        .assign(profile=profile_name)
    )
    family_counts.to_csv(f"outputs/tables/feature_family_counts_{profile_name}.csv", index=False)

    summary_rows.append({
        "profile": profile_name,
        "total_columns": len(out),
        "model_features": int(out["is_model_feature"].sum()),
        "metadata_or_label_columns": int((~out["is_model_feature"]).sum()),
    })

pd.DataFrame(summary_rows).to_csv("outputs/tables/feature_profile_summary.csv", index=False)
print("Saved feature inventory, family count, and profile summary CSVs")
PY
```
