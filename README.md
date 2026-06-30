# Cross-Dataset Generalisation of Flow-Based IoT Device Identification

BSc dissertation, Cardiff University, 2026. Supervised by Dr George Theodorakopoulos. Graded 81%.

## What this is

An empirical evaluation of whether ML models trained to identify IoT devices from network traffic actually learn stable device signatures, or just memorise environment-specific artefacts.

**Key finding:** models achieving 0.72 macro F1 within-dataset collapsed to 0.13 under cross-dataset transfer, showing that standard flow features act as fragile environment proxies rather than stable device signatures.

## Approach

- Extracted flow-level features from three public datasets (UNSW-DI, UNSW-AD, YourThings) using NFStream
- Trained 6 classical classifiers (Random Forest, Gradient Boosting, k-NN, Logistic Regression, Decision Tree, Gaussian Naive Bayes) and a 1D-CNN baseline
- Used GroupShuffleSplit to prevent data leakage between flows from the same capture session
- Designed a three-tier evaluation framework:
  - **Tier 1:** Within-dataset (same environment, held-out test split)
  - **Tier 2:** Cross-environment (UNSW-DI → UNSW-AD, same lab, different time)
  - **Tier 3:** Cross-dataset (UNSW-DI → YourThings, completely different network)
- Permutation importance analysis and feature-family ablation to understand what models actually learn

## Repository structure

```
├── dissertation.pdf          # Full dissertation report
├── REPRODUCIBILITY.md        # Command log for reproducing experiments
├── requirements.txt          # Python dependencies
├── scripts/                  # All pipeline code
│   ├── extract_unsw_nfstream.py
│   ├── prepare_*_dataset.py
│   ├── run_baseline_models.py
│   ├── run_transfer_*.py
│   ├── run_temporal_decay.py
│   ├── run_permutation_importance.py
│   ├── run_feature_ablation.py
│   └── plot_*.py
├── mappings/                 # Device-to-MAC/IP mapping files
├── figures/                  # Generated figures used in the report
└── outputs/                  # Experiment results, tables, and summaries
```

## Datasets

Raw datasets are not included. Obtain them from the original providers:

- [UNSW-DI Traffic Traces](https://iotanalytics.unsw.edu.au/iottraces.html)
- [UNSW-AD (benign traffic from attack dataset)](https://iotanalytics.unsw.edu.au/attack-data.html)
- [YourThings Dataset](https://yourthings.info/data/)

## Running

```bash
pip install -r requirements.txt
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full command sequence used to run extraction, experiments, and plotting.

## Citation

If you find this work useful:

```
Darsh Kanjani, "Cross-Dataset Generalisation of Flow-Based IoT Device
Identification: An Empirical Three-Tier Evaluation," BSc dissertation,
Cardiff University, 2026.
```

## Licence

This repository contains academic work. The code is released under the [MIT Licence](LICENCE). The dissertation PDF is provided for reference only.
