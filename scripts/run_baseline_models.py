#!/usr/bin/env python3
# scripts/run_baseline_models.py

"""
Run within-dataset IoT device-identification baselines on a cleaned flow CSV.

The script supports two evaluation modes:

    - random_stratified:
        Random stratified train/test split by device label.

    - day_holdout:
        Group-based split using source_file so that complete capture files are
        held out for testing.

The script trains selected classical models, optionally trains a 1D-CNN baseline,
and writes summary metrics, classification reports, confusion matrices, split
metadata, and optional prediction files.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


try:
    import tensorflow as tf
    from keras import Sequential
    from keras.callbacks import EarlyStopping
    from keras.layers import (
        BatchNormalization,
        Conv1D,
        Dense,
        Dropout,
        GlobalAveragePooling1D,
        ReLU,
    )
    from keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def log(msg: str) -> None:
    """Print a message immediately during long-running experiments."""
    print(msg, flush=True)


def format_seconds(seconds: float) -> str:
    """Return a compact human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    rem = seconds % 60
    return f"{minutes}m {rem:.1f}s"


class DenseGaussianNB(BaseEstimator, ClassifierMixin):
    """
    Pipeline-compatible wrapper for GaussianNB.

    GaussianNB is included through this wrapper so that it can be evaluated using
    the same interface as the other scikit-learn models.
    """

    def __init__(self):
        self.model = GaussianNB()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def set_global_seed(seed: int) -> None:
    """Set random seeds for supported libraries."""
    random.seed(seed)
    np.random.seed(seed)

    if TF_AVAILABLE:
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass


def build_preprocessor(
    X: pd.DataFrame,
    sparse_output: bool = False,
) -> ColumnTransformer:
    """
    Build the shared preprocessing pipeline.

    Numeric columns:
        median imputation followed by standard scaling.

    Categorical columns:
        most-frequent imputation followed by one-hot encoding.

    Args:
        X: Training feature dataframe.
        sparse_output: If true, use sparse one-hot output to reduce memory use
            in high-cardinality feature profiles.

    Returns:
        A fitted-compatible ColumnTransformer.
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=not sparse_output)),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0 if sparse_output else 0.0,
    )


def make_classical_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Create the classical baseline model registry.

    The registry includes all implemented classical baselines. Linear SVM is
    retained as an implemented option, but it was omitted from the final
    reported experiments because the full extracted flow datasets made it less
    practical than the other evaluated baselines.

    HistGradientBoostingClassifier is used for the gradient-boosting baseline
    because it is substantially more scalable for the large extracted flow
    datasets used in this project.
    """
    return {
        "logreg": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
            verbose=1,
        ),
        "gnb": DenseGaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5),

        # Implemented but omitted from the final reported experiments due to
        # runtime/scalability constraints on the full flow datasets.
        "svm": LinearSVC(
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
            verbose=1,
            dual=False,
        ),

        "dt": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
        ),
        "rf": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=4,
            class_weight="balanced_subsample",
            verbose=1,
        ),
        "gb": HistGradientBoostingClassifier(
            max_iter=100,
            class_weight="balanced",
            random_state=random_state,
            verbose=1,
        ),
    }


def _check_label_coverage(
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[bool, List[str]]:
    """
    Check that every test label is present in the training split.

    Returns:
        Tuple of (coverage_ok, missing_test_labels).
    """
    train_labels = set(y_train.astype(str).unique().tolist())
    test_labels = set(y_test.astype(str).unique().tolist())
    missing = sorted(list(test_labels - train_labels))

    return len(missing) == 0, missing


def prepare_random_stratified_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Prepare a random stratified split by device label.
    """
    X = df.drop(columns=["device"])
    y = df["device"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    metadata = {
        "split_mode": "random_stratified",
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "note": (
            "Random stratified split by device label. "
            "This is an optimistic baseline because flows from the same capture "
            "period may appear in both training and test sets."
        ),
    }

    return X_train, X_test, y_train, y_test, metadata


def prepare_day_holdout_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    max_tries: int = 100,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Prepare a grouped split using source_file as the grouping variable.

    The function retries with different random seeds until every test label is
    also present in the training set, or until max_tries is reached.
    """
    if "source_file" not in df.columns:
        raise ValueError("day_holdout split requires a 'source_file' column.")

    X = df.drop(columns=["device"])
    y = df["device"]
    groups = df["source_file"].astype(str)

    last_missing = None

    for attempt in range(max_tries):
        seed = random_state + attempt

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        ok, missing = _check_label_coverage(y_train, y_test)
        last_missing = missing

        if ok:
            train_days = sorted(
                df.iloc[train_idx]["source_file"].astype(str).unique().tolist()
            )
            test_days = sorted(
                df.iloc[test_idx]["source_file"].astype(str).unique().tolist()
            )

            metadata = {
                "split_mode": "day_holdout",
                "test_size": test_size,
                "random_state": random_state,
                "actual_seed_used": seed,
                "attempt": attempt + 1,
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "train_days": train_days,
                "test_days": test_days,
                "note": (
                    "Complete PCAP/source_file groups are held out for testing. "
                    "The selected split is required to contain no test-only labels."
                ),
            }

            return X_train, X_test, y_train, y_test, metadata

    raise RuntimeError(
        "Could not find a valid day-held-out split where all test labels appear "
        f"in training. Last missing test-only labels: {last_missing}"
    )


def prepare_split(
    df: pd.DataFrame,
    split_mode: str,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Dispatch to the selected train/test split strategy.
    """
    if "device" not in df.columns:
        raise ValueError("Expected a 'device' column in the dataset.")

    if split_mode == "random_stratified":
        return prepare_random_stratified_split(df, test_size, random_state)

    if split_mode == "day_holdout":
        return prepare_day_holdout_split(df, test_size, random_state)

    raise ValueError(f"Unknown split_mode: {split_mode}")


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write a dictionary as an indented JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_predictions_csv(
    path: Path,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Write per-row true and predicted labels."""
    out = pd.DataFrame({
        "y_true": y_true.astype(str).values,
        "y_pred": pd.Series(y_pred).astype(str).values,
    })
    out.to_csv(path, index=False)


def cap_high_cardinality_columns(
    df: pd.DataFrame,
    top_n_requested_server_name: Optional[int] = None,
) -> pd.DataFrame:
    """
    Reduce high-cardinality categorical features before one-hot encoding.

    Currently supported:
        requested_server_name:
            retain the top-N most frequent values and map all other values to
            "other".

    This is primarily used for the extended feature profile.
    """
    df = df.copy()

    if (
        top_n_requested_server_name is not None
        and top_n_requested_server_name > 0
        and "requested_server_name" in df.columns
    ):
        col = "requested_server_name"
        series = df[col].astype("string").fillna("missing")

        top_values = (
            series.value_counts(dropna=False)
            .nlargest(top_n_requested_server_name)
            .index
        )
        df[col] = series.where(series.isin(top_values), "other")

        log(
            f"[cardinality] Capped '{col}' to top {top_n_requested_server_name} "
            f"values plus 'other'; new unique count = {df[col].nunique(dropna=False)}"
        )

    return df


def build_1d_cnn_model(
    input_length: int,
    num_classes: int,
    learning_rate: float = 1e-3,
):
    """
    Build the optional 1D-CNN baseline model.
    """
    model = Sequential([
        Conv1D(filters=32, kernel_size=3, padding="same", input_shape=(input_length, 1)),
        BatchNormalization(),
        ReLU(),

        Conv1D(filters=64, kernel_size=3, padding="same"),
        BatchNormalization(),
        ReLU(),

        Conv1D(filters=128, kernel_size=3, padding="same"),
        BatchNormalization(),
        ReLU(),

        GlobalAveragePooling1D(),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_classical_model(
    name: str,
    model: Any,
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: List[str],
    out_dir: Path,
    save_predictions: bool = False,
) -> Dict[str, Any]:
    """
    Fit and evaluate one classical model using a scikit-learn pipeline.
    """
    log(f"\n=== Training {name} ===")
    model_start = time.time()

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model),
    ])

    log(f"[{name}] fitting model...")
    fit_start = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - fit_start
    log(f"[{name}] fit complete in {format_seconds(fit_time)}")

    log(f"[{name}] predicting...")
    pred_start = time.time()
    y_pred = pipe.predict(X_test)
    pred_time = time.time() - pred_start
    log(f"[{name}] prediction complete in {format_seconds(pred_time)}")

    acc = accuracy_score(y_test, y_pred)

    test_labels = sorted(pd.Series(y_test).astype(str).unique().tolist())

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=test_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=test_labels,
        average="weighted",
        zero_division=0,
    )

    report_test_only = classification_report(
        y_test,
        y_pred,
        labels=test_labels,
        output_dict=True,
        zero_division=0,
    )
    save_json(out_dir / f"{name}_classification_report_test_only.json", report_test_only)

    report_all_seen = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    save_json(out_dir / f"{name}_classification_report_all_seen.json", report_all_seen)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / f"{name}_confusion_matrix.csv")

    cm_test_only = confusion_matrix(y_test, y_pred, labels=test_labels)
    cm_test_only_df = pd.DataFrame(cm_test_only, index=test_labels, columns=test_labels)
    cm_test_only_df.to_csv(out_dir / f"{name}_confusion_matrix_test_only.csv")

    if save_predictions:
        save_predictions_csv(out_dir / f"{name}_predictions.csv", y_test, y_pred)

    total_time = time.time() - model_start

    log(
        f"[{name}] done | accuracy={acc:.4f} | macro_f1={macro_f1:.4f} | "
        f"weighted_f1={weighted_f1:.4f} | total_time={format_seconds(total_time)}"
    )

    return {
        "model": name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fit_seconds": fit_time,
        "predict_seconds": pred_time,
        "total_seconds": total_time,
    }


def evaluate_1d_cnn(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    labels: List[str],
    out_dir: Path,
    random_state: int,
    epochs: int = 20,
    batch_size: int = 256,
    save_predictions: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Fit and evaluate the optional 1D-CNN baseline.
    """
    if not TF_AVAILABLE:
        log("\n[WARN] TensorFlow not available; skipping 1D-CNN baseline.")
        return None

    log("\n=== Training cnn1d ===")
    set_global_seed(random_state)
    model_start = time.time()

    log("[cnn1d] preprocessing train/test arrays...")
    pre_start = time.time()

    preprocessor = build_preprocessor(X_train)
    X_train_arr = preprocessor.fit_transform(X_train)
    X_test_arr = preprocessor.transform(X_test)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    y_test_enc = le.transform(y_test.astype(str))

    X_train_arr = np.asarray(X_train_arr, dtype=np.float32).reshape(
        X_train_arr.shape[0],
        X_train_arr.shape[1],
        1,
    )
    X_test_arr = np.asarray(X_test_arr, dtype=np.float32).reshape(
        X_test_arr.shape[0],
        X_test_arr.shape[1],
        1,
    )

    pre_time = time.time() - pre_start
    log(f"[cnn1d] preprocessing complete in {format_seconds(pre_time)}")
    log(f"[cnn1d] input shape = {X_train_arr.shape}, classes = {len(le.classes_)}")

    model = build_1d_cnn_model(
        input_length=X_train_arr.shape[1],
        num_classes=len(le.classes_),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    log("[cnn1d] fitting neural model...")
    fit_start = time.time()

    history = model.fit(
        X_train_arr,
        y_train_enc,
        validation_split=0.10,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    fit_time = time.time() - fit_start
    log(f"[cnn1d] fit complete in {format_seconds(fit_time)}")

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(out_dir / "cnn1d_history.csv", index=False)

    log("[cnn1d] predicting...")
    pred_start = time.time()

    y_prob = model.predict(X_test_arr, verbose=0)
    y_pred_enc = np.argmax(y_prob, axis=1)
    y_pred = le.inverse_transform(y_pred_enc)

    pred_time = time.time() - pred_start
    log(f"[cnn1d] prediction complete in {format_seconds(pred_time)}")

    acc = accuracy_score(y_test, y_pred)

    test_labels = sorted(pd.Series(y_test).astype(str).unique().tolist())

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=test_labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        labels=test_labels,
        average="weighted",
        zero_division=0,
    )

    report_test_only = classification_report(
        y_test,
        y_pred,
        labels=test_labels,
        output_dict=True,
        zero_division=0,
    )
    save_json(out_dir / "cnn1d_classification_report_test_only.json", report_test_only)

    report_all_seen = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    save_json(out_dir / "cnn1d_classification_report_all_seen.json", report_all_seen)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "cnn1d_confusion_matrix.csv")

    cm_test_only = confusion_matrix(y_test, y_pred, labels=test_labels)
    cm_test_only_df = pd.DataFrame(cm_test_only, index=test_labels, columns=test_labels)
    cm_test_only_df.to_csv(out_dir / "cnn1d_confusion_matrix_test_only.csv")

    if save_predictions:
        save_predictions_csv(out_dir / "cnn1d_predictions.csv", y_test, y_pred)

    total_time = time.time() - model_start

    log(
        f"[cnn1d] done | accuracy={acc:.4f} | macro_f1={macro_f1:.4f} | "
        f"weighted_f1={weighted_f1:.4f} | total_time={format_seconds(total_time)}"
    )

    return {
        "model": "cnn1d",
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "fit_seconds": fit_time,
        "predict_seconds": pred_time,
        "total_seconds": total_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run within-dataset IoT device-identification baselines."
    )
    parser.add_argument(
        "--input_csv",
        default="outputs/clean/unsw_di_all_flow_plus_app.csv",
        help="Path to the cleaned input CSV.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/baseline_results",
        help="Directory for result outputs.",
    )
    parser.add_argument(
        "--split_mode",
        choices=["random_stratified", "day_holdout"],
        default="random_stratified",
        help="Train/test split strategy.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.20,
        help="Fraction of rows or groups reserved for testing.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--include_cnn",
        action="store_true",
        help="Include the optional 1D-CNN baseline.",
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-row true and predicted labels.",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="Print available classical model keys and exit.",
    )
    parser.add_argument(
        "--use_sparse_preprocessor",
        action="store_true",
        help="Use sparse one-hot output to reduce memory usage.",
    )
    parser.add_argument(
        "--top_n_requested_server_name",
        type=int,
        default=None,
        help=(
            "If set, retain only the top-N requested_server_name values and "
            "map all remaining values to 'other'."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of classical models to run.",
    )

    args = parser.parse_args()

    set_global_seed(args.random_state)

    all_models = make_classical_models(random_state=args.random_state)

    if args.list_models:
        log("Available classical model keys:")
        for key in all_models.keys():
            log(f"  - {key}")

        if TF_AVAILABLE:
            log("Optional neural model:")
            log("  - cnn1d, enabled with --include_cnn")
        else:
            log("TensorFlow not available; cnn1d is unavailable.")

        return

    if args.models is None:
        selected_model_names = list(all_models.keys())
    else:
        invalid = [model_name for model_name in args.models if model_name not in all_models]

        if invalid:
            raise ValueError(
                f"Unknown model keys: {invalid}. "
                "Use --list_models to see valid keys."
            )

        selected_model_names = args.models

    selected_models = {
        model_name: all_models[model_name]
        for model_name in selected_model_names
    }

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir) / args.split_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    log(f"Loading dataset: {input_csv}")
    load_start = time.time()

    df = pd.read_csv(input_csv, low_memory=False)
    df = cap_high_cardinality_columns(
        df,
        top_n_requested_server_name=args.top_n_requested_server_name,
    )

    load_time = time.time() - load_start
    log(f"Loaded dataset in {format_seconds(load_time)} | shape={df.shape}")

    log(f"Preparing split: {args.split_mode}")
    split_start = time.time()

    X_train, X_test, y_train, y_test, split_metadata = prepare_split(
        df=df,
        split_mode=args.split_mode,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    split_time = time.time() - split_start

    log(
        f"Split ready in {format_seconds(split_time)} | "
        f"train={len(X_train)} | test={len(X_test)}"
    )

    train_labels = set(y_train.astype(str).unique().tolist())
    test_labels = set(y_test.astype(str).unique().tolist())
    missing_from_test = sorted(list(train_labels - test_labels))

    save_json(
        out_dir / "label_coverage.json",
        {
            "num_train_labels": len(train_labels),
            "num_test_labels": len(test_labels),
            "labels_missing_from_test": missing_from_test,
        },
    )

    if missing_from_test:
        log(
            f"[split] {len(missing_from_test)} train labels absent from test: "
            + ", ".join(missing_from_test[:10])
            + (" ..." if len(missing_from_test) > 10 else "")
        )

    for frame in (X_train, X_test):
        if "source_file" in frame.columns:
            frame.drop(columns=["source_file"], inplace=True)

    labels = sorted(df["device"].astype(str).unique().tolist())
    save_json(out_dir / "split_metadata.json", split_metadata)

    log("Building shared preprocessor...")
    prep_start = time.time()

    preprocessor = build_preprocessor(
        X_train,
        sparse_output=args.use_sparse_preprocessor,
    )

    prep_time = time.time() - prep_start
    log(f"Preprocessor built in {format_seconds(prep_time)}")

    summary_rows = []

    total_models_to_run = len(selected_models) + (1 if args.include_cnn else 0)
    current_idx = 0

    for name, model in selected_models.items():
        current_idx += 1
        log(f"\n[{current_idx}/{total_models_to_run}] Starting model: {name}")

        row = evaluate_classical_model(
            name=name,
            model=model,
            preprocessor=preprocessor,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels,
            out_dir=out_dir,
            save_predictions=args.save_predictions,
        )

        row["split_mode"] = args.split_mode
        summary_rows.append(row)

    if args.include_cnn:
        current_idx += 1
        log(f"\n[{current_idx}/{total_models_to_run}] Starting model: cnn1d")

        cnn_row = evaluate_1d_cnn(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels,
            out_dir=out_dir,
            random_state=args.random_state,
            save_predictions=args.save_predictions,
        )

        if cnn_row is not None:
            cnn_row["split_mode"] = args.split_mode
            summary_rows.append(cnn_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary_df.to_csv(out_dir / "baseline_summary.csv", index=False)

    log("\nSaved results to:")
    log(f"  {out_dir}")
    log("\nSummary:")
    log(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()