"""
Sequence generation for sequential fraud detection.
Groups time-ordered transactions into customer sessions, then builds
sliding windows: Txn1..Txn4 → predict Txn5 fraud label.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"
SEQ_LEN = 5          # 4 context txns + 1 target
INPUT_LEN = SEQ_LEN - 1
SESSION_GAP_SEC = 1800  # new customer session if gap > 30 min


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required = ["Time"] + FEATURE_COLS + [TARGET_COL]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    return df.sort_values("Time").reset_index(drop=True)


def assign_customer_sessions(df: pd.DataFrame, gap_sec: int = SESSION_GAP_SEC) -> pd.DataFrame:
    """Simulate customers by splitting on large time gaps between transactions."""
    df = df.copy()
    time_diff = df["Time"].diff().fillna(0)
    new_session = (time_diff > gap_sec).astype(int)
    df["CustomerID"] = new_session.cumsum()
    return df


def build_sequences(
    df: pd.DataFrame,
    seq_len: int = SEQ_LEN,
    per_customer: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, customer_ids) from transaction dataframe.

    X shape: (n_samples, input_len, n_features)
    y shape: (n_samples,) — fraud label of the target (last) transaction
    """
    sequences_x, sequences_y, cust_ids = [], [], []

    groups = df.groupby("CustomerID") if per_customer else [(0, df)]

    for cust_id, group in groups:
        group = group.reset_index(drop=True)
        if len(group) < seq_len:
            continue

        features = group[FEATURE_COLS].values.astype(np.float32)
        labels = group[TARGET_COL].values.astype(np.int32)

        for start in range(len(group) - seq_len + 1):
            window = features[start : start + seq_len]
            sequences_x.append(window[:-1])
            sequences_y.append(labels[start + seq_len - 1])
            cust_ids.append(cust_id)

    if not sequences_x:
        raise ValueError("No sequences generated. Check dataset size and seq_len.")

    return (
        np.array(sequences_x, dtype=np.float32),
        np.array(sequences_y, dtype=np.int32),
        np.array(cust_ids, dtype=np.int32),
    )


def scale_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on training data only, transform all splits."""
    n_features = X_train.shape[-1]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    def transform(X: np.ndarray) -> np.ndarray:
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_features)).reshape(shape).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def prepare_datasets(
    data_path: str | Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    max_samples: int | None = None,
) -> Dict:
    """
    Full pipeline: load → sessions → sequences → split → scale.
    Returns dict with arrays, scaler, and metadata.
    """
    df = load_raw_data(data_path)
    df = assign_customer_sessions(df)

    X, y, cust_ids = build_sequences(df)

    if max_samples and len(y) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(y), size=max_samples, replace=False)
        X, y, cust_ids = X[idx], y[idx], cust_ids[idx]

    X_train, X_temp, y_train, y_temp, c_train, c_temp = train_test_split(
        X, y, cust_ids, test_size=(test_size + val_size), random_state=random_state, stratify=y
    )
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test, c_val, c_test = train_test_split(
        X_temp, y_temp, c_temp, test_size=(1 - relative_val),
        random_state=random_state, stratify=y_temp,
    )

    X_train, X_val, X_test, scaler = scale_sequences(X_train, X_val, X_test)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "cust_train": c_train,
        "cust_val": c_val,
        "cust_test": c_test,
        "scaler": scaler,
        "n_features": X_train.shape[-1],
        "input_len": X_train.shape[1],
        "feature_cols": FEATURE_COLS,
        "raw_df": df,
    }

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Inverse-frequency class weights for imbalanced fraud data."""
    n_total = len(y)
    n_fraud = int(y.sum())
    n_legit = n_total - n_fraud
    if n_fraud == 0:
        return {0: 1.0, 1: 1.0}
    return {
        0: n_total / (2 * n_legit),
        1: n_total / (2 * n_fraud),
    }
