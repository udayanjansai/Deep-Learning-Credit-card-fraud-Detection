"""
Inference helpers for the Streamlit dashboard.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models import TransactionPositionalEncoding
from src.preprocessing import FEATURE_COLS, INPUT_LEN, SEQ_LEN, assign_customer_sessions, build_sequences
from src.attention_utils import get_transaction_importance, rank_influential_transactions

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

CUSTOM_OBJECTS = {"TransactionPositionalEncoding": TransactionPositionalEncoding}


def load_artifacts() -> Dict:
    """Load scaler, metadata, models from disk."""
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    metadata = joblib.load(MODELS_DIR / "metadata.pkl")

    models = {}
    for name in ["dense", "lstm", "lstm_attention", "lstm_pe_attention"]:
        path = MODELS_DIR / f"{name}.keras"
        if path.exists():
            models[name] = tf.keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)

    extractor = None
    extractor_path = MODELS_DIR / "attention_extractor.keras"
    if extractor_path.exists():
        extractor = tf.keras.models.load_model(extractor_path, custom_objects=CUSTOM_OBJECTS)

    return {
        "scaler": scaler,
        "metadata": metadata,
        "models": models,
        "extractor": extractor,
    }


def csv_to_sequences(df: pd.DataFrame, scaler) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convert uploaded CSV to model-ready sequences.
    Returns (X_scaled, meta_df with sequence indices).
    """
    required = ["Time"] + FEATURE_COLS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = df.copy()
    if "Class" not in work.columns:
        work["Class"] = 0

    work = work.sort_values("Time").reset_index(drop=True)
    work = assign_customer_sessions(work)

    X, y, cust_ids = build_sequences(work)

    n_features = X.shape[-1]
    X_scaled = scaler.transform(X.reshape(-1, n_features)).reshape(X.shape).astype(np.float32)

    meta = pd.DataFrame({
        "sequence_idx": range(len(y)),
        "customer_id": cust_ids,
        "actual_class": y,
        "target_txn_index": list(range(SEQ_LEN - 1, SEQ_LEN - 1 + len(y))),
    })

    return X_scaled, meta


def predict_fraud(
    model,
    X: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict(X, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)
    return probs, preds


def get_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low Risk"
    if probability < 0.70:
        return "Medium Risk"
    return "High Risk"


def analyze_attention(
    extractor,
    X: np.ndarray,
    sample_idx: int = 0,
) -> Dict:
    """Get attention weights and ranked influential transactions for one sequence."""
    sample = X[sample_idx : sample_idx + 1]

    # extractor outputs [fraud_prob, attention_weights] — handle list or tuple
    outputs = extractor.predict(sample, verbose=0)
    if isinstance(outputs, (list, tuple)):
        prob_raw, attn_weights = outputs[0], outputs[1]
    else:
        raise ValueError("Attention extractor must return [prob, attn_weights].")

    # Ensure numpy arrays
    prob_raw = np.array(prob_raw)
    attn_weights = np.array(attn_weights)

    # Extract scalar fraud probability robustly
    fraud_prob = float(prob_raw.flatten()[0])

    # attn_weights shape: (batch, heads, seq, seq) or (batch, seq, seq)
    importance = get_transaction_importance(attn_weights, strategy="target_row")[0]
    txn_labels = [f"Txn{i + 1}" for i in range(len(importance))]
    ranked = rank_influential_transactions(importance, txn_labels)

    # Build attention matrix for heatmap: pick head-0 for 4D, else use as-is
    if attn_weights.ndim == 4:
        attn_matrix = attn_weights[0, 0, :, :]   # (seq, seq)
    elif attn_weights.ndim == 3:
        attn_matrix = attn_weights[0, :, :]       # (seq, seq)
    else:
        attn_matrix = attn_weights                # fallback

    return {
        "fraud_probability": fraud_prob,
        "importance": importance,
        "txn_labels": txn_labels,
        "ranked": ranked,
        "attention_matrix": attn_matrix,
        "top_transaction": ranked[0]["transaction"],
        "top_score": ranked[0]["importance"],
    }


def build_single_sequence_from_history(
    history: List[Dict],
    scaler,
) -> np.ndarray:
    """Build one sequence from a list of prior transactions (for real-time sim)."""
    if len(history) < INPUT_LEN:
        raise ValueError(f"Need at least {INPUT_LEN} prior transactions.")

    recent = history[-INPUT_LEN:]
    features = np.array([[txn[col] for col in FEATURE_COLS] for txn in recent], dtype=np.float32)
    features = features[np.newaxis, :, :]

    n_features = features.shape[-1]
    return scaler.transform(features.reshape(-1, n_features)).reshape(features.shape).astype(np.float32)
