"""
Training pipeline — compares Dense, LSTM, LSTM+Attention, LSTM+PE+Attention.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models import MODEL_BUILDERS, build_attention_extractor, TransactionPositionalEncoding
from src.preprocessing import compute_class_weights, prepare_datasets

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "creditcard.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"


def get_callbacks(model_name: str) -> List:
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / f"{model_name}_best.keras"),
            monitor="val_auc",
            save_best_only=True,
            mode="max",
        ),
    ]


def evaluate_model(model, X_test, y_test, threshold: float = 0.5) -> Dict:
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }


def train_single_model(
    model_name: str,
    data: Dict,
    epochs: int = 15,
    batch_size: int = 256,
) -> Dict:
    input_len = data["input_len"]
    n_features = data["n_features"]

    builder = MODEL_BUILDERS[model_name]
    model = builder(input_len, n_features)

    class_weights = compute_class_weights(data["y_train"])

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name),
        verbose=1,
    )

    metrics = evaluate_model(model, data["X_test"], data["y_test"])
    model.save(str(MODELS_DIR / f"{model_name}.keras"))

    return {
        "model_name": model_name,
        "history": {k: [float(v) for v in vals] for k, vals in history.history.items()},
        "metrics": metrics,
    }


def train_all(
    max_samples: int = 50000,
    epochs: int = 15,
    batch_size: int = 256,
) -> Dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing sequential datasets...")
    data = prepare_datasets(DATA_PATH, max_samples=max_samples)

    # Save scaler for inference
    import joblib
    joblib.dump(data["scaler"], MODELS_DIR / "scaler.pkl")
    joblib.dump(
        {"feature_cols": data["feature_cols"], "input_len": data["input_len"]},
        MODELS_DIR / "metadata.pkl",
    )

    results = {}
    for model_name in MODEL_BUILDERS:
        print(f"\n{'=' * 50}\nTraining {model_name}...\n{'=' * 50}")
        results[model_name] = train_single_model(
            model_name, data, epochs=epochs, batch_size=batch_size
        )

    # Train attention extractor for dashboard (Task 6 & 7)
    print(f"\n{'=' * 50}\nTraining attention extractor...\n{'=' * 50}")
    full_model, extractor = build_attention_extractor(data["input_len"], data["n_features"])
    class_weights = compute_class_weights(data["y_train"])
    full_model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=get_callbacks("lstm_pe_attention"),
        verbose=1,
    )
    full_model.save(str(MODELS_DIR / "lstm_pe_attention.keras"))
    extractor.save(str(MODELS_DIR / "attention_extractor.keras"))

    extractor_metrics = evaluate_model(full_model, data["X_test"], data["y_test"])
    results["attention_extractor"] = {"metrics": extractor_metrics}

    # Save comparison report
    comparison = {
        name: res["metrics"] for name, res in results.items() if "metrics" in res
    }
    with open(OUTPUTS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\nModel Comparison:")
    for name, m in comparison.items():
        print(
            f"  {name:20s} | AUC: {m['roc_auc']:.4f} | "
            f"F1: {m['f1']:.4f} | Recall: {m['recall']:.4f}"
        )

    return results


if __name__ == "__main__":
    train_all(max_samples=50000, epochs=10, batch_size=256)
