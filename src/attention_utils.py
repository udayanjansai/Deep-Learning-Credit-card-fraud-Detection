"""
Attention investigation utilities (Task 6).
Identifies which prior transaction most influenced the fraud prediction.
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def extract_attention_weights(model, X: np.ndarray) -> np.ndarray:
    """
    Run attention extractor model.
    Returns weights shape: (batch, heads, seq_len, seq_len) or (batch, seq_len, seq_len).
    """
    outputs = model.predict(X, verbose=0)
    if isinstance(outputs, list):
        return outputs[-1]
    return outputs


def get_transaction_importance(
    attention_weights: np.ndarray,
    head_idx: int = 0,
    strategy: str = "target_row",
) -> np.ndarray:
    """
    Aggregate attention into per-transaction importance scores.

    strategy='target_row': attention the prediction step pays to each prior txn
    strategy='mean_cols':  mean attention each position receives
    """
    weights = attention_weights
    if weights.ndim == 4:
        weights = weights[:, head_idx, :, :]

    if strategy == "target_row":
        # Last position attends to all prior transactions
        scores = weights[:, -1, :]
    else:
        scores = weights.mean(axis=1)

    # Normalize per sample
    row_min = scores.min(axis=1, keepdims=True)
    row_max = scores.max(axis=1, keepdims=True)
    denom = np.where(row_max - row_min > 0, row_max - row_min, 1.0)
    return (scores - row_min) / denom


def rank_influential_transactions(
    importance: np.ndarray,
    txn_labels: List[str] | None = None,
) -> List[Dict]:
    """Return ranked list of transaction positions by influence for one sample."""
    if txn_labels is None:
        txn_labels = [f"Txn{i + 1}" for i in range(len(importance))]

    ranked = sorted(
        zip(txn_labels, importance),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"transaction": name, "importance": float(score)} for name, score in ranked]


def visualize_attention_bar(
    importance: np.ndarray,
    txn_labels: List[str] | None = None,
    title: str = "Transaction Influence on Fraud Prediction",
    figsize: tuple = (8, 4),
) -> plt.Figure:
    if txn_labels is None:
        txn_labels = [f"Txn{i + 1}" for i in range(len(importance))]

    colors = plt.cm.Reds(importance / (importance.max() + 1e-8))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(txn_labels, importance, color=colors, edgecolor="none")
    ax.set_xlabel("Attention Weight")
    ax.set_title(title)
    ax.set_xlim(0, 1)

    for bar, score in zip(bars, importance):
        ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def visualize_attention_heatmap(
    attention_matrix: np.ndarray,
    txn_labels: List[str] | None = None,
    title: str = "Attention Matrix (Transaction × Transaction)",
    figsize: tuple = (7, 6),
) -> plt.Figure:
    n = attention_matrix.shape[0]
    if txn_labels is None:
        txn_labels = [f"Txn{i + 1}" for i in range(n)]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_matrix[:n, :n], cmap="magma", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(txn_labels, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(txn_labels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def get_top_influential_transaction(
    importance: np.ndarray,
    txn_labels: List[str] | None = None,
) -> Tuple[str, float]:
    """Return the single most influential transaction and its score."""
    ranked = rank_influential_transactions(importance, txn_labels)
    top = ranked[0]
    return top["transaction"], top["importance"]
