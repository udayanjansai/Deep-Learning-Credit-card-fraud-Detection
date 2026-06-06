"""
Positional Encoding for transaction order (Task 5).
Based on Vaswani et al., 2017 — injects order information before attention.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def get_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding matrix.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    positions = np.arange(seq_len)[:, np.newaxis]
    div_term = np.power(10000.0, np.arange(0, d_model, 2) / d_model)

    pe[:, 0::2] = np.sin(positions / div_term)
    pe[:, 1::2] = np.cos(positions / div_term)
    return pe


def visualize_pe_heatmap(
    seq_len: int = 4,
    d_model: int = 32,
    title: str = "Transaction Positional Encoding",
    save_path: Optional[str] = None,
) -> plt.Figure:
    pe = get_positional_encoding(seq_len, d_model)
    labels = [f"Txn{i + 1}" for i in range(seq_len)]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pe, cmap="RdYlBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Encoding Dimension")
    ax.set_ylabel("Transaction Position")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
