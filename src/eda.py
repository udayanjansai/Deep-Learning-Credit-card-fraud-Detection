"""
Exploratory Data Analysis (Task 2).
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def compute_class_stats(df: pd.DataFrame, target_col: str = "Class") -> Dict:
    """Fraud %, legitimate %, and imbalance ratio."""
    counts = df[target_col].value_counts().sort_index()
    total = len(df)
    fraud_count = int(counts.get(1, 0))
    legit_count = int(counts.get(0, 0))

    fraud_pct = fraud_count / total * 100
    legit_pct = legit_count / total * 100
    imbalance_ratio = legit_count / max(fraud_count, 1)

    return {
        "total_transactions": total,
        "fraud_count": fraud_count,
        "legit_count": legit_count,
        "fraud_pct": fraud_pct,
        "legit_pct": legit_pct,
        "imbalance_ratio": imbalance_ratio,
    }


def plot_amount_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Transaction amount distribution — overall and by class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["Amount"], bins=50, color="#4d94ff", edgecolor="white", alpha=0.85)
    axes[0].set_title("Transaction Amount Distribution (All)")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Count")

    fraud_amt = df[df["Class"] == 1]["Amount"]
    legit_amt = df[df["Class"] == 0]["Amount"]
    axes[1].hist(legit_amt, bins=50, alpha=0.6, label="Legitimate", color="#6bcb77")
    axes[1].hist(fraud_amt, bins=50, alpha=0.8, label="Fraud", color="#ff6b6b")
    axes[1].set_title("Amount: Fraud vs Non-Fraud")
    axes[1].set_xlabel("Amount")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Correlation heatmap of Amount, Time, and PCA features."""
    cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=False,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_balance(
    stats: Dict,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels = ["Legitimate", "Fraud"]
    sizes = [stats["legit_count"], stats["fraud_count"]]
    colors = ["#6bcb77", "#ff6b6b"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.3f%%", startangle=90)
    axes[0].set_title("Class Distribution")

    axes[1].bar(labels, sizes, color=colors, edgecolor="white")
    axes[1].set_title(f"Imbalance Ratio: {stats['imbalance_ratio']:.1f}:1")
    axes[1].set_ylabel("Transaction Count")
    for i, v in enumerate(sizes):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_full_eda(
    data_path: str | Path,
    output_dir: str | Path = "outputs",
) -> Dict:
    """Run all EDA steps and save plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    stats = compute_class_stats(df)

    plot_class_balance(stats, save_path=str(output_dir / "class_balance.png"))
    plot_amount_distribution(df, save_path=str(output_dir / "amount_distribution.png"))
    plot_correlation_heatmap(df, save_path=str(output_dir / "correlation_heatmap.png"))

    return {"stats": stats, "dataframe": df}


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    result = run_full_eda(root / "creditcard.csv", root / "outputs")
    s = result["stats"]
    print(f"Total transactions : {s['total_transactions']:,}")
    print(f"Fraud              : {s['fraud_pct']:.4f}% ({s['fraud_count']:,})")
    print(f"Legitimate         : {s['legit_pct']:.4f}% ({s['legit_count']:,})")
    print(f"Imbalance ratio    : {s['imbalance_ratio']:.1f}:1")
    print(f"Plots saved to {root / 'outputs'}")
