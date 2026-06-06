# Deep Learning Fraud Detection System

Sequential financial transaction fraud detection using Dense, LSTM, and LSTM + Attention models with positional encoding.

## Dataset

Place `creditcard.csv` (Kaggle Credit Card Fraud Detection) in the project root.

## Project Structure

```
DL Fraud Detection/
├── creditcard.csv          # Dataset
├── src/
│   ├── preprocessing.py    # Sequence generation (Txn1..4 → predict Txn5)
│   ├── models.py           # Dense, LSTM, LSTM+Attention, LSTM+PE+Attention
│   ├── positional_encoding.py
│   ├── attention_utils.py  # Attention investigation (Task 6)
│   ├── eda.py              # Exploratory analysis (Task 2)
│   ├── train.py            # Training pipeline (Task 4 & 5)
│   └── inference.py        # Dashboard inference helpers
├── app/
│   └── streamlit_app.py    # Fraud Intelligence Dashboard (Task 7)
├── notebooks/
│   └── fraud_detection_analysis.ipynb
├── models/                 # Saved models (after training)
└── outputs/                # EDA plots & metrics
```

## Quick Start

Uses [uv](https://docs.astral.sh/uv/) for dependency management (same as sibling projects in the parent folder).

```bash
# Install dependencies (creates .venv automatically)
uv sync

# Optional: include Jupyter for the notebook
uv sync --extra notebook

# Run EDA + train all models
uv run python -m src.eda
uv run python -m src.train

# Launch dashboard
uv run streamlit run app/streamlit_app.py
```

## Tasks Covered

| Task | Description | Location |
|------|-------------|----------|
| 1 | Business understanding | Notebook + Dashboard "Business Context" tab |
| 2 | EDA (fraud %, imbalance, visualizations) | `src/eda.py`, notebook |
| 3 | Sequence generation | `src/preprocessing.py` |
| 4 | Dense vs LSTM vs LSTM+Attention | `src/models.py`, `src/train.py` |
| 5 | Positional encoding experiment | `src/positional_encoding.py`, Model D |
| 6 | Attention investigation | `src/attention_utils.py`, dashboard |
| 7 | Streamlit dashboard | `app/streamlit_app.py` |
| Bonus | Real-time simulation | Dashboard "Real-Time Simulation" tab |

## Sequence Design

Transactions are sorted by time, grouped into customer sessions (30-min gap), and converted to sliding windows:

```
Txn1 → Txn2 → Txn3 → Txn4  →  Predict Txn5 fraud?
```

## Models

- **Model A** — Dense Network (flattened sequence)
- **Model B** — LSTM (temporal patterns)
- **Model C** — LSTM + MultiHeadAttention
- **Model D** — LSTM + Positional Encoding + Attention
