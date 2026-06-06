"""
Fraud detection model architectures (Task 4 & 5).

Model A: Dense Network
Model B: LSTM
Model C: LSTM + Attention
Model D: LSTM + Positional Encoding + Attention
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


class TransactionPositionalEncoding(layers.Layer):
    """Adds sinusoidal positional encoding to transaction feature sequences."""

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

    def _build_pe(self, seq_len: int) -> tf.Tensor:
        positions = np.arange(seq_len)[:, np.newaxis]
        dims = np.arange(0, self.d_model, 2)
        div_term = np.power(10000.0, dims / self.d_model)
        pe = np.zeros((seq_len, self.d_model))
        pe[:, 0::2] = np.sin(positions / div_term)
        pe[:, 1::2] = np.cos(positions / div_term)
        return tf.cast(pe[np.newaxis, :, :], tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        pe = self._build_pe(self.max_len)[:, :seq_len, :]
        return x + pe


def _compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_dense_model(
    input_len: int,
    n_features: int,
    hidden_units: int = 128,
) -> keras.Model:
    """
    Model A: Flatten sequence → Dense layers → fraud probability.
    """
    inputs = keras.Input(shape=(input_len, n_features), name="transactions")
    x = layers.Flatten()(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(hidden_units // 2, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    model = keras.Model(inputs, outputs, name="dense_fraud_model")
    return _compile_model(model)


def build_lstm_model(
    input_len: int,
    n_features: int,
    lstm_units: int = 64,
) -> keras.Model:
    """
    Model B: LSTM captures temporal dependencies across transactions.
    """
    inputs = keras.Input(shape=(input_len, n_features), name="transactions")
    x = layers.LSTM(lstm_units, return_sequences=False)(inputs)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    model = keras.Model(inputs, outputs, name="lstm_fraud_model")
    return _compile_model(model)


def build_lstm_attention_model(
    input_len: int,
    n_features: int,
    lstm_units: int = 64,
    num_heads: int = 4,
    use_positional_encoding: bool = False,
) -> keras.Model:
    """
    Model C: LSTM + MultiHeadAttention
    Model D: LSTM + Positional Encoding + MultiHeadAttention (Task 5)
    """
    inputs = keras.Input(shape=(input_len, n_features), name="transactions")

    x = layers.Dense(lstm_units, activation="relu")(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    if use_positional_encoding:
        x = TransactionPositionalEncoding(max_len=input_len, d_model=lstm_units, name="pos_encoding")(x)

    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=lstm_units // num_heads,
        name="transaction_attention",
    )(x, x)

    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    name = "lstm_pe_attention_model" if use_positional_encoding else "lstm_attention_model"
    model = keras.Model(inputs, outputs, name=name)
    return _compile_model(model, learning_rate=5e-4)


def build_attention_extractor(
    input_len: int,
    n_features: int,
    lstm_units: int = 64,
    num_heads: int = 4,
    use_positional_encoding: bool = True,
) -> Tuple[keras.Model, keras.Model]:
    """
    Returns (full_model, extractor) where extractor outputs (prediction, attention_weights).
    Used for Task 6 investigation and dashboard visualization.
    """
    inputs = keras.Input(shape=(input_len, n_features), name="transactions")

    x = layers.Dense(lstm_units, activation="relu")(inputs)
    x = layers.LSTM(lstm_units, return_sequences=True)(x)

    if use_positional_encoding:
        x = TransactionPositionalEncoding(max_len=input_len, d_model=lstm_units, name="pos_encoding")(x)

    mha = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=lstm_units // num_heads,
        name="transaction_attention",
    )
    attn_output, attn_weights = mha(x, x, return_attention_scores=True)

    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="fraud_prob")(x)

    full_model = keras.Model(inputs, outputs, name="lstm_pe_attention_model")
    _compile_model(full_model, learning_rate=5e-4)

    extractor = keras.Model(inputs, [outputs, attn_weights], name="attention_extractor")
    return full_model, extractor


MODEL_BUILDERS = {
    "dense": build_dense_model,
    "lstm": build_lstm_model,
    "lstm_attention": lambda il, nf: build_lstm_attention_model(il, nf, use_positional_encoding=False),
    "lstm_pe_attention": lambda il, nf: build_lstm_attention_model(il, nf, use_positional_encoding=True),
}
