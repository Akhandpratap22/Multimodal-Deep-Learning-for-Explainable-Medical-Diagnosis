import tensorflow as tf
from tensorflow.keras import layers, Model

from src.config import (
    CLINICAL_INPUT_DIM,
    CLINICAL_EMBEDDING_DIM,
    DROPOUT_RATE,
)


def build_clinical_encoder(
    input_dim: int = CLINICAL_INPUT_DIM,
    embedding_dim: int = CLINICAL_EMBEDDING_DIM,
    dropout_rate: float = DROPOUT_RATE,
) -> Model:
    """
    Builds the clinical encoder g_phi(c_i) as defined in Section III-D, Eq. (3).
    A Multi-Layer Perceptron (MLP) mapping structured clinical variables + missingness mask
    [c_tilde, m] to a learned representation z_c.
    
    Returns:
        Model with inputs=clinical_input, outputs=clinical_embedding (z_c)
    """
    inputs = layers.Input(shape=(input_dim,), name="clinical_input")
    
    # Layer 1: Expand and learn non-linear patient interaction
    x = layers.Dense(128, activation="relu", name="clinical_dense_1")(inputs)
    x = layers.BatchNormalization(name="clinical_bn_1")(x)
    x = layers.Dropout(dropout_rate, name="clinical_drop_1")(x)
    
    # Layer 2: Compress to clinical embedding z_c
    x = layers.Dense(embedding_dim, activation="relu", name="clinical_dense_2")(x)
    x = layers.BatchNormalization(name="clinical_bn_2")(x)
    outputs = layers.Dropout(dropout_rate * 0.5, name="clinical_drop_2")(x)

    encoder = Model(inputs=inputs, outputs=outputs, name="clinical_encoder_mlp")
    return encoder
