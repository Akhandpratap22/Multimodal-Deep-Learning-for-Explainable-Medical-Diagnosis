import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional

from src.config import (
    IMAGE_SHAPE,
    CLINICAL_INPUT_DIM,
    CLASSIFIER_HIDDEN_DIMS,
    DROPOUT_RATE,
    NUM_CLASSES,
    LEARNING_RATE,
)
from src.models.image_encoder import build_image_encoder
from src.models.clinical_encoder import build_clinical_encoder
from src.models.fusion import build_multimodal_fusion_model


def build_image_only_model(
    pretrained: bool = True,
    dropout_rate: float = DROPOUT_RATE,
    name: str = "image_only_model"
) -> Model:
    """Single-modality baseline predicting from chest radiographs alone (Section III-D)."""
    img_input = layers.Input(shape=IMAGE_SHAPE, name="image_input")
    img_encoder = build_image_encoder(pretrained=pretrained)
    z_i = img_encoder(img_input)

    x = z_i
    for idx, dim in enumerate(CLASSIFIER_HIDDEN_DIMS):
        x = layers.Dense(dim, activation="relu", name=f"img_clf_dense_{idx+1}")(x)
        x = layers.BatchNormalization(name=f"img_clf_bn_{idx+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"img_clf_drop_{idx+1}")(x)

    prediction = layers.Dense(NUM_CLASSES, activation="sigmoid", name="prediction_output")(x)
    return Model(inputs=img_input, outputs=prediction, name=name)


def build_clinical_only_model(
    dropout_rate: float = DROPOUT_RATE,
    name: str = "clinical_only_model"
) -> Model:
    """Single-modality baseline predicting from structured clinical variables alone (Section III-D)."""
    clin_input = layers.Input(shape=(CLINICAL_INPUT_DIM,), name="clinical_input")
    clin_encoder = build_clinical_encoder()
    z_c = clin_encoder(clin_input)

    x = z_c
    for idx, dim in enumerate(CLASSIFIER_HIDDEN_DIMS):
        x = layers.Dense(dim, activation="relu", name=f"clin_clf_dense_{idx+1}")(x)
        x = layers.BatchNormalization(name=f"clin_clf_bn_{idx+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"clin_clf_drop_{idx+1}")(x)

    prediction = layers.Dense(NUM_CLASSES, activation="sigmoid", name="prediction_output")(x)
    return Model(inputs=clin_input, outputs=prediction, name=name)


def build_model(
    model_type: str,
    pretrained: bool = True,
    learning_rate: float = LEARNING_RATE
) -> Model:
    """
    Factory creating one of the 4 experimental study models:
    - 'image_only'
    - 'clinical_only'
    - 'multimodal'
    - 'missing_aware'
    """
    if model_type == "image_only":
        model = build_image_only_model(pretrained=pretrained)
    elif model_type == "clinical_only":
        model = build_clinical_only_model()
    elif model_type == "multimodal":
        model = build_multimodal_fusion_model(pretrained=pretrained, name="multimodal_model")
    elif model_type == "missing_aware":
        model = build_multimodal_fusion_model(pretrained=pretrained, name="missing_data_aware_model")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compile with Binary Cross-Entropy (Eq. 6) and core metrics
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
    return model
