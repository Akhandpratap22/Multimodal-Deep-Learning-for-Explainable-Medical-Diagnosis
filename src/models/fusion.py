import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import List

from src.config import (
    IMAGE_SHAPE,
    CLINICAL_INPUT_DIM,
    CLASSIFIER_HIDDEN_DIMS,
    DROPOUT_RATE,
    NUM_CLASSES,
)
from src.models.image_encoder import build_image_encoder
from src.models.clinical_encoder import build_clinical_encoder


def build_multimodal_fusion_model(
    pretrained: bool = True,
    hidden_dims: List[int] = None,
    dropout_rate: float = DROPOUT_RATE,
    name: str = "multimodal_model"
) -> Model:
    """
    Constructs the end-to-end Multimodal Fusion Architecture defined in
    Section III-E, Eqs. (4) & (5), and Figure 1.
    
    1. Image Encoder extracts z_i = f_theta(x_i)
    2. Clinical Encoder extracts z_c = g_phi(c_i)
    3. Intermediate fusion joins representations: z_f = z_i || z_c
    4. Classifier produces diagnostic prediction: y_hat = h_psi(z_f)
    """
    if hidden_dims is None:
        hidden_dims = CLASSIFIER_HIDDEN_DIMS

    # 1. Inputs
    img_input = layers.Input(shape=IMAGE_SHAPE, name="image_input")
    clin_input = layers.Input(shape=(CLINICAL_INPUT_DIM,), name="clinical_input")

    # 2. Modality Encoders
    img_encoder = build_image_encoder(pretrained=pretrained)
    clin_encoder = build_clinical_encoder()

    z_i = img_encoder(img_input)
    z_c = clin_encoder(clin_input)

    # 3. Intermediate Feature Fusion: Concatenation (Eq. 4)
    z_f = layers.Concatenate(name="fusion_concatenation")([z_i, z_c])

    # 4. Classification Head (Eq. 5)
    x = z_f
    for idx, dim in enumerate(hidden_dims):
        x = layers.Dense(dim, activation="relu", name=f"clf_dense_{idx+1}")(x)
        x = layers.BatchNormalization(name=f"clf_bn_{idx+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"clf_drop_{idx+1}")(x)

    # Sigmoid prediction for binary diagnosis (Eq. 5)
    prediction = layers.Dense(NUM_CLASSES, activation="sigmoid", name="prediction_output")(x)

    model = Model(
        inputs={"image_input": img_input, "clinical_input": clin_input},
        outputs=prediction,
        name=name
    )
    return model
