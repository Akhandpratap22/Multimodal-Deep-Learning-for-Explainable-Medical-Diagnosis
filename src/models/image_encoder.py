import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, ResNet50

from src.config import (
    IMAGE_SHAPE,
    IMAGE_EMBEDDING_DIM,
    IMAGE_BACKBONE,
    GRAD_CAM_LAYER_NAME,
)


def _preprocess_image_for_backbone(images, backbone_name: str):
    """Apply the selected ImageNet backbone's expected input normalization."""
    scaled_images = images * 255.0
    if backbone_name == "DenseNet121":
        return tf.keras.applications.densenet.preprocess_input(scaled_images)
    return tf.keras.applications.resnet50.preprocess_input(scaled_images)


def build_image_encoder(
    input_shape: tuple = IMAGE_SHAPE,
    embedding_dim: int = IMAGE_EMBEDDING_DIM,
    backbone_name: str = IMAGE_BACKBONE,
    pretrained: bool = True
) -> Model:
    """
    Builds the image encoder f_theta(x_i) as defined in Section III-C, Eq. (2).
    Extracts high-level visual features z_i from chest radiographs.
    
    Returns:
        Model with inputs=image_input, outputs=image_embedding (z_i)
    """
    inputs = layers.Input(shape=input_shape, name="image_input")
    
    # Pretrained CNN backbone
    weights = "imagenet" if pretrained else None
    normalized_input = layers.Lambda(
        lambda images: _preprocess_image_for_backbone(images, backbone_name),
        name="image_preprocessing",
    )(inputs)
    if backbone_name == "DenseNet121":
        base_cnn = DenseNet121(weights=weights, include_top=False, input_tensor=normalized_input)
    elif backbone_name == "ResNet50":
        base_cnn = ResNet50(weights=weights, include_top=False, input_tensor=normalized_input)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    if pretrained:
        base_cnn.trainable = False

    # Feature map from final convolutional layer
    features = base_cnn.output
    
    # Global pooling & projection to z_i
    pooled = layers.GlobalAveragePooling2D(name="image_gap")(features)
    x = layers.Dense(embedding_dim, activation="relu", name="image_projection")(pooled)
    x = layers.BatchNormalization(name="image_bn")(x)
    outputs = layers.Dropout(0.2, name="image_dropout")(x)

    encoder = Model(inputs=inputs, outputs=outputs, name=f"image_encoder_{backbone_name.lower()}")
    return encoder
