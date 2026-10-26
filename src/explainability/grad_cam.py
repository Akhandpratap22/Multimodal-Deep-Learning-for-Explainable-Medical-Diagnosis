import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, Optional, Union, Dict

from src.config import CLINICAL_INPUT_DIM, GRAD_CAM_LAYER_NAME, IMAGE_SHAPE


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for chest radiographs
    as described in Section III-H and Figure 1.
    
    Supports both Image-Only and Multimodal Fusion models.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_conv_layer_name: Optional[str] = None,
    ):
        self.model = model
        self.target_conv_layer_name = target_conv_layer_name or GRAD_CAM_LAYER_NAME
        self.conv_extractor = None
        self.is_nested = False
        self.has_clinical_input = len(model.inputs) > 1
        self.grad_model = self._create_gradient_model()

    def _create_gradient_model(self) -> tf.keras.Model:
        """
        Builds a gradient-extraction model that outputs:
        1. Feature maps of the target convolutional layer
        2. Final model prediction output
        """
        target_layer = None
        img_encoder_layer = None
        
        for layer in self.model.layers:
            if "image_encoder" in layer.name:
                img_encoder_layer = layer
                break

        if img_encoder_layer is not None:
            for l in img_encoder_layer.layers:
                if self.target_conv_layer_name in l.name or (isinstance(l, tf.keras.layers.Conv2D) and self.target_conv_layer_name in l.name):
                    target_layer = l
                    break
            if target_layer is None:
                conv_layers = [l for l in img_encoder_layer.layers if "conv" in l.name]
                target_layer = conv_layers[-1] if conv_layers else img_encoder_layer.layers[-5]
            
            self.conv_extractor = tf.keras.Model(
                inputs=img_encoder_layer.inputs,
                outputs=target_layer.output
            )
            self.is_nested = True
            return self.model
        else:
            for l in self.model.layers:
                if self.target_conv_layer_name in l.name:
                    target_layer = l
                    break
            if target_layer is None:
                conv_layers = [l for l in self.model.layers if "conv" in l.name]
                target_layer = conv_layers[-1] if conv_layers else self.model.layers[-5]

            self.is_nested = False
            return tf.keras.Model(
                inputs=self.model.inputs,
                outputs=[target_layer.output, self.model.output]
            )

    def generate_heatmap(
        self,
        image: np.ndarray,
        clinical_input: Optional[np.ndarray] = None,
        class_idx: int = 0
    ) -> np.ndarray:
        """
        Computes Grad-CAM spatial heatmap for the input radiograph.
        
        Args:
            image: array of shape (224, 224, 3) with values in [0, 1]
            clinical_input: optional clinical vector for multimodal models (1, 20)
            class_idx: target prediction index (0 for binary)
            
        Returns:
            heatmap: 2D array of shape (224, 224) with intensity in [0, 1]
        """
        img_tensor = tf.convert_to_tensor(image[np.newaxis, ...], dtype=tf.float32)

        if self.is_nested:
            with tf.GradientTape() as tape:
                conv_outputs = self.conv_extractor(img_tensor)
                tape.watch(conv_outputs)
                if self.has_clinical_input:
                    if clinical_input is None:
                        clinical_input = np.zeros((1, CLINICAL_INPUT_DIM), dtype=np.float32)
                    clin_tensor = tf.convert_to_tensor(clinical_input, dtype=tf.float32)
                    preds = self.model(
                        {"image_input": img_tensor, "clinical_input": clin_tensor},
                        training=False,
                    )
                else:
                    preds = self.model(img_tensor, training=False)
                loss = preds[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
        else:
            with tf.GradientTape() as tape:
                conv_outputs, preds = self.grad_model(img_tensor, training=False)
                loss = preds[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weighted combination of feature map channels
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        # Resize heatmap using PIL to match input dimensions
        pil_heatmap = Image.fromarray((heatmap * 255.0).astype(np.uint8)).resize(
            (image.shape[1], image.shape[0]),
            Image.Resampling.BILINEAR
        )
        heatmap_resized = np.array(pil_heatmap, dtype=np.float32) / 255.0
        return np.clip(heatmap_resized, 0.0, 1.0)

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap_name: str = "jet"
    ) -> np.ndarray:
        """
        Blends Grad-CAM heatmap over the original radiograph using Matplotlib colormap.
        
        Returns:
            overlay: RGB uint8 image of shape (H, W, 3)
        """
        img_float = image.copy()
        if img_float.max() > 1.0:
            img_float = img_float / 255.0

        if img_float.ndim == 2:
            img_float = np.stack([img_float] * 3, axis=-1)

        # Apply colormap
        cmap = plt.get_cmap(colormap_name)
        colored_heatmap = cmap(heatmap)[:, :, :3]  # Strip alpha channel

        # Blend
        overlay = (1.0 - alpha) * img_float + alpha * colored_heatmap
        overlay_uint8 = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
        return overlay_uint8
