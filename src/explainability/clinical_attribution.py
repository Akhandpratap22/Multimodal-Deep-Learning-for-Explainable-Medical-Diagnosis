import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional

from src.config import (
    CLINICAL_FEATURES,
    NUM_CLINICAL_FEATURES,
    INTEGRATED_GRADIENTS_STEPS,
)


class ClinicalIntegratedGradients:
    """
    Computes Integrated Gradients attribution for structured clinical features
    as described in Section III-H.
    
    Estimates the exact contribution (positive or negative) of each clinical variable
    relative to a healthy baseline reference.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        baseline_clinical_vector: Optional[np.ndarray] = None,
        steps: int = INTEGRATED_GRADIENTS_STEPS,
    ):
        self.model = model
        self.steps = steps
        # Default baseline: zero-mean scaled clinical features with full observation mask
        if baseline_clinical_vector is None:
            # 10 zeros for normalized features + 10 ones for observation mask
            base = np.zeros(NUM_CLINICAL_FEATURES * 2, dtype=np.float32)
            base[NUM_CLINICAL_FEATURES:] = 1.0
            self.baseline = base[np.newaxis, :]
        else:
            self.baseline = np.asarray(baseline_clinical_vector, dtype=np.float32)
            if self.baseline.ndim == 1:
                self.baseline = self.baseline[np.newaxis, :]

    def attribute(
        self,
        clinical_input: np.ndarray,
        image_input: Optional[np.ndarray] = None,
        class_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Computes feature attribution for clinical features.
        
        Args:
            clinical_input: 1D or 2D array of shape (1, 20) or (20,)
            image_input: image array (224, 224, 3) for multimodal models
            class_idx: target class index
            
        Returns:
            Dict containing attributions, percentages, baseline_pred, target_pred
        """
        clin = np.asarray(clinical_input, dtype=np.float32)
        if clin.ndim == 1:
            clin = clin[np.newaxis, :]

        # Interpolated path from baseline to patient input: alpha in [0, 1]
        alphas = np.linspace(0.0, 1.0, self.steps)[:, np.newaxis]
        # Shape: (steps, 20)
        interpolated_clin = self.baseline + alphas * (clin - self.baseline)
        interpolated_clin_tensor = tf.convert_to_tensor(interpolated_clin, dtype=tf.float32)

        # Handle multimodal model vs clinical-only model
        is_multimodal = False
        if isinstance(self.model.input, dict):
            is_multimodal = "image_input" in self.model.input
        elif isinstance(self.model.inputs, (list, tuple)) and len(self.model.inputs) > 1:
            is_multimodal = True
        elif hasattr(self.model, "input_names") and self.model.input_names and "image_input" in self.model.input_names:
            is_multimodal = True

        if is_multimodal:
            if image_input is None:
                img_batch = tf.zeros((self.steps, 224, 224, 3), dtype=tf.float32)
            else:
                img_tile = np.tile(image_input[np.newaxis, ...], (self.steps, 1, 1, 1))
                img_batch = tf.convert_to_tensor(img_tile, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(interpolated_clin_tensor)
                preds = self.model({"image_input": img_batch, "clinical_input": interpolated_clin_tensor}, training=False)
                target_score = preds[:, class_idx]
            grads = tape.gradient(target_score, interpolated_clin_tensor)
        else:
            with tf.GradientTape() as tape:
                tape.watch(interpolated_clin_tensor)
                preds = self.model(interpolated_clin_tensor, training=False)
                target_score = preds[:, class_idx]
            grads = tape.gradient(target_score, interpolated_clin_tensor)

        # Average gradients along path (Riemann integral approximation)
        avg_grads = tf.reduce_mean(grads, axis=0).numpy()
        
        # Attribution = (input - baseline) * avg_gradient
        diff = (clin[0] - self.baseline[0])
        total_attributions = diff * avg_grads

        # Extract only the 10 clinical features (ignoring the mask flags in attribution reporting)
        feature_attributions = {}
        feature_scores = total_attributions[:NUM_CLINICAL_FEATURES]
        
        for name, score in zip(CLINICAL_FEATURES, feature_scores):
            feature_attributions[name] = float(score)

        # Compute baseline and target prediction
        if is_multimodal:
            b_pred = float(self.model({
                "image_input": img_batch[:1],
                "clinical_input": tf.convert_to_tensor(self.baseline, dtype=tf.float32)
            }, training=False).numpy()[0, 0])
            t_pred = float(self.model({
                "image_input": img_batch[:1],
                "clinical_input": tf.convert_to_tensor(clin, dtype=tf.float32)
            }, training=False).numpy()[0, 0])
        else:
            b_pred = float(self.model(tf.convert_to_tensor(self.baseline, dtype=tf.float32), training=False).numpy()[0, 0])
            t_pred = float(self.model(tf.convert_to_tensor(clin, dtype=tf.float32), training=False).numpy()[0, 0])

        sorted_features = sorted(feature_attributions.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            "attributions": feature_attributions,
            "sorted_attributions": sorted_features,
            "baseline_probability": b_pred,
            "predicted_probability": t_pred,
            "attribution_sum": float(np.sum(feature_scores)),
            "top_positive_driver": [k for k, v in sorted_features if v > 0][:3],
            "top_negative_driver": [k for k, v in sorted_features if v < 0][:3],
        }
