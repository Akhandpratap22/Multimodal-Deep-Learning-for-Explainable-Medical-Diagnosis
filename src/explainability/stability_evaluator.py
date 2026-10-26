import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import cosine

from src.config import CLINICAL_FEATURES, NUM_CLINICAL_FEATURES, STABILITY_PERTURBATION_SIGMA, PERTURBATION_NUM_RUNS
from src.explainability.clinical_attribution import ClinicalIntegratedGradients
from src.explainability.grad_cam import GradCAM


class ExplanationStabilityEvaluator:
    """
    Evaluates explanation stability and sensitivity as specified in Section III-H:
    1. Perturbation Test: Modifies key inputs to measure prediction changes
    2. Stability Test: Irrelevant input changes (small noise) and attribution consistency
    3. Error Analysis: Explanation inspection on misclassified cases
    """

    def __init__(
        self,
        model: tf.keras.Model,
        ig_explainer: ClinicalIntegratedGradients,
        grad_cam: GradCAM
    ):
        self.model = model
        self.ig_explainer = ig_explainer
        self.grad_cam = grad_cam

    def run_clinical_perturbation_test(
        self,
        clinical_input: np.ndarray,
        image_input: np.ndarray,
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Perturbs top influential clinical features and measures output shift Delta y.
        """
        clin = np.asarray(clinical_input, dtype=np.float32).copy()
        if clin.ndim == 1:
            clin = clin[np.newaxis, :]

        # 1. Base prediction
        base_attr = self.ig_explainer.attribute(clin, image_input=image_input)
        p_orig = base_attr["predicted_probability"]

        # 2. Identify top driving features
        sorted_feats = base_attr["sorted_attributions"]
        top_features = [f[0] for f in sorted_feats[:top_k]]

        # 3. Perturb top features (reset normalized value to baseline 0.0)
        clin_perturbed = clin.copy()
        for feat in top_features:
            idx = CLINICAL_FEATURES.index(feat)
            clin_perturbed[0, idx] = 0.0  # Reset to population mean

        is_multimodal = False
        if isinstance(self.model.input, dict):
            is_multimodal = "image_input" in self.model.input
        elif isinstance(self.model.inputs, (list, tuple)) and len(self.model.inputs) > 1:
            is_multimodal = True
        elif hasattr(self.model, "input_names") and self.model.input_names and "image_input" in self.model.input_names:
            is_multimodal = True
        if is_multimodal:
            img_tensor = tf.convert_to_tensor(image_input[np.newaxis, ...], dtype=tf.float32)
            clin_pert_tensor = tf.convert_to_tensor(clin_perturbed, dtype=tf.float32)
            p_pert = float(self.model({"image_input": img_tensor, "clinical_input": clin_pert_tensor}, training=False).numpy()[0, 0])
        else:
            p_pert = float(self.model(tf.convert_to_tensor(clin_perturbed, dtype=tf.float32), training=False).numpy()[0, 0])

        delta_p = p_orig - p_pert

        return {
            "original_probability": p_orig,
            "perturbed_probability": p_pert,
            "probability_shift": delta_p,
            "perturbed_features": top_features,
            "sensitivity_detected": abs(delta_p) > 0.05
        }

    def run_stability_test(
        self,
        clinical_input: np.ndarray,
        image_input: np.ndarray,
        num_runs: int = PERTURBATION_NUM_RUNS,
        noise_sigma: float = STABILITY_PERTURBATION_SIGMA
    ) -> Dict[str, Any]:
        """
        Measures whether small irrelevant input noise produces markedly different explanations.
        Stability score is the average cosine similarity of attribution vectors under perturbation.
        """
        clin = np.asarray(clinical_input, dtype=np.float32).copy()
        if clin.ndim == 1:
            clin = clin[np.newaxis, :]

        base_res = self.ig_explainer.attribute(clin, image_input=image_input)
        base_vec = np.array([v for _, v in base_res["sorted_attributions"]], dtype=np.float32)
        norm_base = np.linalg.norm(base_vec)

        if norm_base < 1e-8:
            return {"mean_stability_score": 1.0, "stability_std": 0.0, "num_runs": num_runs}

        similarities = []
        for i in range(num_runs):
            # Inject subtle Gaussian noise into continuous clinical features
            noise = np.random.normal(0, noise_sigma, size=clin.shape).astype(np.float32)
            # Do not corrupt the observation mask portion
            noise[0, NUM_CLINICAL_FEATURES:] = 0.0
            noisy_clin = clin + noise

            res_noisy = self.ig_explainer.attribute(noisy_clin, image_input=image_input)
            noisy_vec = np.array([res_noisy["attributions"][k] for k, _ in base_res["sorted_attributions"]], dtype=np.float32)

            norm_noisy = np.linalg.norm(noisy_vec)
            if norm_noisy > 1e-8:
                cos_sim = float(np.dot(base_vec, noisy_vec) / (norm_base * norm_noisy))
                similarities.append(np.clip(cos_sim, -1.0, 1.0))
            else:
                similarities.append(1.0)

        mean_sim = float(np.mean(similarities))
        std_sim = float(np.std(similarities))

        return {
            "mean_stability_score": round(mean_sim, 4),
            "stability_std": round(std_sim, 4),
            "num_runs": num_runs,
            "status": "Robust / Stable" if mean_sim >= 0.85 else "Moderately Sensitive"
        }
