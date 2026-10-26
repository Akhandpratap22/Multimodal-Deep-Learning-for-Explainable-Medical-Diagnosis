import numpy as np
import pytest
import tensorflow as tf

from src.config import IMAGE_SHAPE, CLINICAL_INPUT_DIM, NUM_CLINICAL_FEATURES
from src.models.model_factory import build_model
from src.explainability.grad_cam import GradCAM
from src.explainability.clinical_attribution import ClinicalIntegratedGradients
from src.explainability.stability_evaluator import ExplanationStabilityEvaluator


def test_grad_cam_multimodal():
    model = build_model("multimodal", pretrained=False)
    grad_cam = GradCAM(model)

    dummy_img = np.random.uniform(0, 1, size=IMAGE_SHAPE).astype(np.float32)
    dummy_clin = np.random.uniform(-1, 1, size=(1, CLINICAL_INPUT_DIM)).astype(np.float32)

    heatmap = grad_cam.generate_heatmap(dummy_img, clinical_input=dummy_clin)
    assert heatmap.shape == (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)

    overlay = grad_cam.overlay_heatmap(dummy_img, heatmap)
    assert overlay.shape == IMAGE_SHAPE
    assert overlay.dtype == np.uint8


def test_grad_cam_image_only():
    model = build_model("image_only", pretrained=False)
    grad_cam = GradCAM(model)
    dummy_img = np.random.uniform(0, 1, size=IMAGE_SHAPE).astype(np.float32)

    heatmap = grad_cam.generate_heatmap(dummy_img)

    assert heatmap.shape == IMAGE_SHAPE[:2]
    assert np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0)


def test_clinical_integrated_gradients():
    model = build_model("multimodal", pretrained=False)
    ig = ClinicalIntegratedGradients(model, steps=10)

    dummy_img = np.random.uniform(0, 1, size=IMAGE_SHAPE).astype(np.float32)
    dummy_clin = np.random.uniform(-1, 1, size=(1, CLINICAL_INPUT_DIM)).astype(np.float32)

    res = ig.attribute(dummy_clin, image_input=dummy_img)
    assert "attributions" in res
    assert len(res["attributions"]) == NUM_CLINICAL_FEATURES
    assert "baseline_probability" in res
    assert "predicted_probability" in res


def test_stability_evaluator():
    model = build_model("multimodal", pretrained=False)
    ig = ClinicalIntegratedGradients(model, steps=5)
    grad_cam = GradCAM(model)
    evaluator = ExplanationStabilityEvaluator(model, ig, grad_cam)

    dummy_img = np.random.uniform(0, 1, size=IMAGE_SHAPE).astype(np.float32)
    dummy_clin = np.random.uniform(-1, 1, size=(1, CLINICAL_INPUT_DIM)).astype(np.float32)

    pert = evaluator.run_clinical_perturbation_test(dummy_clin, dummy_img)
    assert "probability_shift" in pert

    stab = evaluator.run_stability_test(dummy_clin, dummy_img, num_runs=3)
    assert "mean_stability_score" in stab
    assert -1.0 <= stab["mean_stability_score"] <= 1.0
