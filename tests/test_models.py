import pytest
import numpy as np
import tensorflow as tf

from src.config import (
    IMAGE_SHAPE,
    CLINICAL_INPUT_DIM,
)
from src.models.model_factory import build_model


def test_image_only_model_forward():
    model = build_model("image_only", pretrained=False)
    dummy_img = np.random.uniform(0, 1, size=(2, *IMAGE_SHAPE)).astype(np.float32)
    pred = model(dummy_img, training=False).numpy()
    assert pred.shape == (2, 1)
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)


def test_clinical_only_model_forward():
    model = build_model("clinical_only", pretrained=False)
    dummy_clin = np.random.uniform(-2, 2, size=(4, CLINICAL_INPUT_DIM)).astype(np.float32)
    pred = model(dummy_clin, training=False).numpy()
    assert pred.shape == (4, 1)
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)


def test_multimodal_model_forward():
    model = build_model("multimodal", pretrained=False)
    dummy_img = np.random.uniform(0, 1, size=(2, *IMAGE_SHAPE)).astype(np.float32)
    dummy_clin = np.random.uniform(-2, 2, size=(2, CLINICAL_INPUT_DIM)).astype(np.float32)
    pred = model({"image_input": dummy_img, "clinical_input": dummy_clin}, training=False).numpy()
    assert pred.shape == (2, 1)
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)


def test_missing_aware_model_forward():
    model = build_model("missing_aware", pretrained=False)
    dummy_img = np.random.uniform(0, 1, size=(3, *IMAGE_SHAPE)).astype(np.float32)
    dummy_clin = np.random.uniform(-2, 2, size=(3, CLINICAL_INPUT_DIM)).astype(np.float32)
    pred = model({"image_input": dummy_img, "clinical_input": dummy_clin}, training=False).numpy()
    assert pred.shape == (3, 1)
    assert np.all(pred >= 0.0) and np.all(pred <= 1.0)
