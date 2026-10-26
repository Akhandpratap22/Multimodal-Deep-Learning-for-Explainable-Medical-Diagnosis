import numpy as np
import pytest

from src.config import CLINICAL_FEATURES, NUM_CLINICAL_FEATURES
from src.data.missing_simulator import MissingDataSimulator


def test_missing_data_simulator_zero_mask():
    imp_vals = np.zeros(NUM_CLINICAL_FEATURES, dtype=np.float32)
    sim = MissingDataSimulator(imp_vals, CLINICAL_FEATURES)
    features = np.ones((5, NUM_CLINICAL_FEATURES), dtype=np.float32)

    c_tilde, m = sim.apply_mask(features, mask_rate=0.0)
    assert np.all(m == 1.0)
    assert np.allclose(c_tilde, features)


def test_missing_data_simulator_simulated_masking():
    imp_vals = np.array([10.0] * NUM_CLINICAL_FEATURES, dtype=np.float32)
    sim = MissingDataSimulator(imp_vals, CLINICAL_FEATURES)
    features = np.zeros((100, NUM_CLINICAL_FEATURES), dtype=np.float32)

    # 50% masking
    c_tilde, m = sim.apply_mask(features, mask_rate=0.5, seed=42)
    observed_ratio = np.mean(m)
    assert 0.35 <= observed_ratio <= 0.65

    # Check Eq. (7): where m == 0, value should equal imp_vals
    assert np.all(c_tilde[m == 0] == 10.0)
    assert np.all(c_tilde[m == 1] == 0.0)


def test_network_input_shape():
    imp_vals = np.zeros(NUM_CLINICAL_FEATURES, dtype=np.float32)
    sim = MissingDataSimulator(imp_vals, CLINICAL_FEATURES)
    features = np.ones((4, NUM_CLINICAL_FEATURES), dtype=np.float32)

    net_input = sim.build_network_input(features, mask_rate=0.3)
    assert net_input.shape == (4, NUM_CLINICAL_FEATURES * 2)


def test_naturally_missing_values_are_imputed_and_marked_unobserved():
    imp_vals = np.arange(NUM_CLINICAL_FEATURES, dtype=np.float32)
    sim = MissingDataSimulator(imp_vals, CLINICAL_FEATURES)
    features = np.ones((1, NUM_CLINICAL_FEATURES), dtype=np.float32)
    features[0, 2] = np.nan

    imputed, mask = sim.apply_mask(features, mask_rate=0.0)

    assert mask[0, 2] == 0.0
    assert imputed[0, 2] == imp_vals[2]
    assert np.all(mask[0, np.arange(NUM_CLINICAL_FEATURES) != 2] == 1.0)


def test_missing_simulator_rejects_invalid_mask_rate():
    sim = MissingDataSimulator(np.zeros(NUM_CLINICAL_FEATURES), CLINICAL_FEATURES)
    with pytest.raises(ValueError, match="mask_rate"):
        sim.apply_mask(np.ones((1, NUM_CLINICAL_FEATURES)), mask_rate=1.1)
