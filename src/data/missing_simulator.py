import numpy as np
from typing import Tuple, Optional


class MissingDataSimulator:
    """
    Simulates missing clinical features as described in Section III-G and Figure 2:
    Formula: c_tilde = m * c + (1 - m) * c_imp
    where:
      m: binary mask (1 = observed, 0 = missing)
      c: original clinical feature vector
      c_imp: imputation value computed strictly from the training set
      Input to clinical encoder is [c_tilde, m]
    """

    def __init__(self, imputation_values: np.ndarray, feature_names: list):
        """
        Args:
            imputation_values: 1D array of shape (num_features,) with mean/median values
            feature_names: list of feature names
        """
        self.imputation_values = np.array(imputation_values, dtype=np.float32)
        self.feature_names = feature_names
        self.num_features = len(imputation_values)

    def apply_mask(
        self,
        features: np.ndarray,
        mask_rate: float = 0.0,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies missingness mask to clinical features.
        
        Args:
            features: 2D array of shape (batch_size, num_features) or 1D array (num_features,)
            mask_rate: fraction of features to mask (e.g. 0.10, 0.30, 0.50)
            mask: optional pre-defined binary mask
            seed: optional RNG seed
            
        Returns:
            masked_features: c_tilde of same shape as features
            mask: binary mask m (1 = observed, 0 = missing)
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random
        features = np.asarray(features, dtype=np.float32)
        is_1d = features.ndim == 1
        if is_1d:
            features = features[np.newaxis, :]
        if features.ndim != 2 or features.shape[1] != self.num_features:
            raise ValueError(
                f"features must have shape (batch, {self.num_features}) or "
                f"({self.num_features},)"
            )
        if not 0.0 <= mask_rate <= 1.0:
            raise ValueError("mask_rate must be between 0.0 and 1.0.")

        batch_size, num_feats = features.shape
        observed_mask = np.isfinite(features).astype(np.float32)
        safe_features = np.where(np.isfinite(features), features, self.imputation_values)

        if mask is None:
            if mask_rate <= 0.0:
                mask = np.ones((batch_size, num_feats), dtype=np.float32)
            else:
                # Generate binary mask where probability of 0 (missing) is mask_rate.
                mask = (rng.rand(batch_size, num_feats) >= mask_rate).astype(np.float32)
        else:
            mask = np.asarray(mask, dtype=np.float32)
            if mask.ndim == 1:
                mask = np.tile(mask, (batch_size, 1))
            if mask.shape != (batch_size, num_feats) or not np.isin(mask, [0.0, 1.0]).all():
                raise ValueError(f"mask must contain only 0/1 values and have shape {(batch_size, num_feats)}.")
        mask *= observed_mask

        # Equation (7): c_tilde = m * c + (1 - m) * c_imp
        imp_broadcast = np.tile(self.imputation_values, (batch_size, 1))
        c_tilde = mask * safe_features + (1.0 - mask) * imp_broadcast

        if is_1d:
            return c_tilde[0], mask[0]
        return c_tilde, mask

    def build_network_input(
        self,
        features: np.ndarray,
        mask_rate: float = 0.0,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Produces [c_tilde, m] concatenated as required by Section III-A / Fig 1.
        Returns array of shape (batch_size, num_features * 2).
        """
        c_tilde, m = self.apply_mask(features, mask_rate=mask_rate, mask=mask, seed=seed)
        if c_tilde.ndim == 1:
            return np.concatenate([c_tilde, m], axis=-1)
        return np.concatenate([c_tilde, m], axis=1)
