import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    IMAGE_SIZE,
    IMAGE_SHAPE,
    CONTINUOUS_FEATURES,
    CLINICAL_FEATURES,
    BATCH_SIZE,
)
from src.data.missing_simulator import MissingDataSimulator


class ClinicalPreprocessor:
    """
    Fits normalization statistics and imputation constants strictly on the training set
    (Section IV-B), and prepares input vectors [c_tilde, m].
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, float]] = {}
        self.imputation_values: np.ndarray = np.zeros(len(CLINICAL_FEATURES), dtype=np.float32)
        self.is_fitted: bool = False
        self.simulator: Optional[MissingDataSimulator] = None

    def fit(self, train_df: pd.DataFrame) -> "ClinicalPreprocessor":
        """Computes mean and std for continuous features, and median imputation defaults."""
        self.stats = {}
        imp_list = []

        for feat in CLINICAL_FEATURES:
            values = pd.to_numeric(train_df[feat], errors="coerce").dropna().to_numpy(dtype=np.float32)
            if values.size == 0:
                raise ValueError(
                    f"Clinical feature '{feat}' has no observed values in the training split."
                )
            if feat in CONTINUOUS_FEATURES:
                mean = float(np.mean(values))
                std = float(np.std(values))
                std = std if std > 1e-6 else 1.0
                median = float(np.median(values))
                self.stats[feat] = {"mean": mean, "std": std, "median": median}
                # Normalized imputation value is (median - mean) / std
                imp_list.append((median - mean) / std)
            else:
                mode = float(pd.Series(values).mode().iloc[0])
                self.stats[feat] = {"mean": 0.0, "std": 1.0, "mode": mode}
                imp_list.append(mode)

        self.imputation_values = np.array(imp_list, dtype=np.float32)
        self.simulator = MissingDataSimulator(self.imputation_values, CLINICAL_FEATURES)
        self.is_fitted = True
        return self

    def transform_features(self, df_or_dict: Any) -> np.ndarray:
        """Scales continuous features and retains categorical values in [0, 1]."""
        if not self.is_fitted:
            raise RuntimeError("ClinicalPreprocessor must be fitted before transform.")

        if isinstance(df_or_dict, dict):
            df = pd.DataFrame([df_or_dict]).reindex(columns=CLINICAL_FEATURES)
        else:
            df = df_or_dict.reindex(columns=CLINICAL_FEATURES)

        transformed = []
        for feat in CLINICAL_FEATURES:
            val = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=np.float32)
            if feat in CONTINUOUS_FEATURES:
                mean = self.stats[feat]["mean"]
                std = self.stats[feat]["std"]
                scaled = (val - mean) / std
            else:
                scaled = val
            transformed.append(scaled)

        # Shape: (batch_size, num_features)
        return np.column_stack(transformed).astype(np.float32)

    def prepare_multimodal_input(
        self,
        df_or_dict: Any,
        mask_rate: float = 0.0,
        mask: Optional[np.ndarray] = None,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces [c_tilde, m] where missing features are imputed and flagged via binary mask.
        Returns:
            network_input: shape (N, num_features * 2)
            applied_mask: shape (N, num_features)
        """
        feats = self.transform_features(df_or_dict)
        c_tilde, m = self.simulator.apply_mask(feats, mask_rate=mask_rate, mask=mask, seed=seed)
        network_input = np.concatenate([c_tilde, m], axis=-1)
        return network_input, m

    def save(self, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.stats,
            "imputation_values": self.imputation_values.tolist(),
            "features": CLINICAL_FEATURES,
        }
        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, load_path: Path) -> "ClinicalPreprocessor":
        instance = cls()
        with open(load_path, "r") as f:
            data = json.load(f)
        if data.get("features") != CLINICAL_FEATURES:
            raise ValueError(
                f"Preprocessor features in {load_path} do not match configured clinical features."
            )
        instance.stats = data["stats"]
        instance.imputation_values = np.array(data["imputation_values"], dtype=np.float32)
        instance.simulator = MissingDataSimulator(instance.imputation_values, CLINICAL_FEATURES)
        instance.is_fitted = True
        return instance


def load_and_preprocess_image(image_path: str, augment: bool = False) -> np.ndarray:
    """
    Loads, resizes and normalizes an image to [0, 1].
    Handles both relative and absolute paths.
    """
    p = Path(image_path)
    if not p.is_absolute():
        p = BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Image not found at {p}")

    with Image.open(p) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

    if augment:
        # Subtle horizontal flip
        if np.random.rand() > 0.5:
            arr = np.fliplr(arr)
        # Subtle brightness jitter [0.9, 1.1]
        brightness_factor = np.random.uniform(0.9, 1.1)
        arr = np.clip(arr * brightness_factor, 0.0, 1.0)

    return arr


def create_tf_dataset(
    df: pd.DataFrame,
    preprocessor: ClinicalPreprocessor,
    batch_size: int = BATCH_SIZE,
    model_type: str = "multimodal",
    mask_rate: float = 0.0,
    shuffle: bool = True,
    augment: bool = False,
    seed: int = 42,
    repeat: bool = False,
) -> tf.data.Dataset:
    """
    Constructs an optimized tf.data.Dataset for training or evaluation.
    
    Args:
        model_type: 'image_only', 'clinical_only', 'multimodal', or 'missing_aware'
        mask_rate: missingness rate applied to clinical features
    """
    image_paths = df["image_path"].tolist()
    labels = df["label"].values.astype(np.float32)
    
    # Keep the natural missing-value mask and draw simulated masks on each pass
    # through the generator so training sees different missingness patterns.
    clinical_features = preprocessor.transform_features(df)
    rng = np.random.RandomState(seed)

    def generator():
        indices = np.arange(len(df))
        if shuffle:
            np.random.shuffle(indices)
        for idx in indices:
            img = load_and_preprocess_image(image_paths[idx], augment=augment)
            clin_values, observed_mask = preprocessor.simulator.apply_mask(
                clinical_features[idx],
                mask_rate=mask_rate,
                seed=int(rng.randint(0, np.iinfo(np.int32).max)),
            )
            clin = np.concatenate([clin_values, observed_mask], axis=-1)
            lbl = np.array([labels[idx]], dtype=np.float32)

            if model_type == "image_only":
                yield img, lbl
            elif model_type == "clinical_only":
                yield clin, lbl
            else:  # multimodal or missing_aware
                yield {"image_input": img, "clinical_input": clin}, lbl

    if model_type == "image_only":
        output_signature = (
            tf.TensorSpec(shape=IMAGE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    elif model_type == "clinical_only":
        output_signature = (
            tf.TensorSpec(shape=(len(CLINICAL_FEATURES) * 2,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    else:
        output_signature = (
            {
                "image_input": tf.TensorSpec(shape=IMAGE_SHAPE, dtype=tf.float32),
                "clinical_input": tf.TensorSpec(shape=(len(CLINICAL_FEATURES) * 2,), dtype=tf.float32)
            },
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )

    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
