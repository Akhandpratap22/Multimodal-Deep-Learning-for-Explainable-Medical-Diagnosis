"""Load a verified patient-level manifest and create reproducible data splits."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    CLINICAL_FEATURES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_RATIO,
    VAL_RATIO,
)

MANIFEST_PATH = RAW_DATA_DIR / "metadata.csv"
REQUIRED_COLUMNS = ("patient_id", "image_path", "label", *CLINICAL_FEATURES)


def _patient_split(patient_df: pd.DataFrame) -> dict[str, str]:
    """Assign every patient, rather than every image, to one fixed split."""
    if patient_df["label"].nunique() != 2:
        raise ValueError("The manifest must contain both binary labels 0 and 1.")
    if patient_df.groupby("label")["patient_id"].nunique().min() < 6:
        raise ValueError("At least six distinct patients per label are required for stratified 70/15/15 splits.")

    train_ids, remainder_ids = train_test_split(
        patient_df["patient_id"],
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=patient_df["label"],
    )
    remainder = patient_df[patient_df["patient_id"].isin(remainder_ids)]
    test_fraction = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_ids, test_ids = train_test_split(
        remainder["patient_id"],
        test_size=test_fraction,
        random_state=RANDOM_SEED,
        stratify=remainder["label"],
    )
    return {
        **dict.fromkeys(train_ids, "train"),
        **dict.fromkeys(val_ids, "val"),
        **dict.fromkeys(test_ids, "test"),
    }


def prepare_dataset(manifest_path: Union[Path, str] = MANIFEST_PATH) -> pd.DataFrame:
    """Validate real paired data from CSV and write patient-level train/val/test splits.

    The manifest must contain patient_id, image_path, label, and every configured
    clinical feature. Empty clinical cells remain missing; no values or images are
    generated to fill gaps.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = BASE_DIR / manifest_path
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Paired-data manifest not found: {manifest_path}. "
            "Create data/raw/metadata.csv from verified patient-linked records; "
            "see the README for the required columns."
        )

    df = pd.read_csv(manifest_path)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Manifest is missing required columns: {', '.join(missing_columns)}"
        )
    if df.empty:
        raise ValueError("The paired-data manifest contains no records.")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    df["patient_id"] = df["patient_id"].astype("string").str.strip()
    if df["patient_id"].isna().any() or df["patient_id"].eq("").any():
        raise ValueError("Every record must have a non-empty patient_id.")

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Every label must be present and binary (0 or 1).")
    df["label"] = labels.astype(np.int32)
    inconsistent_labels = df.groupby("patient_id")["label"].nunique()
    if inconsistent_labels.gt(1).any():
        raise ValueError("All records for a patient must have the same label.")

    image_paths = []
    for value in df["image_path"]:
        if pd.isna(value) or not str(value).strip():
            raise ValueError("Every record must have an image_path.")
        image_path = Path(str(value))
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by the manifest was not found: {image_path}")
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid image referenced by the manifest: {image_path}") from exc
        image_paths.append(str(image_path))
    df["image_path"] = image_paths

    for feature in CLINICAL_FEATURES:
        source_values = df[feature]
        values = pd.to_numeric(source_values, errors="coerce")
        invalid_values = source_values.notna() & values.isna()
        if invalid_values.any():
            raise ValueError(f"Clinical feature '{feature}' contains non-numeric values.")
        if not np.isfinite(values.dropna().to_numpy(dtype=np.float64)).all():
            raise ValueError(f"Clinical feature '{feature}' contains infinite values.")
        df[feature] = values
        if feature in CATEGORICAL_FEATURES and not values.dropna().isin([0, 1]).all():
            raise ValueError(f"Categorical feature '{feature}' must use 0/1 encoding.")
    if df[CLINICAL_FEATURES].notna().sum().eq(0).any():
        empty_features = df[CLINICAL_FEATURES].columns[df[CLINICAL_FEATURES].notna().sum().eq(0)]
        raise ValueError(
            "No observed training data can be provided for clinical feature(s): "
            + ", ".join(empty_features)
        )

    patient_df = df.groupby("patient_id", as_index=False)["label"].first()
    split_by_patient = _patient_split(patient_df)
    df["split"] = df["patient_id"].map(split_by_patient)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / "dataset.csv", index=False)
    for split in ("train", "val", "test"):
        df.loc[df["split"] == split].to_csv(
            PROCESSED_DATA_DIR / f"{split}.csv", index=False
        )

    print(
        "Validated paired dataset: "
        f"{len(df)} records from {df['patient_id'].nunique()} patients; "
        f"splits saved to {PROCESSED_DATA_DIR}"
    )
    return df


if __name__ == "__main__":
    prepare_dataset()
