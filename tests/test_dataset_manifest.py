import numpy as np
import pandas as pd
from PIL import Image

import src.data.dataset_downloader as dataset_loader
from src.config import CATEGORICAL_FEATURES, CLINICAL_FEATURES
from src.data.preprocessor import ClinicalPreprocessor, create_tf_dataset


def test_manifest_builds_patient_level_splits_without_filling_missing_data(tmp_path, monkeypatch):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    records = []
    for index in range(12):
        patient_id = f"patient-{index}"
        label = index % 2
        for study_index in range(2):
            image_name = f"{patient_id}-{study_index}.png"
            Image.new("RGB", (8, 8), color=(label * 255, 0, 0)).save(image_dir / image_name)
            record = {
                "patient_id": patient_id,
                "image_path": f"images/{image_name}",
                "label": label,
                **{
                    feature: float(index % 2 if feature in CATEGORICAL_FEATURES else index + 1)
                    for feature in CLINICAL_FEATURES
                },
            }
            if index == 0 and study_index == 0:
                record[CLINICAL_FEATURES[0]] = np.nan
            records.append(record)

    manifest_path = tmp_path / "metadata.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)
    output_dir = tmp_path / "processed"
    monkeypatch.setattr(dataset_loader, "PROCESSED_DATA_DIR", output_dir)

    dataset = dataset_loader.prepare_dataset(manifest_path)

    assert set(dataset["split"]) == {"train", "val", "test"}
    assert pd.isna(
        dataset.loc[
            (dataset["patient_id"] == "patient-0") & dataset["image_path"].str.endswith("-0.png"),
            CLINICAL_FEATURES[0],
        ].iloc[0]
    )
    assert (output_dir / "train.csv").is_file()
    patients_by_split = {}
    for split in ("train", "val", "test"):
        split_data = pd.read_csv(output_dir / f"{split}.csv")
        patients_by_split[split] = set(split_data["patient_id"])
    assert patients_by_split["train"].isdisjoint(patients_by_split["val"])
    assert patients_by_split["train"].isdisjoint(patients_by_split["test"])
    assert patients_by_split["val"].isdisjoint(patients_by_split["test"])

    train_data = pd.read_csv(output_dir / "train.csv")
    preprocessor = ClinicalPreprocessor().fit(train_data)
    finite_dataset = create_tf_dataset(
        train_data,
        preprocessor,
        batch_size=4,
        model_type="clinical_only",
        shuffle=False,
    )
    assert len(list(finite_dataset.as_numpy_iterator())) == int(np.ceil(len(train_data) / 4))

    repeated_dataset = create_tf_dataset(
        train_data.iloc[[0]],
        preprocessor,
        batch_size=1,
        model_type="clinical_only",
        mask_rate=0.5,
        shuffle=False,
        seed=17,
        repeat=True,
    )
    iterator = iter(repeated_dataset)
    first_mask = next(iterator)[0].numpy()[0, len(CLINICAL_FEATURES):]
    second_mask = next(iterator)[0].numpy()[0, len(CLINICAL_FEATURES):]
    assert not np.array_equal(first_mask, second_mask)
