import json
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    BATCH_SIZE,
    MASKING_RATES,
    RANDOM_SEED,
)
from src.data.dataset_downloader import MANIFEST_PATH, prepare_dataset
from src.data.preprocessor import ClinicalPreprocessor, create_tf_dataset, load_and_preprocess_image
from src.models.model_factory import build_model
from src.evaluation.metrics import calculate_metrics, find_optimal_threshold
from src.evaluation.visualizer import (
    plot_figure_3a_masking_auc,
    plot_figure_3b_roc_curves,
    plot_figure_3c_calibration_curves,
    plot_clinical_attribution_waterfall
)
from src.explainability.grad_cam import GradCAM
from src.explainability.clinical_attribution import ClinicalIntegratedGradients
from src.explainability.stability_evaluator import ExplanationStabilityEvaluator


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_single_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: ClinicalPreprocessor,
    epochs: int = 8,
    mask_rate_train: float = 0.0
) -> tf.keras.Model:
    """Trains a given model architecture with early stopping and learning rate scheduling."""
    print(f"\n=======================================================")
    print(f" TRAINING: {model_type.upper()} (train mask rate={mask_rate_train*100:.0f}%)")
    print(f"=======================================================")

    model = build_model(model_type=model_type, pretrained=True)

    # Prepare datasets
    train_ds = create_tf_dataset(
        train_df,
        preprocessor,
        model_type=model_type,
        mask_rate=mask_rate_train,
        shuffle=True,
        augment=True,
        repeat=True,
    )
    val_ds = create_tf_dataset(
        val_df,
        preprocessor,
        model_type=model_type,
        mask_rate=0.0,
        shuffle=False,
        augment=False,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    steps_per_epoch = max(1, int(np.ceil(len(train_df) / BATCH_SIZE)))
    validation_steps = max(1, int(np.ceil(len(val_df) / BATCH_SIZE)))

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    return model


def evaluate_model_on_test(
    model: tf.keras.Model,
    model_type: str,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: ClinicalPreprocessor,
    mask_rate: float = 0.0,
    seed: int = RANDOM_SEED,
) -> dict:
    """Evaluates model using validation-derived threshold on held-out test data."""
    # Find optimal threshold on validation set
    val_ds = create_tf_dataset(val_df, preprocessor, model_type=model_type, mask_rate=0.0, shuffle=False)
    val_preds = model.predict(val_ds, verbose=0).ravel()
    best_thresh = find_optimal_threshold(val_df["label"].values, val_preds)

    # Test predictions
    test_ds = create_tf_dataset(
        test_df,
        preprocessor,
        model_type=model_type,
        mask_rate=mask_rate,
        shuffle=False,
        seed=seed,
    )
    test_preds = model.predict(test_ds, verbose=0).ravel()

    metrics = calculate_metrics(test_df["label"].values, test_preds, threshold=best_thresh)
    return {
        "metrics": metrics,
        "y_true": test_df["label"].values,
        "y_pred_probs": test_preds
    }


def main():
    set_seed()
    print("=" * 70)
    print(" MULTIMODAL DEEP LEARNING FOR EXPLAINABLE MEDICAL DIAGNOSIS")
    print(" Paper Study Protocol Implementation & Benchmark Pipeline")
    print("=" * 70)

    # 1. Dataset Acquisition
    df = prepare_dataset()
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    val_df = pd.read_csv(PROCESSED_DATA_DIR / "val.csv")
    test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

    metadata_path = SAVED_MODELS_DIR / "training_metadata.json"
    metadata_path.unlink(missing_ok=True)
    with open(RESULTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "not_evaluated",
            "results": {},
            "message": "Training/evaluation is in progress or has not completed.",
        }, f, indent=2)

    # 2. Fit clinical preprocessor strictly on training data
    preprocessor = ClinicalPreprocessor()
    preprocessor.fit(train_df)
    preprocessor_path = PROCESSED_DATA_DIR / "preprocessor.json"
    preprocessor.save(preprocessor_path)
    print(f"Clinical preprocessor fitted and saved to {preprocessor_path}")

    # 3. Train all 4 experimental study models (Table I)
    models = {}
    
    # Model 1: Image-only Baseline
    models["image_only"] = train_single_model("image_only", train_df, val_df, preprocessor, epochs=6)
    
    # Model 2: Clinical-only Baseline
    models["clinical_only"] = train_single_model("clinical_only", train_df, val_df, preprocessor, epochs=10)
    
    # Model 3: Multimodal Standard Model
    models["multimodal"] = train_single_model("multimodal", train_df, val_df, preprocessor, epochs=8, mask_rate_train=0.0)
    
    # Model 4: Missing-data-aware Multimodal Model (trained with 30% simulated missingness)
    models["missing_aware"] = train_single_model("missing_aware", train_df, val_df, preprocessor, epochs=8, mask_rate_train=0.30)

    # 4. Save model weights
    for name, model in models.items():
        save_path = SAVED_MODELS_DIR / f"{name}_weights.weights.h5"
        model.save_weights(save_path)
        print(f"Saved weights: {save_path}")

    # 5. Evaluate all models on complete test set
    print("\n" + "=" * 70)
    print(" EVALUATION RESULTS: TABLE I MODEL PERFORMANCE COMPARISON")
    print("=" * 70)

    model_display_names = {
        "image_only": "Image-only",
        "clinical_only": "Clinical-only",
        "multimodal": "Multimodal",
        "missing_aware": "Missing-data-aware"
    }

    results_table = {}
    roc_plot_data = {}
    calib_plot_data = {}

    for key, model in models.items():
        disp_name = model_display_names[key]
        eval_res = evaluate_model_on_test(model, key, test_df, val_df, preprocessor, mask_rate=0.0)
        results_table[disp_name] = eval_res["metrics"]
        roc_plot_data[disp_name] = {"y_true": eval_res["y_true"], "y_pred_probs": eval_res["y_pred_probs"]}
        calib_plot_data[disp_name] = {"y_true": eval_res["y_true"], "y_pred_probs": eval_res["y_pred_probs"]}

    # Print Table I
    print(f"\n| {'Model':<20} | {'Acc.':<8} | {'Recall':<8} | {'F1-Score':<8} | {'ROC-AUC':<8} | {'Brier':<8} |")
    print(f"|{'-'*22}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|{'-'*10}|")
    for name, m in results_table.items():
        print(f"| {name:<20} | {m['accuracy']:<8.4f} | {m['recall']:<8.4f} | {m['f1_score']:<8.4f} | {m['roc_auc']:<8.4f} | {m['brier_score']:<8.4f} |")

    masking_results = {
        "rates": list(MASKING_RATES),
        "runs": 5,
        "multimodal": [],
        "missing_aware": [],
    }

    # 6. Missing Data Evaluation Experiment (Section III-G, Section IV-F, Fig 3a)
    print("\n" + "=" * 70)
    print(" RUNNING MISSING-DATA EVALUATION AT 0%, 10%, 30%, 50% MASKING RATES")
    print("=" * 70)

    for rate in MASKING_RATES:
        auc_values = {"multimodal": [], "missing_aware": []}
        for run in range(masking_results["runs"]):
            mask_seed = RANDOM_SEED + run
            eval_mm = evaluate_model_on_test(
                models["multimodal"], "multimodal", test_df, val_df, preprocessor,
                mask_rate=rate, seed=mask_seed
            )
            eval_ma = evaluate_model_on_test(
                models["missing_aware"], "missing_aware", test_df, val_df, preprocessor,
                mask_rate=rate, seed=mask_seed
            )
            auc_values["multimodal"].append(eval_mm["metrics"]["roc_auc"])
            auc_values["missing_aware"].append(eval_ma["metrics"]["roc_auc"])

        for model_name in auc_values:
            values = np.asarray(auc_values[model_name], dtype=np.float64)
            masking_results[model_name].append({
                "mean_roc_auc": float(np.mean(values)),
                "std_roc_auc": float(np.std(values)),
                "runs": auc_values[model_name],
            })
        print(
            f"Masking Rate: {rate*100:2.0f}% | "
            f"Multimodal ROC-AUC: {np.mean(auc_values['multimodal']):.4f} "
            f"(±{np.std(auc_values['multimodal']):.4f}) | "
            f"Missing-Aware ROC-AUC: {np.mean(auc_values['missing_aware']):.4f} "
            f"(±{np.std(auc_values['missing_aware']):.4f})"
        )

    multimodal_aucs = [result["mean_roc_auc"] for result in masking_results["multimodal"]]
    missing_aware_aucs = [result["mean_roc_auc"] for result in masking_results["missing_aware"]]

    # 7. Generate publication plots (Fig 3a, 3b, 3c)
    print("\n[Plotting] Generating publication-grade figures...")
    plot_figure_3a_masking_auc(MASKING_RATES, multimodal_aucs, missing_aware_aucs)
    plot_figure_3b_roc_curves(roc_plot_data)
    plot_figure_3c_calibration_curves(calib_plot_data)

    # 8. Explainability & Stability Verification
    print("\n" + "=" * 70)
    print(" EXPLAINABILITY (XAI) & STABILITY EVALUATION")
    print("=" * 70)
    
    test_sample = test_df.iloc[0]
    sample_img = load_and_preprocess_image(test_sample["image_path"])
    sample_clin_input, _ = preprocessor.prepare_multimodal_input(test_sample.to_dict())

    # Grad-CAM
    grad_cam = GradCAM(models["multimodal"])
    heatmap = grad_cam.generate_heatmap(sample_img, clinical_input=sample_clin_input)
    overlay = grad_cam.overlay_heatmap(sample_img, heatmap)
    from PIL import Image as PILImage
    PILImage.fromarray(overlay).save(FIGURES_DIR / f"gradcam_{test_sample['patient_id']}.jpg")

    # Integrated Gradients
    ig = ClinicalIntegratedGradients(models["multimodal"])
    attr_res = ig.attribute(sample_clin_input, image_input=sample_img)
    plot_clinical_attribution_waterfall(attr_res["attributions"], test_sample["patient_id"])

    # Stability & Perturbation Tests
    stability_eval = ExplanationStabilityEvaluator(models["multimodal"], ig, grad_cam)
    pert_res = stability_eval.run_clinical_perturbation_test(sample_clin_input, sample_img)
    stab_res = stability_eval.run_stability_test(sample_clin_input, sample_img, num_runs=10)

    print(f"Patient {test_sample['patient_id']} Label: {int(test_sample['label'])}")
    print(f"Predicted Probability: {attr_res['predicted_probability']:.4f}")
    print(f"Top Positive Attributions: {attr_res['top_positive_driver']}")
    print(f"Perturbation Probability Shift: {pert_res['probability_shift']:.4f}")
    print(f"Explanation Stability Score (Cosine Sim): {stab_res['mean_stability_score']:.4f} ({stab_res['status']})")

    manifest_digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    with open(RESULTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "status": "evaluated",
            "results": results_table,
            "missing_data_evaluation": masking_results,
        }, f, indent=2)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_manifest": str(MANIFEST_PATH.relative_to(BASE_DIR)),
            "dataset_manifest_sha256": manifest_digest,
            "model_types": list(models),
            "patient_counts": {
                split: int(df.loc[df["split"] == split, "patient_id"].nunique())
                for split in ("train", "val", "test")
            },
        }, f, indent=2)

    print("\nBenchmark and pipeline completed successfully!")


if __name__ == "__main__":
    main()
