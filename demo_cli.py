import argparse
import json
from pathlib import Path
from typing import Optional
import numpy as np
import cv2

from src.config import (
    PROCESSED_DATA_DIR,
    SAMPLE_PATIENTS_DIR,
    SAVED_MODELS_DIR,
)
from src.data.preprocessor import ClinicalPreprocessor, load_and_preprocess_image
from src.models.model_factory import build_model
from src.explainability.grad_cam import GradCAM
from src.explainability.clinical_attribution import ClinicalIntegratedGradients
from src.explainability.stability_evaluator import ExplanationStabilityEvaluator


def run_cli_diagnostic(
    image_path: Optional[str],
    clinical_data: dict,
    model_type: str = "multimodal",
    output_dir: str = "demo_outputs"
):
    """Executes multimodal diagnostic inference with Grad-CAM and clinical attribution."""
    print("=" * 65)
    print(" MULTIMODAL RESEARCH MODEL INFERENCE")
    print("=" * 65)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Load Preprocessor
    prep_path = PROCESSED_DATA_DIR / "preprocessor.json"
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {prep_path}. Run train.py first.")
    metadata_path = SAVED_MODELS_DIR / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Training provenance not found at {metadata_path}. "
            "Train with a patient-linked manifest before running inference."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if model_type not in metadata.get("model_types", []):
        raise FileNotFoundError(f"No trained {model_type} model is recorded in {metadata_path}.")
    preprocessor = ClinicalPreprocessor.load(prep_path)

    # 2. Build and Load Model
    model = build_model(model_type=model_type, pretrained=False)
    weights_path = SAVED_MODELS_DIR / f"{model_type}_weights.weights.h5"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Trained weights not found at {weights_path}. Run train.py first.")
    model.load_weights(weights_path)
    print(f"Loaded trained weights from: {weights_path.name}")

    # 3. Preprocess Inputs
    image = load_and_preprocess_image(image_path) if image_path else None
    clin_vector, _ = preprocessor.prepare_multimodal_input(clinical_data)

    # 4. Predict
    if model_type == "image_only":
        if image is None:
            raise ValueError("An image is required for the image-only model.")
        prob = float(model.predict(image[np.newaxis, ...], verbose=0)[0, 0])
    elif model_type == "clinical_only":
        prob = float(model.predict(clin_vector, verbose=0)[0, 0])
    else:
        prob = float(model.predict({"image_input": image[np.newaxis, ...], "clinical_input": clin_vector}, verbose=0)[0, 0])

    diagnosis = "Positive model estimate (label 1)" if prob >= 0.5 else "Negative model estimate (label 0)"

    print(f"\nResearch model output: {diagnosis}")
    print(f"Positive-class probability: {prob:.4f} (research-model output; not a diagnosis)")

    # 5. Explainability (Grad-CAM)
    grad_cam = None
    if model_type != "clinical_only" and image is not None:
        grad_cam = GradCAM(model)
        heatmap = grad_cam.generate_heatmap(image, clinical_input=clin_vector)
        overlay = grad_cam.overlay_heatmap(image, heatmap)
        gradcam_path = out_p / "gradcam_overlay.jpg"
        cv2.imwrite(str(gradcam_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM Heatmap saved to: {gradcam_path}")

    # 6. Clinical Attribution (Integrated Gradients)
    ig = None
    if model_type != "image_only":
        ig = ClinicalIntegratedGradients(model)
        attr_res = ig.attribute(clin_vector, image_input=image)
        print("\nClinical Variable Attribution (Impact on Diagnosis):")
        for feat, score in attr_res["sorted_attributions"][:5]:
            direction = "Risk increase" if score > 0 else "Risk reduction"
            print(f"  {feat:<20}: {score:+.4f} ({direction})")

    # 7. Stability & Perturbation Evaluation
    if ig is not None and grad_cam is not None and image is not None:
        evaluator = ExplanationStabilityEvaluator(model, ig, grad_cam)
        pert = evaluator.run_clinical_perturbation_test(clin_vector, image)
        stab = evaluator.run_stability_test(clin_vector, image, num_runs=5)
        print("\nExplanation Evaluation:")
        print(f"  Perturbation Shift Δp: {pert['probability_shift']:+.4f} (Top features: {pert['perturbed_features']})")
        print(f"  Explanation Stability: {stab['mean_stability_score']:.4f} ({stab['status']})")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Medical Diagnosis CLI")
    parser.add_argument("--image", type=str, help="Path to chest radiograph image")
    parser.add_argument("--case_id", type=str, help="Preloaded sample case ID, if sample cases have been configured")
    parser.add_argument("--clinical-json", type=str, help="JSON file containing real clinical feature values")
    parser.add_argument("--model", type=str, default="multimodal", choices=["image_only", "clinical_only", "multimodal", "missing_aware"])
    args = parser.parse_args()

    sample_file = SAMPLE_PATIENTS_DIR / "sample_cases.json"
    if args.clinical_json:
        if not args.image and args.model != "clinical_only":
            parser.error("--image is required unless --model clinical_only is selected")
        with open(args.clinical_json, "r", encoding="utf-8") as f:
            clinical_data = json.load(f)
        run_cli_diagnostic(args.image, clinical_data, model_type=args.model)
    elif sample_file.exists() and args.case_id:
        with open(sample_file, "r") as f:
            cases = json.load(f)
        selected = next((c for c in cases if c["case_id"] == args.case_id), None)
        if selected is None:
            parser.error(f"Case ID {args.case_id!r} was not found.")
        img = args.image or selected["image_path"]
        print(f"Testing sample patient: {selected['title']}")
        run_cli_diagnostic(img, selected, model_type=args.model)
    else:
        parser.error("Provide --clinical-json with --image, or choose a configured --case_id.")
