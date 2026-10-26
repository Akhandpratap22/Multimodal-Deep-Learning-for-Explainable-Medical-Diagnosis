from __future__ import annotations

import io
import json
import base64
import os
from typing import Optional, Dict, Any, TYPE_CHECKING

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import uvicorn

from src.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    SAVED_MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    CLINICAL_FEATURES,
)

if TYPE_CHECKING:
    from src.data.preprocessor import ClinicalPreprocessor

app = FastAPI(
    title="Multimodal Deep Learning for Explainable Medical Diagnosis",
    description="Academic research prototype for multimodal model evaluation; not for clinical use.",
    version="1.0.0"
)

cors_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

AGGREGATE_FIGURES = {
    "fig3a_masking_rate_vs_roc_auc.png",
    "fig3b_roc_curves.png",
    "fig3c_calibration_curves.png",
}

# Global model cache
MODELS: Dict[str, Any] = {}
PREPROCESSOR: Optional[ClinicalPreprocessor] = None
EXPLAINERS: Dict[str, Any] = {}


def get_preprocessor() -> ClinicalPreprocessor:
    global PREPROCESSOR
    if PREPROCESSOR is None:
        p_path = PROCESSED_DATA_DIR / "preprocessor.json"
        if not p_path.exists():
            raise HTTPException(
                status_code=503,
                detail="A fitted training-set preprocessor is required. Run train.py with a verified data manifest.",
            )
        from src.data.preprocessor import ClinicalPreprocessor

        PREPROCESSOR = ClinicalPreprocessor.load(p_path)
    return PREPROCESSOR


def get_or_load_model(model_type: str):
    """Loads and caches models on demand."""
    global MODELS, EXPLAINERS
    if model_type not in MODELS:
        metadata_path = SAVED_MODELS_DIR / "training_metadata.json"
        if not metadata_path.is_file():
            raise HTTPException(
                status_code=503,
                detail="No trained model provenance is available. Train on patient-linked data before inference.",
            )
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        if model_type not in metadata.get("model_types", []):
            raise HTTPException(
                status_code=503,
                detail=f"No trained {model_type} model is recorded. Run train.py first.",
            )

        weights_path = SAVED_MODELS_DIR / f"{model_type}_weights.weights.h5"
        if not weights_path.is_file():
            raise HTTPException(
                status_code=503,
                detail=f"Trained weights for {model_type} are missing. Run train.py first.",
            )

        from src.models.model_factory import build_model
        from src.explainability.grad_cam import GradCAM
        from src.explainability.clinical_attribution import ClinicalIntegratedGradients
        from src.explainability.stability_evaluator import ExplanationStabilityEvaluator

        print(f"Loading {model_type} into memory...")
        model = build_model(model_type=model_type, pretrained=False)
        model.load_weights(weights_path)
        print(f"Loaded weights for {model_type}")
        MODELS[model_type] = model
        
        # Initialize explainers
        grad_cam = GradCAM(model) if model_type != "clinical_only" else None
        ig = ClinicalIntegratedGradients(model) if model_type != "image_only" else None
        stability = ExplanationStabilityEvaluator(model, ig, grad_cam) if (ig and grad_cam) else None
        
        EXPLAINERS[model_type] = {
            "grad_cam": grad_cam,
            "ig": ig,
            "stability": stability
        }
    return MODELS[model_type], EXPLAINERS[model_type]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = BASE_DIR / "web" / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Web Dashboard template not found</h1>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Returns Table I metrics and figures metadata."""
    metrics_file = RESULTS_DIR / "metrics_summary.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            return json.load(f)
    return {"status": "not_evaluated", "results": {}}


@app.get("/results/figures/{figure_name}")
async def serve_benchmark_figure(figure_name: str):
    """Expose aggregate study figures without serving patient-level result files."""
    if figure_name not in AGGREGATE_FIGURES:
        raise HTTPException(status_code=404, detail="Figure not found.")
    figure_path = FIGURES_DIR / figure_name
    if not figure_path.is_file():
        raise HTTPException(status_code=404, detail="Figure has not been generated.")
    return FileResponse(figure_path, media_type="image/png")


@app.post("/api/diagnose")
async def diagnose(
    image: Optional[UploadFile] = File(None),
    model_type: str = Form("multimodal"),
    mask_rate: float = Form(0.0),
    age: Optional[float] = Form(None),
    sex: Optional[int] = Form(None),
    temperature: Optional[float] = Form(None),
    oxygen_saturation: Optional[float] = Form(None),
    heart_rate: Optional[float] = Form(None),
    respiratory_rate: Optional[float] = Form(None),
    leukocyte_count: Optional[float] = Form(None),
    cough: Optional[int] = Form(None),
    dyspnea: Optional[int] = Form(None),
    fever: Optional[int] = Form(None),
):
    """
    Multimodal diagnosis endpoint executing prediction, Grad-CAM, and Clinical IG.
    """
    allowed_models = {"image_only", "clinical_only", "multimodal", "missing_aware"}
    if model_type not in allowed_models:
        raise HTTPException(status_code=422, detail=f"Unknown model_type: {model_type}")
    if not 0.0 <= mask_rate <= 0.5:
        raise HTTPException(status_code=422, detail="mask_rate must be between 0.0 and 0.5.")
    model, explainers = get_or_load_model(model_type)
    preprocessor = get_preprocessor()

    # 1. Process Image
    img_array = None
    orig_b64 = None
    if image is not None and image.filename:
        content = await image.read()
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Uploaded file is not a readable image.") from exc
        pil_img_resized = pil_img.resize((224, 224), Image.Resampling.BILINEAR)
        img_array = np.array(pil_img_resized, dtype=np.float32) / 255.0
        
        # Base64 string for display
        buf = io.BytesIO()
        pil_img_resized.save(buf, format="JPEG")
        orig_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    if img_array is None and model_type != "clinical_only":
        raise HTTPException(status_code=422, detail="Upload a chest image for this model.")

    # 2. Process Clinical Data
    raw_clinical = {
        "age": age,
        "sex": sex,
        "temperature": temperature,
        "oxygen_saturation": oxygen_saturation,
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "leukocyte_count": leukocyte_count,
        "cough": cough,
        "dyspnea": dyspnea,
        "fever": fever,
    }

    clin_vector, applied_mask = preprocessor.prepare_multimodal_input(
        raw_clinical,
        mask_rate=mask_rate
    )

    # 3. Model Inference
    if model_type == "image_only":
        pred_prob = float(model.predict(img_array[np.newaxis, ...], verbose=0)[0, 0])
    elif model_type == "clinical_only":
        pred_prob = float(model.predict(clin_vector, verbose=0)[0, 0])
    else:  # multimodal or missing_aware
        pred_prob = float(model.predict(
            {"image_input": img_array[np.newaxis, ...], "clinical_input": clin_vector},
            verbose=0
        )[0, 0])

    is_positive = pred_prob >= 0.5
    diagnosis = "Positive model estimate (label 1)" if is_positive else "Negative model estimate (label 0)"
    confidence = pred_prob if is_positive else (1.0 - pred_prob)

    # 4. Grad-CAM (Image XAI)
    gradcam_b64 = None
    if explainers["grad_cam"] is not None:
        try:
            heatmap = explainers["grad_cam"].generate_heatmap(img_array, clinical_input=clin_vector)
            overlay = explainers["grad_cam"].overlay_heatmap(img_array, heatmap)
            buf = io.BytesIO()
            Image.fromarray(overlay).save(buf, format="JPEG", quality=95)
            gradcam_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            print("Grad-CAM error:", e)

    # 5. Integrated Gradients (Clinical XAI)
    attributions = {}
    sorted_attributions = []
    if explainers["ig"] is not None:
        try:
            attr_res = explainers["ig"].attribute(clin_vector, image_input=img_array)
            attributions = attr_res["attributions"]
            sorted_attributions = attr_res["sorted_attributions"]
        except Exception as e:
            print("IG error:", e)

    # 6. Stability and Perturbation evaluation
    stability_data = {}
    if explainers["stability"] is not None:
        try:
            pert = explainers["stability"].run_clinical_perturbation_test(clin_vector, img_array)
            stab = explainers["stability"].run_stability_test(clin_vector, img_array, num_runs=5)
            stability_data = {
                "probability_shift": float(pert["probability_shift"]),
                "perturbed_features": pert["perturbed_features"],
                "stability_score": float(stab["mean_stability_score"]),
                "stability_status": stab["status"]
            }
        except Exception as e:
            print("Stability error:", e)

    return JSONResponse({
        "diagnosis": diagnosis,
        "is_positive": is_positive,
        "probability": round(pred_prob, 4),
        "confidence": round(confidence * 100, 1),
        "model_type": model_type,
        "mask_rate": mask_rate,
        "masked_features": [CLINICAL_FEATURES[i] for i, m in enumerate(applied_mask[0]) if m == 0],
        "image_original": orig_b64,
        "gradcam_overlay": gradcam_b64,
        "clinical_attributions": attributions,
        "sorted_attributions": sorted_attributions,
        "stability": stability_data,
        "clinical_values": raw_clinical
    })


if __name__ == "__main__":
    print("Starting Multimodal Medical Diagnosis Web Server on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
