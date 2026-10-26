# Multimodal Deep Learning for Explainable Medical Diagnosis

This repository implements the **proposed study protocol** in the accompanying report. The report does not present experimental results; therefore, this repository does not claim measured model performance. Training and inference require a real, permitted dataset in which each image, clinical record, and label are correctly linked to a patient.

This is an academic research prototype, not a medical device. It must not be used to diagnose, triage, or guide treatment.

## Study protocol

The implemented comparison follows the report:

1. Image-only baseline using a CNN image encoder.
2. Clinical-only baseline using an MLP.
3. Multimodal model that concatenates image and clinical embeddings.
4. Missing-data-aware multimodal model trained with simulated missing clinical values.

Clinical continuous values are standardized using training-split statistics only. Missing values are imputed from the training split and accompanied by an observation mask. Patient IDs are kept wholly within one split. The planned analysis reports accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, calibration/Brier score, and performance under repeated 10%, 30%, and 50% simulated missingness.

Image explanations use Grad-CAM; clinical explanations use Integrated Gradients as a feature-attribution method. These are inspection tools, not evidence that a prediction is medically correct or causal.

## Dataset requirements

The project intentionally does not download unrelated images, generate synthetic X-rays, or invent clinical measurements. The report requires the dataset and label definitions to be checked before experiments begin. Prepare a CSV manifest at `data/raw/metadata.csv` with these columns:

| Column | Meaning |
| --- | --- |
| `patient_id` | Stable patient identifier used to prevent patient overlap between splits |
| `image_path` | Path to a real image, absolute or relative to `data/raw/metadata.csv` |
| `label` | Verified binary target: `0` or `1`, defined for the selected dataset |
| `age` | Age in years |
| `sex` | Encoded category, `0` or `1` |
| `temperature` | Temperature in degrees Celsius |
| `oxygen_saturation` | SpO2 percentage |
| `heart_rate` | Beats per minute |
| `respiratory_rate` | Breaths per minute |
| `leukocyte_count` | WBC count in `10^9/L` |
| `cough`, `dyspnea`, `fever` | Encoded symptom categories, `0` or `1` |

Clinical cells may be blank to represent values that were not recorded. Do not replace unknown values with guessed values or values derived from the label. Every configured clinical feature needs at least one observed value in the training split. Each patient must have a consistent label, and the current stratified 70/15/15 split requires at least six distinct patients in each class. Rows for a patient with multiple images remain in the same split.

The current model configuration expects the ten clinical features listed above. If the selected dataset uses different clinical fields or label definitions, update the feature configuration and document the mapping before training; do not present mismatched or fabricated values as patient data. Follow the source dataset's access, license, privacy, and data-use requirements.

Unprovenanced model weights, cached splits, or preprocessors already present in a working copy are not trusted for inference. The app and CLI require provenance written by a completed training run. Previously generated plots and sample-patient profiles are not reported as study results.

Example manifest layout:

```text
data/
  raw/
    metadata.csv
    images/
      study-001.png
```

Example CSV header (the sample values below are placeholders, not patient data):

```csv
patient_id,image_path,label,age,sex,temperature,oxygen_saturation,heart_rate,respiratory_rate,leukocyte_count,cough,dyspnea,fever
```

## Setup (Windows)

Use Python 3.10 or newer. From the project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlow may download the ImageNet DenseNet weights the first time training runs. The training pipeline can take substantial time and compute.

## Run

Validate the dataset, create patient-level splits, train all four models, evaluate them, and generate plots:

```powershell
.\.venv\Scripts\python.exe train.py
```

The pipeline stops with a clear error if `data/raw/metadata.csv` is absent or invalid. Successful training writes processed splits, the fitted preprocessor, model weights and provenance metadata, plus evaluated metrics and figures.

Start the web dashboard:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:8000`. The dashboard can be opened before training, but diagnosis requests remain unavailable until trained artifacts with provenance exist. The benchmark page shows a “not evaluated” state until the evaluation pipeline completes.

For command-line inference, provide an image and a JSON file containing real values for the configured clinical features:

```powershell
.\.venv\Scripts\python.exe demo_cli.py --image data\raw\images\study-001.png --clinical-json patient.json --model multimodal
```

The CLI and web application reject untrained/unverified model artifacts rather than returning predictions from random or legacy weights. A CLI JSON example must contain actual patient measurements and is not included in this repository.

## Deploy (GitHub, Netlify, and Render)

The dashboard is a static site on Netlify; the FastAPI API runs separately on Render. The frontend sends API requests directly to the Render service, and `netlify.toml` proxies only aggregate-figure requests. Direct API requests avoid Netlify's proxy timeout for longer inference requests.

1. Push the source repository to GitHub. Raw and processed data, model weights, and generated figures are ignored by default. Do not force-add patient data, credentials, or unverified model artifacts.
2. In Netlify, import the GitHub repository. Netlify reads `netlify.toml` and publishes the `web` directory. Note the site's production URL.
3. In Render, create a new Blueprint and select this repository. Render reads `render.yaml`; enter the exact Netlify site origin (for example, `https://your-site.netlify.app`) for `CORS_ORIGINS`. Wait for the `/api/benchmarks` health check to pass.
4. Open the Netlify site and verify that the benchmark page reports the current study status.

The hosted dashboard can display the protocol and "not evaluated" state without trained artifacts. This repository does not contain a verified patient-linked manifest or trusted trained model artifacts, so diagnosis requests will correctly return an unavailable response until the study is run on approved data. Uploaded images and clinical values are sent to the Render backend; deploy only after confirming that your data-use and privacy requirements allow this. Do not upload patient data or trained artifacts to a public Git repository. If trained artifacts are later deployed, use an appropriately secured persistent storage option on the backend; the free Render service has ephemeral storage and resource limits and is not suitable for dependable model inference.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Repository layout

```text
app.py                         FastAPI dashboard and inference API
demo_cli.py                    Command-line inference
train.py                       Training, evaluation, and figure generation
src/config.py                  Model and clinical-feature configuration
src/data/dataset_downloader.py Patient-linked manifest validation and splitting
src/data/preprocessor.py       Image/clinical preprocessing and TensorFlow input pipeline
src/data/missing_simulator.py  Natural and simulated missing-value masks
src/models/                    Image, clinical, and fusion model builders
src/explainability/            Grad-CAM, Integrated Gradients, stability checks
src/evaluation/                Metrics and plots
tests/                         Unit tests
web/index.html                 Dashboard UI
```

## Clinical and research limitations

Model quality depends on the selected dataset, its patient-level matching, label validity, and representativeness. Randomly masking clinical fields is only a controlled experiment and does not model every real-world reason data may be missing. Attribution and heatmap methods can be unstable or misleading. External validation and expert review would be required before considering any clinical application.
