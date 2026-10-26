import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_PATIENTS_DIR = DATA_DIR / "sample_patients"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_PATIENTS_DIR, SAVED_MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Image Configuration
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (224, 224, 3)
IMAGE_EMBEDDING_DIM = 256
IMAGE_BACKBONE = "DenseNet121"  # "DenseNet121" or "ResNet50"
GRAD_CAM_LAYER_NAME = "conv5_block16_concat"  # Last conv block of DenseNet121

# Clinical Data Configuration
CONTINUOUS_FEATURES = [
    "age",                  # Years
    "temperature",          # Degrees Celsius (normal ~36.8)
    "oxygen_saturation",    # SpO2 % (normal ~98)
    "heart_rate",           # Beats per minute (normal ~75)
    "respiratory_rate",     # Breaths per minute (normal ~16)
    "leukocyte_count",      # WBC 10^9/L (normal ~6.5)
]

CATEGORICAL_FEATURES = [
    "sex",                  # 0: Female, 1: Male
    "cough",                # 0: No, 1: Yes
    "dyspnea",              # 0: No, 1: Yes (Shortness of breath)
    "fever",                # 0: No, 1: Yes
]

CLINICAL_FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
NUM_CLINICAL_FEATURES = len(CLINICAL_FEATURES)  # 10 features
CLINICAL_INPUT_DIM = NUM_CLINICAL_FEATURES * 2   # 10 features + 10 mask indicators
CLINICAL_EMBEDDING_DIM = 64

# Fusion & Classification Head
FUSED_DIM = IMAGE_EMBEDDING_DIM + CLINICAL_EMBEDDING_DIM  # 320
CLASSIFIER_HIDDEN_DIMS = [128, 64]
DROPOUT_RATE = 0.3
NUM_CLASSES = 1  # Binary: 0 = Normal / Non-disease, 1 = Pneumonia/Pathology
DIAGNOSIS_LABELS = {0: "Normal / Healthy", 1: "Pneumonia / Pathology"}

# Missing Data Simulation
MASKING_RATES = [0.0, 0.1, 0.3, 0.5]  # Evaluated as per Section III-G and Fig 3a

# Training Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 15
PATIENCE = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Explainability Parameters
INTEGRATED_GRADIENTS_STEPS = 50
STABILITY_PERTURBATION_SIGMA = 0.05
PERTURBATION_NUM_RUNS = 20
