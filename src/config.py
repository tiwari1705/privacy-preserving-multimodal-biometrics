# Final configuration for the privacy-preserving
# multimodal biometric verification project.

# -----------------------------
# Dataset
# -----------------------------

NUM_VIRTUAL_SUBJECTS = 600
SAMPLES_PER_MODALITY = 10
NUM_MODALITIES = 3

TOTAL_SAMPLES = 18000

TRAIN_SUBJECTS = 480
VALIDATION_SUBJECTS = 60
TEST_SUBJECTS = 60

GENUINE_PAIRS = 2700
IMPOSTOR_PAIRS = 2700
PAIRS_PER_SPLIT = 5400


# -----------------------------
# Modalities
# -----------------------------

MODALITIES = [
    "face",
    "fingerprint",
    "iris"
]


# -----------------------------
# Embedding models
# -----------------------------

BASE_EMBEDDING_DIM = 512

EMBEDDING_MODELS = {
    "face": "InceptionResnetV1",
    "fingerprint": "ResNet-based model",
    "iris": "EfficientNet-B3"
}


# -----------------------------
# Final protected configurations
# -----------------------------

PROTECTED_CONFIG = {
    "face": {
        "projection_dim": 512,
        "binary": False
    },
    "fingerprint": {
        "projection_dim": 1024,
        "binary": True
    },
    "iris": {
        "projection_dim": 256,
        "binary": False
    }
}


# -----------------------------
# Final trimodal fusion weights
# -----------------------------

TRIMODAL_WEIGHTS = {
    "face": 0.35,
    "fingerprint": 0.25,
    "iris": 0.40
}


# -----------------------------
# CKKS configuration
# -----------------------------

CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = [60, 40, 40, 60]
CKKS_GLOBAL_SCALE = 2 ** 40
CKKS_LIBRARY = "TenSEAL"


# -----------------------------
# Evaluation metrics
# -----------------------------

EVALUATION_METRICS = [
    "EER",
    "ROC-AUC",
    "Accuracy",
    "TAR@FAR=1%",
    "TAR@FAR=0.1%"
]


# -----------------------------
# Evaluation modes
# -----------------------------

EVALUATION_MODES = [
    "plain",
    "protected",
    "protected_ckks",
    "same_key",
    "cross_key"
]
