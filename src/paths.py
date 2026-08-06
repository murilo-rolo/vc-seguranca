import os
import sys
from pathlib import Path

IN_COLAB = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE' in os.environ

if IN_COLAB:
    PROJECT_ROOT = Path("/content/drive/Othercomputers/Meu laptop/cv-security-threat-detection-develop")
    DATASET_ROOT = Path("/content/dataset")
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATASET_ROOT = PROJECT_ROOT / "dataset"

DATA_ROOT         = PROJECT_ROOT / "data"
RAW_DATA_ROOT     = DATA_ROOT / "raw"
PROCESSED_ROOT    = DATA_ROOT / "processed"
POSE_ROOT         = DATA_ROOT / "pose"
EMOTION_ROOT      = DATA_ROOT / "emotion"

RESULTS_ROOT      = PROJECT_ROOT / "results"
MODELS_ROOT       = RESULTS_ROOT / "models"
MULTIMODAL_ROOT   = RESULTS_ROOT / "multimodal"
CNN3D_ROOT        = RESULTS_ROOT / "cnn3d"
EMOTION_MODELS_ROOT = RESULTS_ROOT / "emotion"
REPORTS_ROOT      = RESULTS_ROOT / "reports"
EXPERIMENTS_ROOT  = RESULTS_ROOT / "experiments"
COMPARISON_ROOT   = RESULTS_ROOT / "comparison"

RWF2000_ROOT      = DATASET_ROOT / "RWF-2000"
UCF101_ROOT       = DATASET_ROOT / "UCF101"
AFFECTNET_ROOT    = DATASET_ROOT / "AffectNet"

# ── Novos paths por modelo (estrutura organizada) ──────────────────────────────
MODELS_BASE       = PROJECT_ROOT / "models"

# ResNet-LSTM
RESNET_LSTM_ROOT       = MODELS_BASE / "resnet_lstm"
RESNET_LSTM_WEIGHTS    = RESNET_LSTM_ROOT / "weights"
RESNET_LSTM_EXPERIMENTS = RESNET_LSTM_ROOT / "experiments"

# Emotion CNN (DeiT)
EMOTION_CNN_ROOT       = MODELS_BASE / "emotion_cnn"
EMOTION_CNN_WEIGHTS    = EMOTION_CNN_ROOT / "weights"
EMOTION_CNN_EXPERIMENTS = EMOTION_CNN_ROOT / "experiments"

# CNN 3D
CNN3D_MODELS_ROOT      = MODELS_BASE / "cnn3d"
CNN3D_WEIGHTS           = CNN3D_MODELS_ROOT / "weights"
CNN3D_EXPERIMENTS       = CNN3D_MODELS_ROOT / "experiments"
CNN3D_UCF101_WEIGHTS    = CNN3D_WEIGHTS / "ucf101"
CNN3D_RWF2000_WEIGHTS   = CNN3D_WEIGHTS / "rwf2000"
CNN3D_UCF101_EXPERIMENTS = CNN3D_EXPERIMENTS / "ucf101"
CNN3D_RWF2000_EXPERIMENTS = CNN3D_EXPERIMENTS / "rwf2000"

# Multimodal
MULTIMODAL_MODELS_ROOT  = MODELS_BASE / "multimodal"
MULTIMODAL_WEIGHTS      = MULTIMODAL_MODELS_ROOT / "weights"
MULTIMODAL_EXPERIMENTS  = MULTIMODAL_MODELS_ROOT / "experiments"
