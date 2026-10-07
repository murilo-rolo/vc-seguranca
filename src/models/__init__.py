"""
Módulo de modelos para o projeto de detecção de violência em vídeos.
"""

from .emotion_cnn import EmotionNet, create_emotion_model
from .multimodal_risk import MultimodalRiskDetector, create_multimodal_model
from .cnn3d_risk import CNN3DRiskDetector, create_cnn3d_model

__all__ = [
    "EmotionNet",
    "create_emotion_model",
    "MultimodalRiskDetector",
    "create_multimodal_model",
    "CNN3DRiskDetector",
    "create_cnn3d_model"
]

