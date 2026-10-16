"""
Modelo de Vision Transformer para reconhecimento de emoções faciais (FER).

Arquitetura: DeiT-Small (22M parâmetros, embed_dim=384) pré-treinada no
ImageNet, com classifier 2-layer (384 → 128 → num_emotions).

Classes (2): 0 violent / 1 non_violent.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import timm


class EmotionNet(nn.Module):
    """DeiT-Small adaptada para classificação binária de emoções faciais."""

    EMOTION_CLASSES = ['violent', 'non_violent']

    def __init__(
        self,
        num_emotions: int = 2,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        self.num_emotions = num_emotions
        self.input_size = input_size

        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            drop_path_rate=0.1,
        )
        self.feature_size = self.backbone.num_features  # 384

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_emotions),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: (B, 3, H, W) -> logits (B, num_emotions)."""
        return self.classifier(self.backbone(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embedding da penúltima camada do classifier (B, 128):
        ativação pós-ReLU antes do Linear final.
        """
        return self.classifier[:3](self.backbone(x))

    def predict_emotions(self, x: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """Prediz emoções (softmax por padrão, ou logits)."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1) if return_probs else logits

    def get_emotion_name(self, emotion_idx: int) -> str:
        if 0 <= emotion_idx < len(self.EMOTION_CLASSES):
            return self.EMOTION_CLASSES[emotion_idx]
        return "unknown"

    @classmethod
    def get_emotion_classes(cls) -> list:
        return cls.EMOTION_CLASSES.copy()


def create_emotion_model(
    num_emotions: int = 2,
    pretrained: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    checkpoint_path: Optional[str] = None,
    resume_training: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Cria o modelo de emoção e carrega um checkpoint (opcional).

    Returns:
        EmotionNet, ou (EmotionNet, checkpoint) se resume_training=True.
    """
    model = EmotionNet(
        num_emotions=num_emotions,
        pretrained=pretrained,
        input_size=input_size,
    )

    ckpt = None
    if checkpoint_path is not None:
        try:
            from .checkpoints import load_state
            ckpt = load_state(model, checkpoint_path, device=device)
            print(f"Checkpoint carregado de: {checkpoint_path}")
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            print("Usando modelo com pesos ImageNet pré-treinados")

    model = model.to(device)

    if resume_training and ckpt is not None:
        model.train()
        return model, ckpt

    model.eval()
    return model
