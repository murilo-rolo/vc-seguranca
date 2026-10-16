"""
Modelo CNN 3D para detecção de risco/violência em vídeos.
"""

from typing import Optional, Literal

import torch
import torch.nn as nn

try:
    import torchvision.models.video as video_models
    HAS_VIDEO_MODELS = True
except ImportError:
    HAS_VIDEO_MODELS = False
    print("Aviso: torchvision.models.video não disponível. Use torchvision >= 0.13.0")

_AVAILABLE_MODELS = {
    "r3d_18": "R3D_18_Kinetics400_Weights",
    "r2plus1d_18": "R2Plus1D_18_Kinetics400_Weights",
    "mc3_18": "MC3_18_Kinetics400_Weights",
}


def ensure_channels_first(x: torch.Tensor) -> torch.Tensor:
    """
    Converte clips frame-last (B, T, C, H, W) para o formato torchvision
    (B, C, T, H, W). Tensores já em (B, C, T, H, W) são devolvidos iguais.
    """
    if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
        x = x.permute(0, 2, 1, 3, 4)
    return x


def _create_backbone(model_name: str, pretrained: bool) -> nn.Module:
    if not HAS_VIDEO_MODELS:
        raise ImportError(
            "torchvision.models.video não disponível. "
            "Instale torchvision >= 0.13.0: pip install torchvision>=0.13.0"
        )
    if model_name not in _AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo não suportado: {model_name}. Opções: {list(_AVAILABLE_MODELS.keys())}"
        )
    constructor = getattr(video_models, model_name)
    weights = None
    if pretrained:
        weights_enum = getattr(video_models, _AVAILABLE_MODELS[model_name], None)
        if weights_enum is not None:
            try:
                weights = weights_enum.DEFAULT
            except Exception:
                weights = None
    return constructor(weights=weights)


def _strip_classifier(backbone: nn.Module) -> tuple[nn.Module, int]:
    """
    Remove a camada final (Linear) do backbone e retorna (backbone, feature_dim).

    A remoção usa Sequential(*children[:-1]) para preservar as chaves dos
    checkpoints já treinados (backbone.0.*, backbone.1.*, ...).
    """
    for attr in ("fc", "head"):
        head = getattr(backbone, attr, None)
        if isinstance(head, nn.Linear):
            feature_dim = head.in_features
            children = list(backbone.children())
            if children and children[-1] is head:
                return nn.Sequential(*children[:-1]), feature_dim
    children = list(backbone.children())
    if children and isinstance(children[-1], nn.Linear):
        return nn.Sequential(*children[:-1]), children[-1].in_features
    return backbone, 512


class CNN3DRiskDetector(nn.Module):
    """
    Modelo CNN 3D para detecção de risco/violência.

    Input: (batch, T, C, H, W) ou (batch, C, T, H, W) — auto-detectado.
    Output: Logits (batch, num_classes)
    """

    def __init__(
        self,
        model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18"] = "r2plus1d_18",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.num_frames = 16

        backbone = _create_backbone(model_name, pretrained)
        self.backbone, self.feature_dim = _strip_classifier(backbone)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai o vetor clip-level (batch, feature_dim) antes de dropout e
        classificador — útil como clip token (B, D_v) para o modelo multimodal.
        """
        features = self.backbone(ensure_channels_first(x))
        if features.dim() > 2:
            features = features.flatten(1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.get_features(x)))


def create_cnn3d_model(
    model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18"] = "r2plus1d_18",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> CNN3DRiskDetector:
    """Cria o modelo CNN 3D (sem pesos Kinetics quando há checkpoint customizado)."""
    model = CNN3DRiskDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained and checkpoint_path is None,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
    model.num_frames = num_frames

    if checkpoint_path is not None:
        from .checkpoints import load_state
        load_state(model, checkpoint_path, device=device)
        print(f"✓ Checkpoint carregado de: {checkpoint_path}")

    return model.to(device)
