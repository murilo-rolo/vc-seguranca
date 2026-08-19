"""
Modelo CNN 3D para detecção de risco/violência em vídeos.
"""

import torch
import torch.nn as nn
from typing import Optional, Literal

try:
    import torchvision.models.video as video_models
    HAS_VIDEO_MODELS = True
except ImportError:
    HAS_VIDEO_MODELS = False
    print("Aviso: torchvision.models.video não disponível. Use torchvision >= 0.13.0")

_AVAILABLE_MODELS = {
    "r3d_18": {"constructor": "r3d_18", "weights_enum": "R3D_18_Kinetics400_Weights"},
    "r2plus1d_18": {"constructor": "r2plus1d_18", "weights_enum": "R2Plus1D_18_Kinetics400_Weights"},
    "mc3_18": {"constructor": "mc3_18", "weights_enum": "MC3_18_Kinetics400_Weights"},
}

_FEATURE_DIM_MAP = {"r3d_18": 512, "r2plus1d_18": 512, "mc3_18": 512}


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
    info = _AVAILABLE_MODELS[model_name]
    constructor = getattr(video_models, info["constructor"])
    if pretrained:
        weights_enum = getattr(video_models, info["weights_enum"], None)
        if weights_enum is not None:
            try:
                return constructor(weights=weights_enum.DEFAULT)
            except Exception:
                return constructor(weights=None)
        return constructor(weights=None)
    return constructor(weights=None)


def _get_feature_dim(backbone: nn.Module, model_name: str) -> int:
    if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
        return backbone.fc.in_features
    if hasattr(backbone, 'head') and hasattr(backbone.head, 'in_features'):
        return backbone.head.in_features
    children = list(backbone.children())
    if children:
        last = children[-1]
        if isinstance(last, nn.Linear):
            return last.in_features
    return _FEATURE_DIM_MAP.get(model_name, 512)


def _strip_classifier(backbone: nn.Module) -> nn.Module:
    children = list(backbone.children())
    if children and isinstance(children[-1], (nn.Linear, nn.Dropout)):
        return nn.Sequential(*children[:-1])
    return backbone


class CNN3DRiskDetector(nn.Module):
    """
    Modelo CNN 3D para detecção de risco/violência.

    Input: (batch, T, C, H, W) ou (batch, C, T, H, W)
    Output: Logits (batch, num_classes)
    """

    def __init__(
        self,
        model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18", "mvit"] = "r2plus1d_18",
        num_classes: int = 2,
        pretrained: bool = True,
        pretrained_dataset: str = "kinetics400",
        dropout: float = 0.5,
        freeze_backbone: bool = False
    ):
        super(CNN3DRiskDetector, self).__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.pretrained_dataset = pretrained_dataset
        self.num_frames = 16

        self.backbone = _create_backbone(model_name, pretrained)
        feature_dim = _get_feature_dim(self.backbone, model_name)
        self.backbone = _strip_classifier(self.backbone)
        self.feature_dim = feature_dim

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Detecta automaticamente:
        - (batch, C, T, H, W) -> mantido (formato torchvision)
        - (batch, T, C, H, W) -> convertido para (batch, C, T, H, W)
        """
        if len(x.shape) == 5:
            _b, dim1, dim2, _h, _w = x.shape
            if dim2 == 3 and dim1 != 3:
                x = x.permute(0, 2, 1, 3, 4)

        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        features = self.dropout(features)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai o vetor clip-level antes da classificação (backbone -> flatten).

        Retorna o vetor (batch, feature_dim) da penúltima camada, SEM dropout
        e SEM classifier. Útil como clip token (B, D_v) para o modelo multimodal.

        Detecta automaticamente:
        - (batch, C, T, H, W) -> mantido (formato torchvision)
        - (batch, T, C, H, W) -> convertido para (batch, C, T, H, W)

        Args:
            x: Tensor de entrada (batch, T, C, H, W) ou (batch, C, T, H, W)

        Returns:
            Features (batch, feature_dim)
        """
        if len(x.shape) == 5:
            _b, dim1, dim2, _h, _w = x.shape
            if dim2 == 3 and dim1 != 3:
                x = x.permute(0, 2, 1, 3, 4)

        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        return features
        
        # Extrair features com backbone
        # Backbone retorna (batch, feature_dim, 1, 1, 1) após pooling
        features = self.backbone(x)
        
        # Remover dimensões espaciais e temporais
        # features shape: (batch, feature_dim, 1, 1, 1) ou (batch, feature_dim)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Aplicar dropout
        features = self.dropout(features)
        
        # Classificação final
        logits = self.classifier(features)
        
        return logits
    
    def freeze_backbone_layers(self, num_layers: int = -1):
        """
        Congela as primeiras N camadas do backbone.
        
        Args:
            num_layers: Número de camadas para congelar (-1 = todas)
        """
        if num_layers == -1:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Congelar primeiras N camadas
            layers = list(self.backbone.children())
            for i, layer in enumerate(layers[:num_layers]):
                for param in layer.parameters():
                    param.requires_grad = False


def create_cnn3d_model(
    model_name: Literal["r3d_18", "r2plus1d_18", "mc3_18"] = "r2plus1d_18",
    num_classes: int = 2,
    pretrained: bool = True,
    pretrained_dataset: str = "kinetics400",
    dropout: float = 0.5,
    freeze_backbone: bool = False,
    checkpoint_path: Optional[str] = None,
    num_frames: int = 16,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> CNN3DRiskDetector:
    """
    Função auxiliar para criar modelo CNN 3D.
    
    Args:
        model_name: Nome do modelo 3D
        num_classes: Número de classes
        pretrained: Se True, usa pesos pré-treinados
        pretrained_dataset: Dataset de pré-treinamento
        dropout: Taxa de dropout
        freeze_backbone: Se True, congela backbone
        checkpoint_path: Caminho para checkpoint customizado (opcional)
        device: Device para mover o modelo
    
    Returns:
        Modelo CNN3DRiskDetector no device especificado
    """
    model = CNN3DRiskDetector(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained and checkpoint_path is None,  # Não usar pretrained se checkpoint fornecido
        pretrained_dataset=pretrained_dataset,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    # Definir num_frames
    model.num_frames = num_frames
    
    # Carregar checkpoint customizado se fornecido
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Checkpoint carregado de: {checkpoint_path}")
    
    model = model.to(device)
    return model

