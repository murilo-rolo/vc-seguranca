"""
Modelo ResNet-18 + LSTM para classificação de violência em vídeos.

Arquitetura:
1. ResNet-18 pré-treinada no ImageNet (sem última camada FC)
2. Extração de features por frame (512 dims)
3. LSTM para modelagem temporal
4. Camada FC final para classificação binária (violent/non-violent)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple

# Tentar importar ResNet18_Weights (disponível em torchvision >= 0.13)
try:
    from torchvision.models import ResNet18_Weights
    HAS_WEIGHTS_API = True
except ImportError:
    HAS_WEIGHTS_API = False


class ResNetLSTM(nn.Module):
    """
    Modelo híbrido ResNet-18 + LSTM para classificação de vídeos.
    
    Processa sequências de frames:
    - Cada frame passa por ResNet-18 para extrair features
    - Features temporais são modeladas por LSTM
    - Saída final é classificação binária
    """
    
    def __init__(
        self,
        num_frames: int = 16,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
        pretrained: bool = True
    ):
        """
        Inicializa o modelo.
        
        Args:
            num_frames: Número de frames por vídeo
            hidden_size: Tamanho do hidden state da LSTM
            num_layers: Número de camadas LSTM
            dropout: Taxa de dropout
            num_classes: Número de classes (2: non-violent, violent)
            pretrained: Se True, usa ResNet-18 pré-treinada no ImageNet
        """
        super(ResNetLSTM, self).__init__()
        
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Carregar ResNet-18 pré-treinada
        # Usar sintaxe moderna do torchvision (weights ao invés de pretrained)
        if pretrained:
            if HAS_WEIGHTS_API:
                # Sintaxe moderna (torchvision >= 0.13)
                resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                # Sintaxe antiga (torchvision < 0.13)
                resnet = models.resnet18(pretrained=True)
        else:
            if HAS_WEIGHTS_API:
                resnet = models.resnet18(weights=None)
            else:
                resnet = models.resnet18(pretrained=False)
        
        # Remover a última camada FC e a camada de pooling adaptativa
        # Manter até a camada antes do FC (avgpool)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature size do ResNet-18: 512
        self.feature_size = 512
        
        # LSTM para modelagem temporal
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Camada de dropout
        self.dropout = nn.Dropout(dropout)
        
        # Camada FC final para classificação
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Inicializar pesos da camada FC
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch_size, num_frames, C, H, W)
        
        Returns:
            Tensor de saída (batch_size, num_classes) com logits
        """
        batch_size, num_frames, C, H, W = x.shape
        
        # Redimensionar para processar todos os frames de uma vez
        # (batch_size * num_frames, C, H, W)
        x = x.view(batch_size * num_frames, C, H, W)
        
        # Extrair features com ResNet-18
        # (batch_size * num_frames, 512, 1, 1) após avgpool
        features = self.resnet_features(x)
        
        # Remover dimensões espaciais (1, 1)
        # (batch_size * num_frames, 512)
        features = features.view(batch_size * num_frames, self.feature_size)
        
        # Redimensionar para sequência temporal
        # (batch_size, num_frames, 512)
        features = features.view(batch_size, num_frames, self.feature_size)
        
        # Passar pela LSTM
        # lstm_out: (batch_size, num_frames, hidden_size)
        # hidden: (num_layers, batch_size, hidden_size)
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Usar o último output da LSTM (último frame)
        # (batch_size, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Aplicar dropout
        last_output = self.dropout(last_output)
        
        # Classificação final
        # (batch_size, num_classes)
        output = self.fc(last_output)
        
        return output
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai features do modelo (última camada antes da classificação).
        Útil para análise ou transfer learning.
        
        Args:
            x: Tensor de entrada (batch_size, num_frames, C, H, W)
        
        Returns:
            Features extraídas (batch_size, hidden_size)
        """
        batch_size, num_frames, C, H, W = x.shape
        
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.resnet_features(x)
        features = features.view(batch_size * num_frames, self.feature_size)
        features = features.view(batch_size, num_frames, self.feature_size)
        
        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]
        
        return last_output


def create_model(
    num_frames: int = 16,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.3,
    num_classes: int = 2,
    pretrained: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> ResNetLSTM:
    """
    Função auxiliar para criar e mover o modelo para o device.
    
    Args:
        num_frames: Número de frames por vídeo
        hidden_size: Tamanho do hidden state da LSTM
        num_layers: Número de camadas LSTM
        dropout: Taxa de dropout
        num_classes: Número de classes
        pretrained: Se True, usa ResNet-18 pré-treinada
        device: Device para mover o modelo
    
    Returns:
        Modelo ResNetLSTM no device especificado
    """
    model = ResNetLSTM(
        num_frames=num_frames,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    model = model.to(device)
    
    return model

