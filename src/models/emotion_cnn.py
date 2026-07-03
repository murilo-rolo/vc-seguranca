"""
Modelo de CNN para reconhecimento de emoções faciais (FER - Facial Expression Recognition).

Arquitetura baseada em ResNet-34 pré-treinada no ImageNet, adaptada para classificação
de emoções usando o dataset AffectNet.

Classes de emoção (8 classes padrão):
- 0: Neutral
- 1: Happy
- 2: Sad
- 3: Anger
- 4: Fear
- 5: Disgust
- 6: Surprise
- 7: Contempt
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional

try:
    from torchvision.models import ResNet34_Weights
    HAS_WEIGHTS_API = True
except ImportError:
    HAS_WEIGHTS_API = False


class EmotionNet(nn.Module):
    """
    Modelo CNN para classificação de emoções faciais (FER).
    
    Baseado em ResNet-34 pré-treinada no ImageNet, adaptado para FER
    com classifier Linear(512, 8). As camadas de baixo nível
    (conv1, bn1, layer1, layer2) são congeladas para preservar
    features genéricas do ImageNet e evitar overfitting.
    
    Arquitetura:
    1. ResNet-34 pré-treinada (backbone, sem FC)
    2. Conv1 + Layer1 + Layer2 congelados
    3. AdaptiveAvgPool2d (embutida no backbone)
    4. Classifier 1-layer: Linear(512, 8)
    """

    EMOTION_CLASSES = [
        'neutral', 'happy', 'sad', 'anger', 
        'fear', 'disgust', 'surprise', 'contempt'
    ]
    
    def __init__(
        self,
        num_emotions: int = 8,
        pretrained: bool = True,
        input_size: Tuple[int, int] = (224, 224),
    ):
        """
        Args:
            num_emotions: Número de classes de emoção (padrão: 8 para AffectNet)
            pretrained: Se True, usa ResNet-34 pré-treinada no ImageNet
            input_size: Tamanho de entrada (altura, largura) - padrão: (224, 224)
        """
        super(EmotionNet, self).__init__()
        
        self.num_emotions = num_emotions
        self.input_size = input_size
        
        if pretrained:
            if HAS_WEIGHTS_API:
                resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            else:
                resnet = models.resnet34(pretrained=True)
        else:
            if HAS_WEIGHTS_API:
                resnet = models.resnet34(weights=None)
            else:
                resnet = models.resnet34(pretrained=False)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_size = 512
        
        self.backbone[0].requires_grad_(False)
        self.backbone[1].requires_grad_(False)
        self.backbone[4].requires_grad_(False)
        self.backbone[5].requires_grad_(False)
        
        self.classifier = nn.Linear(self.feature_size, num_emotions)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass do modelo.
        
        Args:
            x: Tensor de entrada (batch_size, 3, H, W) - faces detectadas
        
        Returns:
            Tensor de saída (batch_size, num_emotions) com logits
        """
        features = self.backbone(x)                       # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)    # (B, 512)
        logits = self.classifier(features)                 # (B, num_emotions)
        return logits
    
    def predict_emotions(
        self,
        x: torch.Tensor,
        return_probs: bool = True
    ) -> torch.Tensor:
        """
        Prediz emoções com probabilidades.
        
        Args:
            x: Tensor de entrada (batch_size, 3, H, W)
            return_probs: Se True, retorna probabilidades (softmax), senão logits
        
        Returns:
            Tensor (batch_size, num_emotions) com probabilidades ou logits
        """
        logits = self.forward(x)
        
        if return_probs:
            probs = torch.softmax(logits, dim=1)
            return probs
        else:
            return logits
    
    def get_emotion_name(self, emotion_idx: int) -> str:
        """
        Retorna o nome da emoção dado o índice.
        
        Args:
            emotion_idx: Índice da emoção (0-7)
        
        Returns:
            Nome da emoção
        """
        if 0 <= emotion_idx < len(self.EMOTION_CLASSES):
            return self.EMOTION_CLASSES[emotion_idx]
        return "unknown"
    
    @classmethod
    def get_emotion_classes(cls) -> list:
        """Retorna lista de classes de emoção."""
        return cls.EMOTION_CLASSES.copy()


def create_emotion_model(
    num_emotions: int = 8,
    pretrained: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    checkpoint_path: Optional[str] = None,
    resume_training: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Função auxiliar para criar e carregar modelo de emoção.
    
    Args:
        num_emotions: Número de classes de emoção
        pretrained: Se True, usa ResNet-34 pré-treinada
        input_size: Tamanho de entrada
        checkpoint_path: Caminho para checkpoint pré-treinado (opcional)
        resume_training: Se True, retorna (model, checkpoint) e não força eval()
        device: Device para mover o modelo
    
    Returns:
        EmotionNet ou (EmotionNet, dict) se resume_training=True
    """
    model = EmotionNet(
        num_emotions=num_emotions,
        pretrained=pretrained,
        input_size=input_size,
    )
    
    ckpt = None
    
    # Carregar checkpoint se fornecido
    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            
            # Lidar com diferentes formatos de checkpoint
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    model.load_state_dict(ckpt['model_state_dict'])
                elif 'state_dict' in ckpt:
                    model.load_state_dict(ckpt['state_dict'])
                else:
                    model.load_state_dict(ckpt)
            else:
                model.load_state_dict(ckpt)
            
            print(f"Checkpoint carregado de: {checkpoint_path}")
        except Exception as e:
            print(f"Erro ao carregar checkpoint: {e}")
            print("Usando modelo com pesos ImageNet pré-treinados")
            ckpt = None
    
    model = model.to(device)
    
    if resume_training and ckpt is not None:
        model.train()
        return model, ckpt
    
    model.eval()
    return model

