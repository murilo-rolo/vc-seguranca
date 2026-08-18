"""
Modelo Multimodal para Detecção de Risco/Violência.

Este módulo implementa fusão de múltiplas modalidades:
- Video Features (ResNet-LSTM ou CNN3D)
- Pose Features (keypoints de pose)
- Emotion Features (vetores de emoção facial)

Estratégia de fusão (AD-005):
- Cross-Attention: cada modalidade atende a concatenação das outras duas
  (MultiheadAttention), seguida de pooling (média) e classificação.
"""

import torch
import torch.nn as nn
from typing import Literal


class MultimodalRiskDetector(nn.Module):
    """
    Modelo multimodal para detecção de risco/violência.

    Combina três modalidades:
    - Video: features de vídeo como clip token (D_v) ou sequência (T, D_v)
    - Pose: keypoints de pose (T x D_p)
    - Emotion: vetores de emoção (T x D_e)

    Arquitetura:
    1. Projeção por modalidade para o espaço de fusão (fusion_dim)
    2. Cross-attention: cada modalidade atende a concatenação das outras duas
    3. Pooling (média) dos tokens atendidos
    4. Classificação binária (violent/non-violent)

    Nota (temporal aggregation): o vídeo chega como um clip token e é
    projetado diretamente (sem LSTM). A LSTM é usada apenas para pose e
    emoção (sequências frame-a-frame sem modelagem temporal interna).
    """

    def __init__(
        self,
        # Dimensões de entrada
        video_feature_dim: int = 512,      # D_v: saída do CNN 3D (padrão) / ResNet-LSTM
        pose_feature_dim: int = 99,        # D_p: 33 joints * 3 (x, y, visibility) se flatten
        emotion_feature_dim: int = 128,    # D_e: 128-d embeddings de emoção
        num_frames: int = 16,              # T: tamanho da janela temporal

        # Dimensões de processamento
        video_hidden_dim: int = 128,       # Dimensão após processamento de vídeo
        pose_hidden_dim: int = 64,         # Dimensão após processamento de pose
        emotion_hidden_dim: int = 32,      # Dimensão após processamento de emoção

        # Fusão
        fusion_dim: int = 256,             # Dimensão do espaço de fusão
        fusion_method: Literal["cross_attention"] = "cross_attention",

        # Classificação
        num_classes: int = 2,              # 2: violent/non-violent
        dropout: float = 0.5,

        # Processamento temporal
        use_temporal_modeling: bool = True,  # LSTM apenas para pose/emotion
        temporal_hidden_size: int = 64
    ):
        """
        Inicializa o modelo multimodal.

        Args:
            video_feature_dim: Dimensão das features de vídeo (D_v)
            pose_feature_dim: Dimensão das features de pose (D_p)
            emotion_feature_dim: Dimensão das features de emoção (D_e)
            num_frames: Número de frames na sequência (T)
            video_hidden_dim: Dimensão oculta para processamento de vídeo
            pose_hidden_dim: Dimensão oculta para processamento de pose
            emotion_hidden_dim: Dimensão oculta para processamento de emoção
            fusion_dim: Dimensão do espaço de fusão
            fusion_method: Método de fusão ("cross_attention")
            num_classes: Número de classes de saída
            dropout: Taxa de dropout
            use_temporal_modeling: Se True, usa LSTM para pose/emotion
            temporal_hidden_size: Tamanho do hidden state para LSTM temporal

        Raises:
            ValueError: Se fusion_method != "cross_attention".
        """
        if fusion_method != "cross_attention":
            raise ValueError(
                f"Método de fusão não suportado: {fusion_method}. "
                f"Suportado apenas: 'cross_attention'"
            )

        super(MultimodalRiskDetector, self).__init__()

        self.video_feature_dim = video_feature_dim
        self.pose_feature_dim = pose_feature_dim
        self.emotion_feature_dim = emotion_feature_dim
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        self.use_temporal_modeling = use_temporal_modeling

        # Vídeo: projeção direta (sem LSTM). O backbone de vídeo já modela
        # tempo; o clip token (B, D_v) é projetado diretamente. Se um input
        # legado (B, T, D_v) chegar, ele é mean-pooled no forward.
        self.video_proj = nn.Linear(video_feature_dim, fusion_dim)

        # Pose: LSTM (último timestep) se temporal, senão MLP; depois projeção
        if use_temporal_modeling:
            self.pose_processor = nn.LSTM(
                pose_feature_dim,
                temporal_hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0
            )
            pose_output_dim = temporal_hidden_size
        else:
            self.pose_processor = nn.Sequential(
                nn.Linear(pose_feature_dim, pose_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            pose_output_dim = pose_hidden_dim
        self.pose_proj = nn.Linear(pose_output_dim, fusion_dim)

        # Emotion: mesmo padrão da pose
        if use_temporal_modeling:
            self.emotion_processor = nn.LSTM(
                emotion_feature_dim,
                temporal_hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0
            )
            emotion_output_dim = temporal_hidden_size
        else:
            self.emotion_processor = nn.Sequential(
                nn.Linear(emotion_feature_dim, emotion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            emotion_output_dim = emotion_hidden_dim
        self.emotion_proj = nn.Linear(emotion_output_dim, fusion_dim)

        # Cross-attention: para cada modalidade m, query = token projetado de m,
        # key/value = concatenação dos tokens das OUTRAS duas modalidades.
        self.cross_attn_video = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_pose = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.cross_attn_emotion = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Camadas de fusão e classificação
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classificador final
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

        # Inicializar pesos
        self._initialize_weights()

    def _initialize_weights(self):
        """Inicializa pesos das camadas."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _process_temporal(
        self,
        processor: nn.Module,
        features: torch.Tensor,
        is_lstm: bool = False
    ) -> torch.Tensor:
        """
        Processa features temporais.

        Args:
            processor: Processador (LSTM ou MLP)
            features: Features de entrada (batch, T, D)
            is_lstm: Se True, processor é LSTM

        Returns:
            Features processadas (batch, D_out)
        """
        # Verificar e ajustar shape se necessário
        if len(features.shape) == 2:
            # (batch, D) -> (batch, 1, D)
            features = features.unsqueeze(1)
        elif len(features.shape) != 3:
            raise ValueError(
                f"Features devem ter shape (batch, T, D) ou (batch, D), "
                f"mas recebeu shape {features.shape}"
            )

        if is_lstm:
            # LSTM retorna (batch, T, hidden_size)
            lstm_out, _ = processor(features)
            # Usar último timestep
            return lstm_out[:, -1, :]
        else:
            # MLP processa cada timestep
            batch_size, T, D = features.shape
            features_flat = features.view(batch_size * T, D)
            processed = processor(features_flat)
            processed = processed.view(batch_size, T, -1)
            # Média temporal
            return processed.mean(dim=1)

    def forward(
        self,
        video_features: torch.Tensor,
        pose_features: torch.Tensor,
        emotion_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass do modelo multimodal.

        Args:
            video_features: Features de vídeo (batch, D_v) clip token ou
                (batch, T, D_v) legado (mean-pooled sobre T)
            pose_features: Features de pose (batch, T, D_p) ou
                (batch, T, num_joints, 3)
            emotion_features: Features de emoção (batch, T, D_e)

        Returns:
            Logits de classificação (batch, num_classes)
        """
        batch_size = video_features.shape[0]

        # Vídeo: clip token (B, D_v) projetado diretamente (sem LSTM).
        # Tolerar legado (B, T, D_v) mean-pooling sobre T.
        if len(video_features.shape) == 3:
            video_features = video_features.mean(dim=1)  # (B, D_v)
        video_proj = self.video_proj(video_features)  # (B, fusion_dim)

        if len(pose_features.shape) == 4 and pose_features.shape[-1] == 3:
            # Pose é (batch, T, num_joints, 3), flatten para (batch, T, num_joints*3)
            pose_features = pose_features.view(batch_size, pose_features.shape[1], -1)
        elif len(pose_features.shape) == 2:
            pose_features = pose_features.unsqueeze(1)  # (batch, 1, D_p)

        if len(emotion_features.shape) == 2:
            emotion_features = emotion_features.unsqueeze(1)  # (batch, 1, D_e)

        # Padding por último timestep apenas para pose/emotion.
        # O vídeo entra como clip token e NÃO é padded.
        T = max(pose_features.shape[1], emotion_features.shape[1])

        if pose_features.shape[1] < T:
            last_frame = pose_features[:, -1:, :]
            padding = last_frame.repeat(1, T - pose_features.shape[1], 1)
            pose_features = torch.cat([pose_features, padding], dim=1)

        if emotion_features.shape[1] < T:
            last_frame = emotion_features[:, -1:, :]
            padding = last_frame.repeat(1, T - emotion_features.shape[1], 1)
            emotion_features = torch.cat([emotion_features, padding], dim=1)

        # Processar pose/emotion (a LSTM nunca é aplicada ao vídeo)
        pose_out = self._process_temporal(
            self.pose_processor, pose_features, is_lstm=self.use_temporal_modeling
        )  # (B, pose_output_dim)
        emotion_out = self._process_temporal(
            self.emotion_processor, emotion_features, is_lstm=self.use_temporal_modeling
        )  # (B, emotion_output_dim)

        pose_proj = self.pose_proj(pose_out)           # (B, fusion_dim)
        emotion_proj = self.emotion_proj(emotion_out)  # (B, fusion_dim)

        # Tokens das três modalidades
        tokens = torch.stack([video_proj, pose_proj, emotion_proj], dim=1)  # (B, 3, fusion_dim)

        # Cross-attention: cada modalidade atende a concatenação das outras duas
        video_att, _ = self.cross_attn_video(
            tokens[:, 0:1],
            torch.cat([tokens[:, 1:2], tokens[:, 2:3]], dim=1),
            torch.cat([tokens[:, 1:2], tokens[:, 2:3]], dim=1)
        )  # (B, 1, fusion_dim)

        pose_att, _ = self.cross_attn_pose(
            tokens[:, 1:2],
            torch.cat([tokens[:, 0:1], tokens[:, 2:3]], dim=1),
            torch.cat([tokens[:, 0:1], tokens[:, 2:3]], dim=1)
        )  # (B, 1, fusion_dim)

        emotion_att, _ = self.cross_attn_emotion(
            tokens[:, 2:3],
            torch.cat([tokens[:, 0:1], tokens[:, 1:2]], dim=1),
            torch.cat([tokens[:, 0:1], tokens[:, 1:2]], dim=1)
        )  # (B, 1, fusion_dim)

        attended = torch.cat([video_att, pose_att, emotion_att], dim=1)  # (B, 3, fusion_dim)
        fusion_input = attended.mean(dim=1)  # (B, fusion_dim)

        # Fusão e classificação
        fused = self.fusion_layers(fusion_input)
        logits = self.classifier(fused)

        return logits


def create_multimodal_model(
    video_feature_dim: int = 512,
    pose_feature_dim: int = 99,
    emotion_feature_dim: int = 128,
    num_frames: int = 16,
    fusion_method: Literal["cross_attention"] = "cross_attention",
    use_temporal_modeling: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs
) -> MultimodalRiskDetector:
    """
    Função auxiliar para criar modelo multimodal.

    Args:
        video_feature_dim: Dimensão das features de vídeo
        pose_feature_dim: Dimensão das features de pose
        emotion_feature_dim: Dimensão das features de emoção
        num_frames: Número de frames
        fusion_method: Método de fusão ("cross_attention")
        use_temporal_modeling: Se True, usa LSTM para pose/emotion
        device: Device para mover o modelo
        **kwargs: Argumentos adicionais para MultimodalRiskDetector

    Returns:
        Modelo MultimodalRiskDetector no device especificado
    """
    model = MultimodalRiskDetector(
        video_feature_dim=video_feature_dim,
        pose_feature_dim=pose_feature_dim,
        emotion_feature_dim=emotion_feature_dim,
        num_frames=num_frames,
        fusion_method=fusion_method,
        use_temporal_modeling=use_temporal_modeling,
        **kwargs
    )

    model = model.to(device)
    return model
