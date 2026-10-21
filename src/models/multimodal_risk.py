"""
Modelo Multimodal para Detecção de Risco/Violência.

Fusão de três modalidades via cross-attention (AD-005):
- Video (clip token do CNN 3D), Pose (keypoints) e Emotion (embeddings).
Cada modalidade atende a concatenação das outras duas (MultiheadAttention),
seguida de pooling (média) e classificação binária.
"""

from typing import Literal

import torch
import torch.nn as nn


def _build_temporal_processor(
    input_dim: int,
    temporal_hidden: int,
    mlp_hidden: int,
    use_temporal: bool,
    dropout: float,
) -> tuple[nn.Module, int]:
    """Processador por modalidade: LSTM (último timestep) ou MLP (média temporal)."""
    if use_temporal:
        return nn.LSTM(
            input_dim, temporal_hidden, num_layers=1, batch_first=True, dropout=0
        ), temporal_hidden
    return nn.Sequential(
        nn.Linear(input_dim, mlp_hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
    ), mlp_hidden


def _pad_to_length(x: torch.Tensor, length: int) -> torch.Tensor:
    """Repete o último timestep até atingir `length` (alinhamento pose/emotion)."""
    if x.shape[1] < length:
        padding = x[:, -1:, :].repeat(1, length - x.shape[1], 1)
        x = torch.cat([x, padding], dim=1)
    return x


class MultimodalRiskDetector(nn.Module):
    """
    Modelo multimodal para detecção de risco/violência.

    - Vídeo chega como clip token (B, D_v) e é projetado diretamente
      (o backbone CNN 3D já modela o tempo; sequência legada é mean-pooled).
    - Pose e emoção passam por LSTM (use_temporal_modeling) ou MLP.
    """

    def __init__(
        self,
        video_feature_dim: int = 512,
        pose_feature_dim: int = 51,
        emotion_feature_dim: int = 128,
        num_frames: int = 16,
        video_hidden_dim: int = 128,
        pose_hidden_dim: int = 64,
        emotion_hidden_dim: int = 32,
        fusion_dim: int = 256,
        fusion_method: Literal["cross_attention"] = "cross_attention",
        num_classes: int = 2,
        dropout: float = 0.3,
        use_temporal_modeling: bool = True,
        temporal_hidden_size: int = 64,
        emotion_dropout_prob: float = 0.0,
        pose_dropout_prob: float = 0.0,
        aux_video_weight: float = 0.0,
    ):
        if fusion_method != "cross_attention":
            raise ValueError(
                f"Método de fusão não suportado: {fusion_method}. "
                f"Suportado apenas: 'cross_attention'"
            )

        super().__init__()

        self.video_feature_dim = video_feature_dim
        self.pose_feature_dim = pose_feature_dim
        self.emotion_feature_dim = emotion_feature_dim
        self.num_frames = num_frames
        self.fusion_method = fusion_method
        self.use_temporal_modeling = use_temporal_modeling
        self.emotion_dropout_prob = float(emotion_dropout_prob)
        self.pose_dropout_prob = float(pose_dropout_prob)
        self.aux_video_weight = float(aux_video_weight)

        self.video_proj = nn.Linear(video_feature_dim, fusion_dim)

        self.pose_processor, pose_out_dim = _build_temporal_processor(
            pose_feature_dim, temporal_hidden_size, pose_hidden_dim,
            use_temporal_modeling, dropout,
        )
        self.pose_proj = nn.Linear(pose_out_dim, fusion_dim)

        self.emotion_processor, emotion_out_dim = _build_temporal_processor(
            emotion_feature_dim, temporal_hidden_size, emotion_hidden_dim,
            use_temporal_modeling, dropout,
        )
        self.emotion_proj = nn.Linear(emotion_out_dim, fusion_dim)

        # Cross-attention nomeada (chaves de checkpoint preservadas):
        # query = token da modalidade; key/value = concat das outras duas.
        self.cross_attn_video = nn.MultiheadAttention(
            fusion_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_pose = nn.MultiheadAttention(
            fusion_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.cross_attn_emotion = nn.MultiheadAttention(
            fusion_dim, num_heads=4, dropout=dropout, batch_first=True)

        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

        # Tokens "modality ausente" (modal dropout) e head auxiliar só-vídeo.
        # Só criados quando ativos — checkpoints antigos seguem com load estrito.
        if self.emotion_dropout_prob > 0:
            self.emotion_missing_token = nn.Parameter(torch.zeros(fusion_dim))
        if self.pose_dropout_prob > 0:
            self.pose_missing_token = nn.Parameter(torch.zeros(fusion_dim))
        if self.aux_video_weight > 0:
            self.video_aux_head = nn.Linear(fusion_dim, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def _process_temporal(processor: nn.Module, features: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, D_out): último timestep da LSTM ou média temporal do MLP."""
        if features.dim() == 2:
            features = features.unsqueeze(1)
        elif features.dim() != 3:
            raise ValueError(
                f"Features devem ter shape (batch, T, D) ou (batch, D), "
                f"mas recebeu shape {features.shape}"
            )

        if isinstance(processor, nn.LSTM):
            lstm_out, _ = processor(features)
            return lstm_out[:, -1, :]

        B, T, D = features.shape
        processed = processor(features.reshape(B * T, D))
        return processed.view(B, T, -1).mean(dim=1)

    def _maybe_drop_modality(
        self,
        tokens: torch.Tensor,
        index: int,
        prob: float,
        missing_token: torch.Tensor,
    ) -> torch.Tensor:
        """Substitui o token da modalidade por um embedding learnable (treino)."""
        if prob <= 0 or not self.training or missing_token is None:
            return tokens
        mask = (torch.rand(tokens.shape[0], device=tokens.device) < prob).unsqueeze(1)
        tokens = tokens.clone()
        tokens[:, index] = torch.where(
            mask, missing_token.unsqueeze(0).expand_as(tokens[:, index]), tokens[:, index]
        )
        return tokens

    def forward(
        self,
        video_features: torch.Tensor,
        pose_features: torch.Tensor,
        emotion_features: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor:
        batch_size = video_features.shape[0]

        if video_features.dim() == 3:  # legado (B, T, D_v) -> mean-pool
            video_features = video_features.mean(dim=1)

        if pose_features.dim() == 4 and pose_features.shape[-1] == 3:
            pose_features = pose_features.view(batch_size, pose_features.shape[1], -1)
        elif pose_features.dim() == 2:
            pose_features = pose_features.unsqueeze(1)

        if emotion_features.dim() == 2:
            emotion_features = emotion_features.unsqueeze(1)

        # Padding por último timestep apenas para pose/emotion.
        T = max(pose_features.shape[1], emotion_features.shape[1])
        pose_features = _pad_to_length(pose_features, T)
        emotion_features = _pad_to_length(emotion_features, T)

        tokens = torch.stack([
            self.video_proj(video_features),
            self.pose_proj(self._process_temporal(self.pose_processor, pose_features)),
            self.emotion_proj(self._process_temporal(self.emotion_processor, emotion_features)),
        ], dim=1)  # (B, 3, fusion_dim)

        # Modal dropout (só treino): face é o atalho dominante — sem ela no
        # treino, a fusão aprende a decidir com vídeo+pose.
        tokens = self._maybe_drop_modality(
            tokens, 2, self.emotion_dropout_prob,
            getattr(self, "emotion_missing_token", None))
        tokens = self._maybe_drop_modality(
            tokens, 1, self.pose_dropout_prob,
            getattr(self, "pose_missing_token", None))

        # Head auxiliar sobre o token de vídeo pré-atenção: força a stream de
        # vídeo a ser preditiva por si (deep supervision, só no treino).
        video_token = tokens[:, 0]
        aux_logits = None
        if self.aux_video_weight > 0:
            aux_logits = self.video_aux_head(video_token)

        # Cada modalidade atende a concatenação das outras duas.
        attended = []
        for i, attn in enumerate(
            (self.cross_attn_video, self.cross_attn_pose, self.cross_attn_emotion)
        ):
            others = torch.cat([tokens[:, j:j + 1] for j in range(3) if j != i], dim=1)
            out, _ = attn(tokens[:, i:i + 1], others, others)
            attended.append(out)

        fused = self.fusion_layers(torch.cat(attended, dim=1).mean(dim=1))
        logits = self.classifier(fused)

        if return_aux:
            if aux_logits is None:
                raise ValueError(
                    "return_aux=True exige aux_video_weight > 0 no construtor."
                )
            return logits, aux_logits
        return logits


def create_multimodal_model(
    video_feature_dim: int = 512,
    pose_feature_dim: int = 51,
    emotion_feature_dim: int = 128,
    num_frames: int = 16,
    fusion_method: Literal["cross_attention"] = "cross_attention",
    use_temporal_modeling: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    **kwargs,
) -> MultimodalRiskDetector:
    """Cria o modelo multimodal e o move para o device."""
    model = MultimodalRiskDetector(
        video_feature_dim=video_feature_dim,
        pose_feature_dim=pose_feature_dim,
        emotion_feature_dim=emotion_feature_dim,
        num_frames=num_frames,
        fusion_method=fusion_method,
        use_temporal_modeling=use_temporal_modeling,
        **kwargs,
    )
    return model.to(device)
