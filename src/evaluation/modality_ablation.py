"""
Ablação de modalidades no split de teste do dataset pareado (Fase C).

Dois diagnósticos complementares, ambos reutilizando evaluate_cells:

1. Sub-modelos standalone (una modalidade, pesos do checkpoint multimodal):
   medem o que a TREINAMENTO da fusão aprendeu de cada stream isoladamente.
   - video_fusion:  clip token -> video_proj -> fusion_layers -> classifier
   - emotion_fusion: emotion_processor -> emotion_proj -> fusion -> classifier
   - pose_fusion:    PoseSubModel existente (pose_processor -> pose_proj -> ...)
   Mais o CNN 3D standalone (capacidade pura de vídeo, pesos próprios).

2. Máscara de modalidade na entrada do modelo multimodal completo:
   mede a dependência da fusão treinada por stream (zerar emotion/pose/video).

O dataset pareado fornece frames/pose/emotion; todos os modelos aqui expõem a
assinatura (video, pose, emotion) para reaproveitar MetricsCalculator.
"""

from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn

from src import paths as p
from src.models.cnn3d_risk import create_cnn3d_model, ensure_channels_first
from src.models.multimodal_risk import create_multimodal_model

from .model_loading import MultimodalEvalWrapper, PoseSubModel, _infer_pose_submodel_dims


class UnimodalWrapper(nn.Module):
    """Adapta um modelo de entrada única à assinatura (video, pose, emotion)."""

    def __init__(self, model: nn.Module, modality: str):
        super().__init__()
        if modality not in ("video", "pose", "emotion"):
            raise ValueError(f"Modalidade inválida: {modality}")
        self.model = model
        self.modality = modality

    def forward(self, video, pose, emotion):
        x = {"video": video, "pose": pose, "emotion": emotion}[self.modality]
        if self.modality == "video":
            x = ensure_channels_first(x)
        return self.model(x)


class ModalityMaskWrapper(nn.Module):
    """
    Modelo multimodal completo com streams zeradas na entrada.

    Diferente de zerar frames brutos, zera o clip token / features já no nível
    do multimodal (a saída do backbone tem viés próprio; zerar frames não zera
    o token).
    """

    def __init__(self, multimodal, video_model, mask: Set[str]):
        super().__init__()
        unknown = mask - {"video", "pose", "emotion"}
        if unknown:
            raise ValueError(f"Streams desconhecidas na máscara: {sorted(unknown)}")
        self.multimodal = multimodal
        self.video_model = video_model
        self.mask = set(mask)

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if len(video.shape) == 5:
                video = video.permute(0, 2, 1, 3, 4)
                video = self.video_model.get_features(video)
        if "video" in self.mask:
            video = torch.zeros_like(video)
        if "pose" in self.mask:
            pose = torch.zeros_like(pose)
        if "emotion" in self.mask:
            emotion = torch.zeros_like(emotion)
        return self.multimodal(video, pose, emotion)


class VideoFusionSubModel(nn.Module):
    """Branch de vídeo do multimodal: clip token -> video_proj -> fusão -> classifier."""

    def __init__(self, video_model, fusion_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.video_model = video_model
        self.video_proj = nn.Linear(video_model.feature_dim, fusion_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, video, pose=None, emotion=None):
        with torch.no_grad():
            if len(video.shape) == 5:
                video = ensure_channels_first(video)
                video = self.video_model.get_features(video)
        return self.classifier(self.fusion_layers(self.video_proj(video)))


class EmotionFusionSubModel(nn.Module):
    """Branch de emoção do multimodal: processor -> emotion_proj -> fusão -> classifier."""

    def __init__(
        self,
        emotion_feature_dim: int = 128,
        emotion_hidden_dim: int = 32,
        fusion_dim: int = 256,
        use_temporal_modeling: bool = False,
        temporal_hidden_size: int = 64,
        num_classes: int = 2,
    ):
        super().__init__()
        if use_temporal_modeling:
            self.emotion_processor = nn.LSTM(
                emotion_feature_dim, temporal_hidden_size, num_layers=1,
                batch_first=True, dropout=0,
            )
            out_dim = temporal_hidden_size
        else:
            self.emotion_processor = nn.Sequential(
                nn.Linear(emotion_feature_dim, emotion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            out_dim = emotion_hidden_dim
        self.emotion_proj = nn.Linear(out_dim, fusion_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, emotion, video=None, pose=None):
        if emotion.dim() == 2:
            emotion = emotion.unsqueeze(1)
        if isinstance(self.emotion_processor, nn.LSTM):
            out, _ = self.emotion_processor(emotion)
            last = out[:, -1, :]
        else:
            B, T, D = emotion.shape
            h = self.emotion_processor(emotion.reshape(B * T, D))
            last = h.reshape(B, T, -1).mean(dim=1)
        return self.classifier(self.fusion_layers(self.emotion_proj(last)))


def _infer_fusion_branch_dims(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """Infere fusion_dim/num_classes e o tipo de processor a partir do checkpoint."""
    dims = _infer_pose_submodel_dims(state_dict)
    if "video_proj.weight" in state_dict:
        dims["video_feature_dim"] = state_dict["video_proj.weight"].shape[1]
    if "emotion_processor.0.weight" in state_dict:
        dims["emotion_feature_dim"] = state_dict["emotion_processor.0.weight"].shape[1]
        dims["emotion_hidden_dim"] = state_dict["emotion_processor.0.weight"].shape[0]
        dims["use_temporal_modeling"] = False
    elif "emotion_processor.weight_ih_l0" in state_dict:
        w = state_dict["emotion_processor.weight_ih_l0"]
        dims["emotion_feature_dim"] = w.shape[1]
        dims["emotion_hidden_dim"] = w.shape[0] // 4
        dims["use_temporal_modeling"] = True
    return dims


def _load_fusion_branch(
    submodel: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    prefixes: List[str],
    label: str,
) -> None:
    """Carrega somente as chaves dos prefixes informados (strict=False)."""
    filtered = {
        k: v for k, v in state_dict.items()
        if any(k.startswith(pref) for pref in prefixes)
    }
    missing, unexpected = submodel.load_state_dict(filtered, strict=False)
    real_missing = [
        k for k in missing
        if not k.startswith(("video_model.",))
    ]
    if unexpected:
        print(f"⚠️  [{label}] chaves inesperadas ignoradas: {sorted(unexpected)[:5]}")
    critical = [
        k for k in real_missing
        if k.split(".")[0] in ("video_proj", "emotion_proj", "pose_proj",
                               "fusion_layers", "classifier")
    ]
    if critical:
        print(f"⚠️  [{label}] chaves ausentes no checkpoint: {critical[:5]}")


def build_ablation_models(
    multimodal_model_path: str,
    device: str,
    video_model_path: Optional[str] = None,
) -> Dict[str, nn.Module]:
    """
    Monta todos os modelos do diagnóstico de ablação.

    Returns:
        dict nome -> modelo com assinatura (video, pose, emotion) em eval mode.
    """
    checkpoint = torch.load(multimodal_model_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    dims = _infer_fusion_branch_dims(state_dict)

    vckpt_path = video_model_path or str(p.CNN3D_WEIGHTS / "best_model.pth")
    vckpt = torch.load(vckpt_path, map_location=device)
    vname = vckpt.get("model_name") or vckpt.get("backbone") or "r2plus1d_18"

    def _make_video_backbone():
        model = create_cnn3d_model(
            model_name=vname, num_classes=2, checkpoint_path=vckpt_path, device=device,
        )
        model.eval()
        return model

    video_fusion = VideoFusionSubModel(
        video_model=_make_video_backbone(),
        fusion_dim=dims["fusion_dim"],
        num_classes=dims["num_classes"],
    ).to(device)
    _load_fusion_branch(video_fusion, state_dict, ["video_proj", "fusion_layers", "classifier"],
                        "video_fusion")

    emotion_fusion = EmotionFusionSubModel(
        emotion_feature_dim=dims.get("emotion_feature_dim", 128),
        emotion_hidden_dim=dims.get("emotion_hidden_dim", 32),
        fusion_dim=dims["fusion_dim"],
        use_temporal_modeling=dims.get("use_temporal_modeling", False),
        num_classes=dims["num_classes"],
    ).to(device)
    _load_fusion_branch(
        emotion_fusion, state_dict,
        ["emotion_processor", "emotion_proj", "fusion_layers", "classifier"],
        "emotion_fusion",
    )

    pose_fusion = PoseSubModel(
        pose_feature_dim=dims["pose_feature_dim"],
        pose_hidden_dim=dims["pose_hidden_dim"],
        fusion_dim=dims["fusion_dim"],
        use_temporal_modeling=dims["use_temporal_modeling"],
        num_classes=dims["num_classes"],
    ).to(device)
    _load_fusion_branch(
        pose_fusion, state_dict,
        ["pose_processor", "pose_proj", "fusion_layers", "classifier"],
        "pose_fusion",
    )

    cnn3d = UnimodalWrapper(_make_video_backbone(), "video").to(device)

    multimodal = create_multimodal_model(
        video_feature_dim=dims.get("video_feature_dim", 512),
        pose_feature_dim=dims["pose_feature_dim"],
        emotion_feature_dim=128,
        num_frames=16,
        fusion_method="cross_attention",
        use_temporal_modeling=dims["use_temporal_modeling"],
        device=device,
    )
    multimodal.load_state_dict(state_dict)
    multimodal.eval()

    models: Dict[str, nn.Module] = {
        "video_cnn3d (standalone)": cnn3d,
        "video_fusion (branch)": UnimodalWrapper(video_fusion, "video"),
        "emotion_fusion (branch)": UnimodalWrapper(emotion_fusion, "emotion"),
        "pose_fusion (branch)": UnimodalWrapper(pose_fusion, "pose"),
    }
    for name, stream in (
        ("multimodal_mask_emotion", "emotion"),
        ("multimodal_mask_pose", "pose"),
        ("multimodal_mask_video", "video"),
    ):
        wrapper = ModalityMaskWrapper(multimodal, _make_video_backbone(), {stream}).to(device)
        models[name] = wrapper

    for model in models.values():
        model.eval()
    return models


__all__ = [
    "UnimodalWrapper",
    "ModalityMaskWrapper",
    "VideoFusionSubModel",
    "EmotionFusionSubModel",
    "build_ablation_models",
]
