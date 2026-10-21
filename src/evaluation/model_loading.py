"""
Carregamento de checkpoints para avaliação.

Centraliza o que vivia espalhado em run_evaluation.py / run_cross_label_
evaluation.py: sub-modelo de pose, wrapper multimodal+backbone de vídeo,
load_model por tipo e montagem de curvas ROC/PR.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import paths as p
from src.models.cnn3d_risk import create_cnn3d_model
from src.models.emotion_cnn import create_emotion_model
from src.models.multimodal_risk import create_multimodal_model


class PoseSubModel(nn.Module):
    """
    Sub-modelo de pose para avaliação standalone (F5, EVAL-T1).

    Replica o branch de pose do MultimodalRiskDetector: processador temporal
    (LSTM ou MLP) -> projeção -> camadas de fusão -> classificador binário.
    Construído a partir das dimensões inferidas do checkpoint multimodal e
    carregado com strict=False (somente as chaves pose/fusão/classificador).
    """

    def __init__(
        self,
        pose_feature_dim: int = 51,
        pose_hidden_dim: int = 64,
        fusion_dim: int = 256,
        use_temporal_modeling: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        if use_temporal_modeling:
            self.pose_processor = nn.LSTM(
                pose_feature_dim, pose_hidden_dim, num_layers=1,
                batch_first=True, dropout=0
            )
            pose_output_dim = pose_hidden_dim
        else:
            self.pose_processor = nn.Sequential(
                nn.Linear(pose_feature_dim, pose_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            pose_output_dim = pose_hidden_dim
        self.pose_proj = nn.Linear(pose_output_dim, fusion_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, T, num_joints, 3) ou (B, T, D_p)
        Returns:
            Logits (B, num_classes)
        """
        if pose.ndim == 4 and pose.shape[-1] == 3:
            pose = pose.view(pose.shape[0], pose.shape[1], -1)
        elif pose.ndim == 2:
            pose = pose.unsqueeze(1)

        if isinstance(self.pose_processor, nn.LSTM):
            lstm_out, _ = self.pose_processor(pose)
            last = lstm_out[:, -1, :]
        else:
            B, T, D = pose.shape
            h = self.pose_processor(pose.reshape(B * T, D))
            last = h.reshape(B, T, -1).mean(dim=1)

        proj = self.pose_proj(last)
        return self.classifier(self.fusion_layers(proj))


class MultimodalEvalWrapper(nn.Module):
    """Envolve multimodal + backbone de vídeo para avaliação.

    O dataloader multimodal fornece frames brutos (B, T, C, H, W); o backbone
    extrai o clip token (B, D_v) antes do forward multimodal (cross-attention).
    """

    def __init__(self, multimodal, video_model):
        super().__init__()
        self.multimodal = multimodal
        self.video_model = video_model

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if len(video.shape) == 5:
                # Frames em frame-last (B, T, C, H, W); CNN3D espera (B, C, T, H, W)
                video = video.permute(0, 2, 1, 3, 4)
                video = self.video_model.get_features(video)  # (B, D_v) clip token
        return self.multimodal(video, pose, emotion)


def _infer_pose_submodel_dims(state_dict) -> Dict[str, Any]:
    """Infere as dimensões do sub-modelo de pose a partir do state_dict."""
    dims = {"use_temporal_modeling": True, "pose_feature_dim": 51,
            "pose_hidden_dim": 64, "fusion_dim": 256, "num_classes": 2}

    if "pose_processor.weight_ih_l0" in state_dict:  # LSTM
        w = state_dict["pose_processor.weight_ih_l0"]
        dims["pose_feature_dim"] = w.shape[1]
        dims["pose_hidden_dim"] = w.shape[0] // 4
    elif "pose_processor.0.weight" in state_dict:  # MLP
        w = state_dict["pose_processor.0.weight"]
        dims["pose_feature_dim"] = w.shape[1]
        dims["pose_hidden_dim"] = w.shape[0]
        dims["use_temporal_modeling"] = False

    if "fusion_layers.0.weight" in state_dict:
        dims["fusion_dim"] = state_dict["fusion_layers.0.weight"].shape[0]
    if "classifier.weight" in state_dict:
        dims["num_classes"] = state_dict["classifier.weight"].shape[0]

    return dims


def _create_video_backbone(device: str, video_model_path: Optional[str] = None):
    """Backbone CNN3D usado como extrator de clip token do multimodal."""
    vckpt_path = video_model_path or str(p.CNN3D_WEIGHTS / "best_model.pth")
    vckpt = torch.load(vckpt_path, map_location=device)
    vmodel_name = vckpt.get("model_name") or vckpt.get("backbone") or "r2plus1d_18"
    video_model = create_cnn3d_model(
        model_name=vmodel_name,
        num_classes=2,
        checkpoint_path=vckpt_path,
        device=device,
    )
    video_model.eval()
    return video_model


def load_model(
    model_type: str,
    model_path: str,
    device: str,
    video_model_path: Optional[str] = None,
) -> torch.nn.Module:
    """Carrega modelo do checkpoint para avaliação."""
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "multimodal":
        fusion_method = checkpoint.get("fusion_method", "cross_attention")
        use_temporal = checkpoint.get("use_temporal_modeling", True)
        # Dimensão/backbone de vídeo lidos do checkpoint (AD-013: cnn3d default)
        video_backbone_ckpt = checkpoint.get("video_backbone", "cnn3d")
        if video_backbone_ckpt != "cnn3d":
            raise ValueError(
                f"Backbone de vídeo '{video_backbone_ckpt}' não é mais suportado; "
                "o projeto usa apenas 'cnn3d'."
            )
        video_feature_dim = checkpoint.get("video_feature_dim") or 512
        # Dim da pose inferida do checkpoint (nº de juntas varia por dataset).
        state = checkpoint.get("model_state_dict", checkpoint)
        pose_feature_dim = _infer_pose_submodel_dims(state)["pose_feature_dim"]
        # Flags de robustez (modal dropout/aux head) gravadas no treino;
        # ausentes em checkpoints antigos -> defaults 0 (sem chaves novas).
        emotion_dropout_prob = checkpoint.get("emotion_dropout_prob", 0.0)
        pose_dropout_prob = checkpoint.get("pose_dropout_prob", 0.0)
        aux_video_weight = checkpoint.get("aux_video_weight", 0.0)

        model = create_multimodal_model(
            video_feature_dim=video_feature_dim,
            pose_feature_dim=pose_feature_dim,
            emotion_feature_dim=128,
            num_frames=16,
            fusion_method=fusion_method,
            use_temporal_modeling=use_temporal,
            emotion_dropout_prob=emotion_dropout_prob,
            pose_dropout_prob=pose_dropout_prob,
            aux_video_weight=aux_video_weight,
            device=device,
        )
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = MultimodalEvalWrapper(model, _create_video_backbone(device, video_model_path))

    elif model_type in ("cnn3d", "video"):
        # Backbone selecionado pela metadata do checkpoint (model_name/backbone),
        # com fallback para o default r2plus1d_18 (pesos Kinetics-400).
        model_name = checkpoint.get("model_name") or checkpoint.get("backbone") or "r2plus1d_18"
        model = create_cnn3d_model(
            model_name=model_name,
            num_classes=2,
            checkpoint_path=model_path,
            device=device,
        )

    elif model_type == "pose":
        # Sub-modelo de pose: branch de pose do multimodal, avaliado standalone.
        # O checkpoint é o multimodal (ou um checkpoint de pose dedicado).
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        dims = _infer_pose_submodel_dims(state_dict)
        model = PoseSubModel(
            pose_feature_dim=dims["pose_feature_dim"],
            pose_hidden_dim=dims["pose_hidden_dim"],
            fusion_dim=dims["fusion_dim"],
            use_temporal_modeling=dims["use_temporal_modeling"],
            num_classes=dims["num_classes"],
        ).to(device)
        loaded = model.load_state_dict(state_dict, strict=False)
        if loaded.missing_keys and "pose_processor.weight_ih_l0" in loaded.missing_keys:
            print(
                "⚠️  Aviso: checkpoint sem as chaves do branch de pose "
                "(pose_processor/pose_proj/fusion_layers/classifier). "
                "O sub-modelo de pose usará pesos aleatórios."
            )

    elif model_type == "emotion":
        # Sub-modelo de emoção: EmotionNet com classificação binária
        model = create_emotion_model(
            num_emotions=2,
            pretrained=True,
            checkpoint_path=model_path,
            device=device,
        )

    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")

    model.eval()
    return model


def load_multimodal_model(
    model_path: str,
    device: str,
    video_model_path: Optional[str] = None,
) -> torch.nn.Module:
    """Carrega o modelo multimodal + backbone de vídeo (wrapper de avaliação)."""
    model = load_model("multimodal", model_path, device, video_model_path)
    model.eval()
    return model


def load_feature_stats_for_checkpoint(model_path: str) -> Optional[Dict[str, Any]]:
    """
    Stats de normalização compatíveis com o checkpoint.

    Checkpoints treinados com normalize_features=True exigem as mesmas
    stats no avaliação (senão há mismatch de escala train/eval). Checkpoints
    antigos (sem a flag) devolvem None — avaliaram features brutas sempre.
    """
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as e:  # noqa: BLE001 - diagnóstico, não bloqueia a avaliação
        print(f"⚠️  Não foi possível ler metadata de {model_path}: {e}")
        return None
    if not checkpoint.get("normalize_features", False):
        return None
    from src.datasets.feature_stats import load_feature_stats
    stats = load_feature_stats(p.MULTIMODAL_FEATURE_STATS)
    if stats is None:
        print(
            f"⚠️  Checkpoint normalizado, mas {p.MULTIMODAL_FEATURE_STATS} ausente. "
            "Avaliação será feita SEM normalização (reentrener para regenerar)."
        )
    else:
        print(f"✓ Feature stats do train aplicadas na avaliação: {sorted(stats)}")
    return stats


def resolve_sub_model_path(
    model_type: str,
    model_path: str,
    video_model_path: Optional[str] = None,
    pose_model_path: Optional[str] = None,
    emotion_model_path: Optional[str] = None,
) -> str:
    """Resolve o caminho do checkpoint para um sub-modelo (F5, EVAL-T1)."""
    explicit = {
        "video": video_model_path,
        "pose": pose_model_path,
        "emotion": emotion_model_path,
    }.get(model_type)
    if explicit:
        return explicit

    defaults = {
        # video: CNN 3D
        "video": str(p.CNN3D_WEIGHTS / "best_model.pth"),
        # pose: branch de pose do checkpoint multimodal (--model_path)
        "pose": model_path,
        # emotion: EmotionNet treinado
        "emotion": str(p.EMOTION_CNN_WEIGHTS / "best_model.pth"),
    }
    if model_type not in defaults:
        raise ValueError(f"Sub-modelo não suportado: {model_type}")
    return defaults[model_type]


def build_roc_pr(
    model_type: str,
    y_true,
    y_proba,
    num_classes: int,
    class_names: List[str],
) -> Any:
    """
    Monta a estrutura de curvas ROC/PR para os gráficos (EVAL-T2).

    Binário -> uma curva; multiclasse -> uma curva por classe (one-vs-rest).
    """
    if num_classes > 2:
        curves = []
        for c in range(num_classes):
            curve = _roc_pr_curve((y_true == c).astype(int), y_proba[:, c])
            curve["label"] = str(class_names[c])
            curves.append(curve)
        return curves

    curve = _roc_pr_curve(y_true, y_proba)
    curve["label"] = model_type
    return curve


def _roc_pr_curve(y_true, y_score) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    empty: Dict[str, Any] = {
        "fpr": [], "tpr": [], "precision": [], "recall": [],
        "auc": None, "ap": None, "label": "",
    }
    # Células single-class: ROC/AP degenerados (auc nan / ap 0 ou 1 trivial).
    if np.unique(y_true).size < 2:
        return empty

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    curve: Dict[str, Any] = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auc": None,
        "ap": None,
        "label": "",
    }
    try:
        curve["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        pass
    try:
        curve["ap"] = float(average_precision_score(y_true, y_score))
    except ValueError:
        pass
    return curve
