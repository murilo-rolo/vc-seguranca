"""
Script de treinamento para modelo multimodal de detecção de risco.

Hiperparâmetros: configs/treino.toml (seção [multimodal]).

Mitigação do atalho facial (cross-label):
- emotion_swap_prob: troca a face pela de label oposto (só train)
- emotion_dropout_prob/pose_dropout_prob: modal dropout (tokens learnable)
- aux_video_weight: loss auxiliar sobre o token de vídeo pré-atenção
- normalize_features: normaliza emotion/pose com stats do train

Uso:
    python train_multimodal.py
    python train_multimodal.py --epochs 10 --device cpu
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src import paths as p
from src.datasets.feature_stats import (
    compute_feature_stats,
    load_feature_stats,
    save_feature_stats,
)
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.models.cnn3d_risk import create_cnn3d_model, ensure_channels_first
from src.models.multimodal_risk import create_multimodal_model
from src.training.config import add_config_args, resolve_config
from src.training.loop import train_loop


class MultimodalWithVideoBackbone(nn.Module):
    """Encapsula extração de features de vídeo (CNN 3D) + forward multimodal."""

    def __init__(self, multimodal_model, video_model):
        super().__init__()
        self.multimodal = multimodal_model
        self.video_model = video_model

    def forward(self, video, pose, emotion, return_aux: bool = False):
        with torch.no_grad():
            if video.dim() == 5:
                video = self.video_model.get_features(ensure_channels_first(video))
        return self.multimodal(video, pose, emotion, return_aux=return_aux)


class AuxTrainWrapper(nn.Module):
    """
    Wrapper de treino com loss auxiliar só-vídeo.

    Devolve apenas os logits fused (compatível com run_epoch); guarda
    aux_logits para o AuxCrossEntropyLoss somar λ·CE(video_aux).
    Durante validação (eval), aux_logits fica None -> loss = CE puro.
    """

    def __init__(self, model: MultimodalWithVideoBackbone, aux_weight: float):
        super().__init__()
        self.model = model
        self.aux_weight = aux_weight
        self.aux_logits = None

    def forward(self, video, pose, emotion):
        self.aux_logits = None
        if self.training and self.aux_weight > 0:
            logits, aux = self.model(video, pose, emotion, return_aux=True)
            self.aux_logits = aux
            return logits
        return self.model(video, pose, emotion)


class AuxCrossEntropyLoss(nn.Module):
    """CE(fused) + λ·CE(video_aux). Aux ausente (validação) -> CE puro."""

    def __init__(self, aux_weight: float, aux_source: AuxTrainWrapper,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.aux_source = aux_source
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, labels):
        loss = self.ce(outputs, labels)
        aux = self.aux_source.aux_logits
        if aux is not None:
            loss = loss + self.aux_weight * self.ce(aux, labels)
        return loss


def _multimodal_hook(batch):
    return (batch[0], batch[1], batch[2]), batch[3]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_feature_stats(cfg) -> dict | None:
    """Carrega (ou calcula e salva) stats do train para normalização."""
    if not cfg.get("normalize_features", False):
        return None
    stats_path = p.MULTIMODAL_FEATURE_STATS
    stats = load_feature_stats(stats_path)
    if stats is None:
        print(f"Calculando feature stats do train -> {stats_path}")
        from src.datasets.paired_dataset import PairedSurveillanceDataset
        probe = PairedSurveillanceDataset(
            split="train",
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
        )
        stats = compute_feature_stats(probe)
        save_feature_stats(stats, stats_path)
    print(f"✓ Feature stats: {sorted(stats.keys())}")
    return stats


def _infer_pose_feature_dim(dataset) -> int:
    """Dimensão flatten da pose (joints*3); fallback = default do dataset (zeros)."""
    import numpy as np
    from src.datasets.paired_dataset import DEFAULT_POSE_JOINTS
    for sample in dataset.samples:
        path = sample["pair_dir"] / "pose.npy"
        if path.exists():
            return int(np.load(path, mmap_mode="r").shape[1]) * 3
    # Sem pose.npy o dataset usa zeros (DEFAULT_POSE_JOINTS, 3).
    return DEFAULT_POSE_JOINTS * 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treinar modelo multimodal de detecção de risco (configs/treino.toml)"
    )
    add_config_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = resolve_config("multimodal", args)

    _set_seed(cfg.get("seed", 42))

    output_dir = p.MULTIMODAL_WEIGHTS
    experiments_dir = p.MULTIMODAL_EXPERIMENTS

    aux_video_weight = cfg.get("aux_video_weight", 0.0)
    emotion_dropout_prob = cfg.get("emotion_dropout_prob", 0.0)
    pose_dropout_prob = cfg.get("pose_dropout_prob", 0.0)
    emotion_swap_prob = cfg.get("emotion_swap_prob", 0.0)

    print("=" * 60)
    print("Treinamento do Modelo Multimodal")
    print("=" * 60)
    print(f"Fusion method: cross_attention")
    print(f"Temporal modeling: {cfg['use_temporal_modeling']}")
    print(f"Device: {cfg['device']} | Épocas: {cfg['epochs']} | "
          f"Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")
    print(f"Mitigação de atalho: swap={emotion_swap_prob} "
          f"drop(emotion)={emotion_dropout_prob} drop(pose)={pose_dropout_prob} "
          f"aux_video={aux_video_weight} "
          f"norm={cfg.get('normalize_features', False)}")
    print("=" * 60)

    print("Carregando modelo de vídeo...")
    video_model = create_cnn3d_model(
        num_classes=2,
        checkpoint_path=cfg["video_model_path"],
        num_frames=cfg["num_frames"],
        device=cfg["device"],
    )
    video_model.eval()
    print(f"Modelo de vídeo CNN 3D carregado de: {cfg['video_model_path']}")

    feature_stats = None
    pose_feature_dim = 51
    print("Criando DataLoaders...")
    from src.datasets.paired_dataset import get_paired_dataloaders, paired_root_exists
    if paired_root_exists():
        print(f"✓ Dataset pareado: {p.PAIRED_ROOT}")
        feature_stats = _resolve_feature_stats(cfg)
        train_loader, val_loader, _ = get_paired_dataloaders(
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
            feature_stats=feature_stats,
            emotion_swap_prob=emotion_swap_prob,
        )
        pose_feature_dim = _infer_pose_feature_dim(train_loader.dataset)
        print(f"✓ Pose feature dim inferida: {pose_feature_dim}")
    else:
        print("⚠️  dataset/paired não encontrado — usando CSV legado "
              "(gerar com: python -m src.preprocessing.build_paired_dataset)")
        train_loader, val_loader, _ = get_multimodal_dataloaders(
            video_data_root=str(p.PROCESSED_ROOT),
            pose_data_root=str(p.POSE_ROOT),
            emotion_data_root=str(p.EMOTION_ROOT),
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
            dataset_name="rwf2000",
            index_csv=str(p.PIPELINE_CSV_PATH) if p.PIPELINE_CSV_PATH.exists() else None,
        )

    print("Criando modelo multimodal...")
    multimodal_model = create_multimodal_model(
        video_feature_dim=video_model.feature_dim,
        pose_feature_dim=pose_feature_dim,
        emotion_feature_dim=128,
        num_frames=cfg["window_size"],
        fusion_method="cross_attention",
        use_temporal_modeling=cfg["use_temporal_modeling"],
        emotion_dropout_prob=emotion_dropout_prob,
        pose_dropout_prob=pose_dropout_prob,
        aux_video_weight=aux_video_weight,
        device=cfg["device"],
    )
    model = MultimodalWithVideoBackbone(multimodal_model, video_model).to(
        torch.device(cfg["device"])
    )

    if aux_video_weight > 0:
        train_model = AuxTrainWrapper(model, aux_video_weight).to(torch.device(cfg["device"]))
        criterion = AuxCrossEntropyLoss(
            aux_video_weight, train_model,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        train_model = model
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.0)
        )

    optimizer = optim.Adam(
        multimodal_model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_val_acc, _ = train_loop(
        train_model, train_loader, val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=torch.device(cfg["device"]), epochs=cfg["epochs"],
        output_dir=output_dir, experiments_dir=experiments_dir,
        model_hook=_multimodal_hook, config=cfg,
        early_stop_patience=cfg.get("early_stop_patience"),
        checkpoint_model=multimodal_model,
        checkpoint_extra={
            "fusion_method": "cross_attention",
            "use_temporal_modeling": cfg["use_temporal_modeling"],
            "emotion_feature_dim": multimodal_model.emotion_feature_dim,
            "video_feature_dim": multimodal_model.video_feature_dim,
            "video_backbone": "cnn3d",
            "emotion_dropout_prob": emotion_dropout_prob,
            "pose_dropout_prob": pose_dropout_prob,
            "aux_video_weight": aux_video_weight,
            "normalize_features": cfg.get("normalize_features", False),
        },
    )

    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
