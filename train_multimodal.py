"""
Script de treinamento para modelo multimodal de detecção de risco.

Hiperparâmetros: configs/treino.toml (seção [multimodal]).

Mitigação do atalho facial (cross-label):
- pares incongruentes materializados no dataset pareado
  (--cross-label-train-ratio) e val com 4 células; seleção por val_acc
- emotion_swap_prob: swap estocástico (0 por padrão, pares explícitos bastam)
- emotion_dropout_prob/pose_dropout_prob: modal dropout (tokens learnable)
- normalize_features: normaliza emotion/pose com stats do train

Uso:
    python train_multimodal.py
    python train_multimodal.py --epochs 10 --device cpu
"""

import argparse
import json
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


class FrozenVideoFeatureExtractor(nn.Module):
    """Extrai o clip token com o CNN3D congelado antes do forward multimodal."""

    def __init__(self, multimodal_model, video_model):
        super().__init__()
        self.multimodal = multimodal_model
        self.video_model = video_model
        for param in self.video_model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.video_model.eval()
        return self

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if video.dim() == 5:
                video = self.video_model.get_features(ensure_channels_first(video))
        return self.multimodal(video, pose, emotion)


def _multimodal_hook(batch):
    return (batch[0], batch[1], batch[2]), batch[3]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_paired_metadata() -> dict:
    """Lê do manifest a config do dataset pareado para rastrear no checkpoint."""
    manifest_path = p.PAIRED_ROOT / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path) as f:
        manifest = json.load(f)
    config = manifest.get("config", {})
    return {
        "cross_label_train_ratio": config.get("cross_label_train_ratio", 0.0),
        "val_cells": config.get("val_cells", []),
    }


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

    emotion_dropout_prob = cfg.get("emotion_dropout_prob", 0.0)
    pose_dropout_prob = cfg.get("pose_dropout_prob", 0.0)
    emotion_swap_prob = cfg.get("emotion_swap_prob", 0.0)
    weight_decay = cfg.get("weight_decay", 0.0)

    print("=" * 60)
    print("Treinamento do Modelo Multimodal")
    print("=" * 60)
    print(f"Fusion method: cross_attention")
    print(f"Temporal modeling: {cfg['use_temporal_modeling']}")
    print(f"Device: {cfg['device']} | Épocas: {cfg['epochs']} | "
          f"Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")
    
    print("=" * 60)

    print("Carregando modelo de vídeo...")
    video_model = create_cnn3d_model(
        num_classes=2,
        checkpoint_path=cfg["video_model_path"],
        device=cfg["device"],
    )
    video_model.eval()
    print(f"Modelo de vídeo CNN 3D carregado de: {cfg['video_model_path']}")

    feature_stats = None
    pose_feature_dim = 51
    paired_metadata = {}
    print("Criando DataLoaders...")
    from src.datasets.paired_dataset import (
        get_paired_dataloaders,
        paired_root_exists,
    )
    if paired_root_exists():
        print(f"✓ Dataset pareado: {p.PAIRED_ROOT}")
        paired_metadata = _read_paired_metadata()
        print(f"✓ Cross-label no train: {paired_metadata['cross_label_train_ratio']:.2f} "
              f"| val cells: {paired_metadata['val_cells']}")
        train_loader, val_loader, _ = get_paired_dataloaders(
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
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
        fusion_method="cross_attention",
        use_temporal_modeling=cfg["use_temporal_modeling"],
        emotion_dropout_prob=emotion_dropout_prob,
        pose_dropout_prob=pose_dropout_prob,
        device=cfg["device"],
    )

    train_model = FrozenVideoFeatureExtractor(multimodal_model, video_model)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.0)
    )

    optimizer = optim.Adam(
        multimodal_model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=weight_decay,
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
            "paired_dataset": paired_metadata,
            "emotion_feature_dim": multimodal_model.emotion_feature_dim,
            "video_feature_dim": multimodal_model.video_feature_dim,
            "video_backbone": "cnn3d",
            "emotion_swap_prob": emotion_swap_prob,
            "emotion_dropout_prob": emotion_dropout_prob,
            "pose_dropout_prob": pose_dropout_prob,
            "video_model_state_dict": video_model.state_dict(),
        },
    )

    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
