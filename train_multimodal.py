"""
Script de treinamento para modelo multimodal de detecção de risco.

Hiperparâmetros: configs/treino.toml (seção [multimodal]).

Uso:
    python train_multimodal.py
    python train_multimodal.py --epochs 10 --device cpu
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src import paths as p
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

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if video.dim() == 5:
                video = self.video_model.get_features(ensure_channels_first(video))
        return self.multimodal(video, pose, emotion)


def _multimodal_hook(batch):
    return (batch[0], batch[1], batch[2]), batch[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treinar modelo multimodal de detecção de risco (configs/treino.toml)"
    )
    add_config_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = resolve_config("multimodal", args)

    output_dir = p.MULTIMODAL_WEIGHTS
    experiments_dir = p.MULTIMODAL_EXPERIMENTS

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
        num_frames=cfg["num_frames"],
        device=cfg["device"],
    )
    video_model.eval()
    print(f"Modelo de vídeo CNN 3D carregado de: {cfg['video_model_path']}")

    print("Criando modelo multimodal...")
    multimodal_model = create_multimodal_model(
        video_feature_dim=video_model.feature_dim,
        pose_feature_dim=51,
        emotion_feature_dim=128,
        num_frames=cfg["window_size"],
        fusion_method="cross_attention",
        use_temporal_modeling=cfg["use_temporal_modeling"],
        device=cfg["device"],
    )
    model = MultimodalWithVideoBackbone(multimodal_model, video_model).to(
        torch.device(cfg["device"])
    )

    print("Criando DataLoaders...")
    from src.datasets.paired_dataset import get_paired_dataloaders, paired_root_exists
    if paired_root_exists():
        print(f"✓ Dataset pareado: {p.PAIRED_ROOT}")
        train_loader, val_loader, _ = get_paired_dataloaders(
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
        )
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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(multimodal_model.parameters(), lr=cfg["learning_rate"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_val_acc, _ = train_loop(
        model, train_loader, val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=torch.device(cfg["device"]), epochs=cfg["epochs"],
        output_dir=output_dir, experiments_dir=experiments_dir,
        model_hook=_multimodal_hook, config=cfg,
        checkpoint_model=multimodal_model,
        checkpoint_extra={
            "fusion_method": "cross_attention",
            "use_temporal_modeling": cfg["use_temporal_modeling"],
            "emotion_feature_dim": multimodal_model.emotion_feature_dim,
            "video_feature_dim": multimodal_model.video_feature_dim,
            "video_backbone": "cnn3d",
        },
    )

    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
