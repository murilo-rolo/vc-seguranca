"""
Script de fine-tuning de CNN 3D pré-treinada no Kinetics400 para
RWF-2000 (2 classes: violent/non-violent).

Hiperparâmetros: configs/treino.toml (seção [cnn3d]).

Uso:
    python train_cnn3d.py
    python train_cnn3d.py --epochs 10 --device cpu
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from src import paths as p
from src.datasets.video3d_dataset import get_rwf2000_3d_dataloaders
from src.models.cnn3d_risk import create_cnn3d_model, ensure_channels_first
from src.training.config import add_config_args, resolve_config
from src.training.loop import train_loop


def _clip_hook(batch):
    """Clipes frame-last (B,T,C,H,W) -> (B,C,T,H,W) + labels."""
    return ensure_channels_first(batch[0]), batch[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tuning da CNN 3D no RWF-2000 (configs/treino.toml)"
    )
    add_config_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = resolve_config("cnn3d", args)
    clip_size = tuple(cfg["clip_size"])

    print("=" * 60)
    print("Fine-tuning em RWF-2000")
    print("=" * 60)
    print(f"Modelo: {cfg['model_name']} | Device: {cfg['device']}")
    print(f"Épocas: {cfg['epochs']} | Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")
    print(f"Freeze backbone: {cfg['freeze_backbone']} | Frames: {cfg['num_frames']}")
    print("=" * 60)

    output_dir = p.CNN3D_WEIGHTS
    experiments_dir = p.CNN3D_EXPERIMENTS

    model = create_cnn3d_model(
        model_name=cfg["model_name"],
        num_classes=2,
        pretrained=True,
        dropout=cfg["dropout"],
        freeze_backbone=cfg["freeze_backbone"],
        num_frames=cfg["num_frames"],
        device=cfg["device"],
    )
    print(f"✓ Modelo criado: {cfg['model_name']}")
    if cfg["freeze_backbone"]:
        print("  Backbone congelado, apenas classifier será treinado")
    n_trainable = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    print(f"  Parâmetros treináveis: {n_trainable:,}")

    train_loader, val_loader, _ = get_rwf2000_3d_dataloaders(
        dataset_root=str(p.RWF2000_ROOT),
        batch_size=cfg["batch_size"],
        num_frames=cfg["num_frames"],
        clip_size=clip_size,
        num_workers=cfg["num_workers"],
    )
    print(f"✓ DataLoaders: {len(train_loader)} train / {len(val_loader)} val batches")

    # Fine-tuning: learning rate menor no backbone
    lr = cfg["learning_rate"] * (0.1 if cfg["freeze_backbone"] else 0.5)
    criterion = nn.CrossEntropyLoss()
    if cfg["freeze_backbone"]:
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': lr},
            {'params': model.classifier.parameters(), 'lr': cfg["learning_rate"]},
        ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    print(f"\nLearning rate: {lr} (backbone), {cfg['learning_rate']} (classifier)")
    print("Iniciando fine-tuning...")

    best_val_acc, _ = train_loop(
        model, train_loader, val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=torch.device(cfg["device"]), epochs=cfg["epochs"],
        output_dir=output_dir, experiments_dir=experiments_dir,
        model_hook=_clip_hook, config=cfg,
        checkpoint_extra={"model_name": cfg["model_name"], "num_classes": 2},
    )

    print("=" * 60)
    print("Fine-tuning concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
