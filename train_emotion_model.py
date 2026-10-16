"""
Script de treinamento para modelo de Emotion Recognition no dataset AffectNet.

Arquitetura: DeiT-Small pré-treinada no ImageNet, classificação binária
violent / non_violent.

Hiperparâmetros: configs/treino.toml (seção [emotion]).

Uso:
    python train_emotion_model.py
    python train_emotion_model.py --epochs 5 --device cpu
    python train_emotion_model.py --resume
"""

import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

from src import paths as p
from src.datasets.affectnet_dataset import (
    AffectNetDataset,
    get_transforms,
    resolve_affectnet_root,
    resolve_affectnet_splits,
)
from src.evaluation.charts import plot_confusion_matrix
from src.models.emotion_cnn import create_emotion_model
from src.models.losses import FocalLoss
from src.training.config import add_config_args, resolve_config
from src.training.loop import train_loop
from src.training.utils import create_dataloader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treinar EmotionNet no AffectNet (configs/treino.toml)"
    )
    add_config_args(parser)
    parser.add_argument(
        "--resume", action="store_true",
        help="Retomar treino a partir do último best_model.pth",
    )
    return parser


def main():
    args = build_parser().parse_args()
    cfg = resolve_config("emotion", args)

    dataset_root = resolve_affectnet_root(cfg["dataset_path"])
    train_split, val_split = resolve_affectnet_splits(dataset_root)

    output_dir = p.EMOTION_CNN_WEIGHTS
    experiments_dir = p.EMOTION_CNN_EXPERIMENTS
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = AffectNetDataset(
        data_root=dataset_root,
        split=train_split,
        transform=get_transforms(is_train=True),
        min_confidence=cfg["min_confidence"],
    )
    val_dataset = AffectNetDataset(
        data_root=dataset_root,
        split=val_split,
        transform=get_transforms(is_train=False),
        min_confidence=cfg["min_confidence"],
    )

    train_sampler = None
    if cfg["balance_classes"]:
        class_counts = torch.zeros(len(train_dataset.class_to_idx))
        for _, label in train_dataset.samples:
            class_counts[label] += 1
        weights = [1.0 / class_counts[label].item() for _, label in train_dataset.samples]
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = create_dataloader(
        train_dataset, batch_size=cfg["batch_size"],
        shuffle=not cfg["balance_classes"], sampler=train_sampler,
        num_workers=cfg["num_workers"],
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )

    print("=" * 60)
    print("Treinamento do Modelo de Emotion Recognition")
    print("=" * 60)
    print(f"Dataset: {dataset_root}")
    print(f"Device: {cfg['device']} | Épocas: {cfg['epochs']} | "
          f"Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")
    print(f"Weight decay: {cfg['weight_decay']} | Warmup: {cfg['warmup_epochs']}")
    print(f"Mixed precision: {'ON' if cfg['amp'] else 'OFF'} | "
          f"Grad clip: {cfg['grad_clip']}")
    print(f"Focal Loss (γ={cfg['focal_gamma']}) | "
          f"Sampler balanceado: {cfg['balance_classes']}")
    print(f"Early stop patience: {cfg['early_stop_patience']}")
    print(f"Classifier: Linear(384→128→2)")
    print("=" * 60, "\n")

    # ── Modelo / resume ────────────────────────────────────────────────────
    resume_path = output_dir / 'best_model.pth'
    start_epoch, best_val_acc = 1, -1.0
    resume_ckpt = None

    if args.resume and resume_path.exists():
        model, resume_ckpt = create_emotion_model(
            num_emotions=2, pretrained=True,
            checkpoint_path=str(resume_path), resume_training=True,
            device=cfg["device"],
        )
        start_epoch = resume_ckpt['epoch'] + 1
        best_val_acc = resume_ckpt['val_acc']
        print(f"Retomando treino da época {resume_ckpt['epoch']} "
              f"(val_acc: {resume_ckpt['val_acc']:.2f}%)")
    else:
        model = create_emotion_model(num_emotions=2, pretrained=True, device=cfg["device"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"] and torch.cuda.is_available())
    criterion = FocalLoss(gamma=cfg["focal_gamma"])
    optimizer = optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=cfg["warmup_epochs"]),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg["epochs"] - cfg["warmup_epochs"])
            ),
        ],
        milestones=[cfg["warmup_epochs"]],
    )

    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in resume_ckpt:
            try:
                scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            except Exception:
                print("⚠️  Scheduler state incompatível. Recriando scheduler.")

    best_val_acc, _ = train_loop(
        model, train_loader, val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=torch.device(cfg["device"]), epochs=cfg["epochs"],
        output_dir=output_dir, experiments_dir=experiments_dir,
        config=cfg, scaler=scaler if cfg["amp"] else None,
        grad_clip=cfg["grad_clip"], return_metrics=True,
        early_stop_patience=cfg["early_stop_patience"],
        start_epoch=start_epoch, best_val_acc=best_val_acc,
    )

    # Gráficos da melhor época (CM vem do loop via confusion_matrix.npy)
    cm_path = experiments_dir / 'confusion_matrix.npy'
    if cm_path.exists():
        plot_confusion_matrix(
            np.load(cm_path),
            class_names=list(train_dataset.class_to_idx.keys()),
            output_path=experiments_dir / 'confusion_matrix.png',
            title="Melhor época — matriz de confusão",
        )

    checkpoint = torch.load(output_dir / 'best_model.pth', map_location=cfg["device"])
    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo recarregado do best checkpoint (época {checkpoint['epoch']})")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
