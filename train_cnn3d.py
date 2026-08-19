"""
Script de fine-tuning de CNN 3D pré-treinada no Kinetics400 para
RWF-2000 (2 classes: violent/non-violent).
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from src.models.cnn3d_risk import create_cnn3d_model
from src.datasets.video3d_dataset import get_rwf2000_3d_dataloaders
from src.training.utils import run_epoch
from src import paths as p


def _permute_clips(batch):
    """Converte batch (B,T,C,H,W) para (B,C,T,H,W) para modelos 3D."""
    clips, labels = batch
    if len(clips.shape) == 5:
        clips = clips.permute(0, 2, 1, 3, 4)
    return clips, labels

def finetune_rwf2000(args):
    """Fine-tuning em RWF-2000."""
    print("=" * 60)
    print("Fine-tuning em RWF-2000")
    print("=" * 60)
    
    # Criar diretórios de saída (nova estrutura)
    output_dir = p.CNN3D_WEIGHTS
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = p.CNN3D_EXPERIMENTS
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Criar modelo (2 classes para RWF-2000)
    print("Criando modelo para fine-tuning...")
    
    # Criar modelo com 2 classes, backbone pré-treinado no Kinetics400
    model = create_cnn3d_model(
        model_name=args.model_name,
        num_classes=2,  # Binary classification
        pretrained=True,
        pretrained_dataset="kinetics400",
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
        num_frames=args.num_frames,
        device=args.device
    )

    print(f"✓ Backbone pré-treinado no Kinetics400 carregado")
    
    # Se freeze_backbone, só treinar classifier
    if args.freeze_backbone:
        print("  Backbone congelado, apenas classifier será treinado")
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
    
    print(f"✓ Modelo criado: {args.model_name}")
    print(f"  Parâmetros treináveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Criar DataLoaders
    print("Criando DataLoaders...")
    train_loader, val_loader = get_rwf2000_3d_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        clip_size=args.clip_size,
        num_workers=args.num_workers
    )
    print(f"✓ DataLoaders criados")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Loss e optimizer
    # Para fine-tuning, usar learning rate menor
    finetune_lr = args.learning_rate * 0.1 if args.freeze_backbone else args.learning_rate * 0.5
    
    criterion = nn.CrossEntropyLoss()
    
    # Se freeze_backbone, só otimizar classifier
    if args.freeze_backbone:
        optimizer = optim.Adam(model.classifier.parameters(), lr=finetune_lr)
    else:
        # Diferentes learning rates para backbone e classifier
        optimizer = optim.Adam([
            {'params': model.backbone.parameters(), 'lr': finetune_lr},
            {'params': model.classifier.parameters(), 'lr': args.learning_rate}
        ])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Treinamento
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nLearning rate: {finetune_lr} (backbone), {args.learning_rate} (classifier)")
    print("Iniciando fine-tuning...")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
            desc=f"Epoch {epoch} [Train]",
            model_hook=_permute_clips
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            is_train=False, desc=f"Epoch {epoch} [Val]",
            model_hook=_permute_clips
        )
        
        # Atualizar learning rate
        scheduler.step()
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Salvar melhor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': args.model_name,
                'num_classes': 2
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"\n✓ Melhor modelo salvo! Val Acc: {val_acc:.2f}%")
        
        print()
    
    # Salvar histórico
    with open(experiments_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Fine-tuning concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Treinar CNN 3D para detecção de violência")
    
    # Modelo
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["r3d_18", "r2plus1d_18", "mc3_18"],
        default="r2plus1d_18",
        help="Nome do modelo 3D (padrão: 'r2plus1d_18')"
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Congelar backbone durante fine-tuning (apenas treinar classifier)"
    )
    
    # Treinamento
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Número de frames por clipe"
    )
    parser.add_argument(
        "--clip_size",
        type=int,
        nargs=2,
        default=[112, 112],
        help="Tamanho do clipe (H, W) - padrão: 112 112"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Taxa de dropout"
    )
    
    # Outros
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Número de workers para DataLoader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para treinamento"
    )
    
    args = parser.parse_args()
    
    # Converter clip_size para tupla
    args.clip_size = tuple(args.clip_size)
    
    # Fine-tuning com modelo pré-treinado no Kinetics400
    args.dataset_root = str(p.RWF2000_ROOT)
    finetune_rwf2000(args)


if __name__ == "__main__":
    main()

