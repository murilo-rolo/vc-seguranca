"""
Script de treinamento para CNN 3D com duas etapas:
1. Pré-treinamento em UCF101 (9 classes relevantes)
2. Fine-tuning em RWF-2000 (2 classes: violent/non-violent)
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from src.models.cnn3d_risk import create_cnn3d_model
from src.datasets.video3d_dataset import (
    get_ucf101_dataloaders,
    get_rwf2000_3d_dataloaders
)
from src.training.utils import run_epoch
from src import paths as p


def _permute_clips(batch):
    """Converte batch (B,T,C,H,W) para (B,C,T,H,W) para modelos 3D."""
    clips, labels = batch
    if len(clips.shape) == 5:
        clips = clips.permute(0, 2, 1, 3, 4)
    return clips, labels


def pretrain_ucf101(args):
    """Etapa 1: Pré-treinamento em UCF101."""
    print("=" * 60)
    print("ETAPA 1: Pré-treinamento em UCF101")
    print("=" * 60)
    
    # Criar diretórios de saída (nova estrutura)
    output_dir = p.CNN3D_UCF101_WEIGHTS
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = p.CNN3D_UCF101_EXPERIMENTS
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Criar modelo (9 classes relevantes para UCF101)
    print("Criando modelo...")
    model = create_cnn3d_model(
        model_name=args.model_name,
        num_classes=9,  # UCF101 filtrado: 9 classes relevantes
        pretrained=args.pretrained,
        pretrained_dataset="kinetics400",
        dropout=args.dropout,
        freeze_backbone=False,
        device=args.device
    )
    print(f"✓ Modelo criado: {args.model_name}")
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Criar DataLoaders
    print("Criando DataLoaders...")
    train_loader, test_loader = get_ucf101_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        clip_size=args.clip_size,
        num_workers=args.num_workers
    )
    print(f"✓ DataLoaders criados")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Loss e optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Treinamento
    best_test_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print("\nIniciando treinamento...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
            desc=f"Epoch {epoch} [Train]",
            model_hook=_permute_clips
        )

        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, device,
            is_train=False, desc=f"Epoch {epoch} [Val]",
            model_hook=_permute_clips
        )
        
        # Atualizar learning rate
        scheduler.step()
        
        # Salvar histórico
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Salvar melhor modelo
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'model_name': args.model_name,
                'num_classes': 9
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"\n✓ Melhor modelo salvo! Test Acc: {test_acc:.2f}%")
        
        print()
    
    # Salvar histórico
    with open(experiments_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Pré-treinamento concluído!")
    print(f"Melhor Test Acc: {best_test_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)
    
    return output_dir / 'best_model.pth'


def finetune_rwf2000(args, pretrained_path: Path):
    """Etapa 2: Fine-tuning em RWF-2000."""
    print("=" * 60)
    print("ETAPA 2: Fine-tuning em RWF-2000")
    print("=" * 60)
    
    # Criar diretórios de saída (nova estrutura)
    output_dir = p.CNN3D_RWF2000_WEIGHTS
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = p.CNN3D_RWF2000_EXPERIMENTS
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Criar modelo (2 classes para RWF-2000)
    print("Criando modelo para fine-tuning...")
    
    # Carregar checkpoint do pré-treinamento (9 classes UCF101)
    # Depois adaptar para 2 classes (RWF-2000)
    checkpoint = torch.load(pretrained_path, map_location=device)
    
    # Criar modelo com 2 classes
    model = create_cnn3d_model(
        model_name=args.model_name,
        num_classes=2,  # Binary classification
        pretrained=False,
        pretrained_dataset="kinetics400",
        dropout=args.dropout,
        freeze_backbone=False,  # Não congelar ainda
        num_frames=args.num_frames,
        device=args.device
    )
    
    # Carregar pesos do backbone (ignorar classifier que tem 9 classes do UCF101)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Carregar apenas pesos do backbone
    backbone_state_dict = {}
    for key, value in state_dict.items():
        if 'classifier' not in key:
            backbone_state_dict[key] = value
    
    try:
        model.backbone.load_state_dict(backbone_state_dict, strict=False)
        print("✓ Pesos do backbone carregados do checkpoint UCF101")
    except Exception as e:
        print(f"⚠ Aviso ao carregar pesos: {e}")
        print("  Continuando com pesos aleatórios...")
    
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
                'num_classes': 2,
                'pretrained_path': str(pretrained_path)
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
    
    # Etapa
    parser.add_argument(
        "--stage",
        type=str,
        choices=["pretrain", "finetune", "both"],
        required=True,
        help="Etapa de treinamento: 'pretrain' (UCF101), 'finetune' (RWF-2000), ou 'both'"
    )
    
    # Modelo
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["r3d_18", "r2plus1d_18", "mc3_18"],
        default="r2plus1d_18",
        help="Nome do modelo 3D (padrão: 'r2plus1d_18')"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Usar pesos pré-treinados do Kinetics400 (apenas para pretrain)"
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default=None,
        help="Caminho para modelo pré-treinado em UCF101 (para finetune)"
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
        default=0.5,
        help="Taxa de dropout"
    )
    
    # Outros
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
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
    
    if args.stage == "pretrain" or args.stage == "both":
        # Etapa 1: Pré-treinamento
        args.dataset_root = str(p.UCF101_ROOT)
        pretrained_path = pretrain_ucf101(args)
        
        if args.stage == "pretrain":
            return
    
    if args.stage == "finetune" or args.stage == "both":
        # Etapa 2: Fine-tuning
        if args.pretrained_path:
            pretrained_path = Path(args.pretrained_path)
        elif args.stage == "both":
            # Usar modelo recém-treinado
            pass  # pretrained_path já definido acima
        else:
            raise ValueError("Para fine-tuning, forneça --pretrained_path ou use --stage both")
        
        args.dataset_root = str(p.RWF2000_ROOT)
        finetune_rwf2000(args, pretrained_path)


if __name__ == "__main__":
    main()

