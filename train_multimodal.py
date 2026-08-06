"""
Script de treinamento para modelo multimodal de detecção de risco.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json

from src.models.multimodal_risk import create_multimodal_model
from src.models.resnet_lstm import create_model as create_video_model
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.training.utils import run_epoch
from src import paths as p


class _MultimodalWrapper(nn.Module):
    """Wrapper que encapsula extração de features de vídeo + forward multimodal."""
    def __init__(self, multimodal_model, video_model):
        super().__init__()
        self.multimodal = multimodal_model
        self.video_model = video_model

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if len(video.shape) == 5:
                video_features = self.video_model.get_features(video)
                video_features = video_features.unsqueeze(1).repeat(1, video.shape[1], 1)
            else:
                video_features = video
        return self.multimodal(video_features, pose, emotion)


def _multimodal_hook(batch):
    return (batch[0], batch[1], batch[2]), batch[3]


def main():
    parser = argparse.ArgumentParser(description="Treinar modelo multimodal de detecção de risco")
    
    # Modelo
    parser.add_argument(
        "--fusion_method",
        type=str,
        choices=["early", "late", "attention"],
        default="late",
        help="Método de fusão multimodal"
    )
    parser.add_argument(
        "--use_temporal_modeling",
        action="store_true",
        help="Usar LSTM para modelagem temporal por modalidade"
    )
    parser.add_argument(
        "--video_model_path",
        type=str,
        default=None,
        help="Caminho para modelo de vídeo pré-treinado (opcional)"
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
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Número de frames por vídeo"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Tamanho da janela temporal"
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
    
    # Criar diretórios de saída (nova estrutura)
    output_dir = p.MULTIMODAL_WEIGHTS
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir = p.MULTIMODAL_EXPERIMENTS
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    
    # Carregar modelo de vídeo (para extrair features)
    print("Carregando modelo de vídeo...")
    video_model = create_video_model(
        num_frames=args.num_frames,
        hidden_size=256,
        num_layers=2,
        dropout=0.5,
        num_classes=2,
        pretrained=True,
        device=args.device
    )
    
    if args.video_model_path:
        try:
            checkpoint = torch.load(args.video_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                video_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                video_model.load_state_dict(checkpoint)
            print(f"Modelo de vídeo carregado de: {args.video_model_path}")
        except Exception as e:
            print(f"Erro ao carregar modelo de vídeo: {e}")
            print("Usando modelo com pesos ImageNet")
    
    # Criar modelo multimodal com wrapper
    print("Criando modelo multimodal...")
    multimodal_model = create_multimodal_model(
        video_feature_dim=256, pose_feature_dim=99, emotion_feature_dim=8,
        num_frames=args.window_size, fusion_method=args.fusion_method,
        use_temporal_modeling=args.use_temporal_modeling, device=args.device
    )
    model = _MultimodalWrapper(multimodal_model, video_model).to(device)

    print("Criando DataLoaders...")
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(
        video_data_root=str(p.PROCESSED_ROOT), pose_data_root=str(p.POSE_ROOT),
        emotion_data_root=str(p.EMOTION_ROOT), batch_size=args.batch_size,
        num_frames=args.num_frames, window_size=args.window_size,
        video_mode="frames", pose_mode="flatten",
        num_workers=args.num_workers, dataset_name="rwf2000"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(multimodal_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print("=" * 60)
    print("Treinamento do Modelo Multimodal")
    print("=" * 60)
    print(f"Fusion method: {args.fusion_method}")
    print(f"Temporal modeling: {args.use_temporal_modeling}")
    print(f"Device: {args.device}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print("=" * 60)
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
            desc=f"Epoch {epoch} [Train]",
            model_hook=_multimodal_hook
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            is_train=False, desc=f"Epoch {epoch} [Val]",
            model_hook=_multimodal_hook
        )

        scheduler.step()
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch, 'model_state_dict': multimodal_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc, 'val_loss': val_loss,
                'fusion_method': args.fusion_method,
                'use_temporal_modeling': args.use_temporal_modeling
            }, output_dir / 'best_model.pth')
            print(f"\n✓ Melhor modelo salvo! Val Acc: {val_acc:.2f}%")
        print()

    with open(experiments_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("=" * 60)
    print("Treinamento concluído!")
    print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {p.MULTIMODAL_WEIGHTS / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()

