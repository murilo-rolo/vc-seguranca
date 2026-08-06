"""
Script principal para treinamento do modelo de detecção de violência em vídeos.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models import create_model
from src.datasets import get_dataloaders
from src.training.utils import run_epoch, setup_device, set_seed
from src import paths as p


def train(
    processed_data_root: str = str(p.PROCESSED_ROOT),
    batch_size: int = 8,
    num_frames: int = 16,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    hidden_size: int = 256,
    num_layers: int = 2,
    dropout: float = 0.5,
    num_workers: int = 4,
    device: str = None,
    save_dir: str = str(p.RESNET_LSTM_WEIGHTS),
    seed: int = 42,
    early_stopping_patience: int = 10,
    use_scheduler: bool = True
):
    device = setup_device(device)
    set_seed(seed)
    print(f"Usando device: {device}")

    print("Carregando DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(
        processed_data_root=processed_data_root,
        batch_size=batch_size, num_frames=num_frames,
        num_workers=num_workers, seed=seed
    )

    print("Criando modelo...")
    model = create_model(
        num_frames=num_frames, hidden_size=hidden_size,
        num_layers=num_layers, dropout=dropout,
        num_classes=2, pretrained=True, device=device
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = None
    if use_scheduler:
        try:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=True
            )
        except TypeError:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5
            )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    print("\n" + "="*50)
    print("Iniciando treinamento...")
    print("="*50 + "\n")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer,
            desc=f"Epoch {epoch} [Train]"
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
            is_train=False, desc=f"Epoch {epoch} [Val]"
        )

        if scheduler is not None:
            scheduler.step(val_acc)

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        if scheduler is not None:
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc, 'val_loss': val_loss,
                'train_acc': train_acc, 'train_loss': train_loss,
                'num_frames': num_frames, 'hidden_size': hidden_size,
                'num_layers': num_layers, 'dropout': dropout
            }, save_path / "best_model.pth")

            print(f"  ✓ Melhor modelo salvo! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            print(f"\n⚠️  Early stopping ativado após {patience_counter} épocas sem melhoria.")
            print(f"Melhor validação: {best_val_acc:.2f}% na época {best_epoch}")
            break

        print("-" * 50)

    print("\n" + "="*50)
    print("Treinamento concluído!")
    print(f"Melhor validação: {best_val_acc:.2f}% na época {best_epoch}")
    print(f"Modelo salvo em: {save_path / 'best_model.pth'}")
    print("="*50)


def main():
    """Função principal com argparse."""
    parser = argparse.ArgumentParser(
        description="Treinar modelo de detecção de violência em vídeos"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=16,
        help="Número de frames por vídeo"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=50,
        help="Número de épocas"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Taxa de aprendizado"
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        help="Tamanho do hidden state da LSTM"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Número de camadas LSTM"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Taxa de dropout"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Número de workers para DataLoader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu). Se None, detecta automaticamente"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodutibilidade"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Número de épocas sem melhoria para early stopping (0 = desabilitado)"
    )
    parser.add_argument(
        "--use_scheduler",
        action="store_true",
        default=True,
        help="Usar learning rate scheduler"
    )
    parser.add_argument(
        "--no_scheduler",
        dest="use_scheduler",
        action="store_false",
        help="Desabilitar learning rate scheduler"
    )
    
    args = parser.parse_args()
    
    train(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        use_scheduler=args.use_scheduler
    )


if __name__ == "__main__":
    main()

