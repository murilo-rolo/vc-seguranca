"""
Exemplo de uso do pipeline completo de detecção de violência em vídeos.

Este script demonstra como usar os módulos do projeto de forma programática.
"""

import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import create_model
from src.datasets import get_dataloaders
from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset
from src import paths as p


def example_preprocessing():
    """Exemplo de pré-processamento."""
    print("="*50)
    print("EXEMPLO: Pré-processamento")
    print("="*50)
    
    # Organizar vídeos
    print("\n1. Organizando vídeos...")
    num_violent, num_non_violent = organize_rwf2000_dataset(
        dataset_root=str(p.RWF2000_ROOT),
        output_root=str(p.RAW_DATA_ROOT)
    )
    print(f"   Organizados: {num_violent} violentos, {num_non_violent} não violentos")
    
    # Extrair frames
    print("\n2. Extraindo frames...")
    preprocess_dataset(
        raw_data_root=str(p.RAW_DATA_ROOT),
        processed_data_root=str(p.PROCESSED_ROOT),
        num_frames=16,
        target_size=(112, 112),
        normalize=True
    )
    print("   Frames extraídos com sucesso!")


def example_dataloader():
    """Exemplo de uso do DataLoader."""
    print("\n" + "="*50)
    print("EXEMPLO: DataLoader")
    print("="*50)
    
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_data_root=str(p.PROCESSED_ROOT),
        batch_size=4,
        num_frames=16,
        num_workers=0,  # 0 para debug
        seed=42
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Obter um batch de exemplo
    frames, labels = next(iter(train_loader))
    print(f"\nExemplo de batch:")
    print(f"  Frames shape: {frames.shape}")  # (batch, num_frames, C, H, W)
    print(f"  Labels shape: {labels.shape}")   # (batch,)
    print(f"  Labels: {labels.tolist()}")


def example_model():
    """Exemplo de criação e uso do modelo."""
    print("\n" + "="*50)
    print("EXEMPLO: Modelo")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Criar modelo
    model = create_model(
        num_frames=16,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        num_classes=2,
        pretrained=True,
        device=device
    )
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal de parâmetros: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Forward pass de exemplo
    batch_size = 2
    example_frames = torch.randn(batch_size, 16, 3, 112, 112).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(example_frames)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {example_frames.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output (logits): {output.cpu().numpy()}")
    
    # Predições
    _, predicted = torch.max(output, 1)
    print(f"  Predições: {predicted.cpu().numpy()}")


if __name__ == "__main__":
    print("EXEMPLOS DE USO DO PIPELINE DE DETECÇÃO DE VIOLÊNCIA")
    print("="*50)
    
    # Descomente as funções que deseja executar:
    
    # example_preprocessing()  # Requer dataset na pasta dataset/
    # example_dataloader()     # Requer dados processados
    example_model()            # Pode executar sem dados
    
    print("\n" + "="*50)
    print("Exemplos concluídos!")
    print("="*50)

