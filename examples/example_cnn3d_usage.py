"""
Exemplo de uso do módulo CNN 3D.

Este script demonstra como:
1. Criar modelo CNN 3D
2. Carregar dados
3. Fazer forward pass
4. Carregar pesos pré-treinados
"""

import torch
from src.models.cnn3d_risk import create_cnn3d_model
from src.datasets.video3d_dataset import (
    get_rwf2000_3d_dataloaders
)
from src import paths as p


def example_create_model():
    """Exemplo de criação de modelo CNN 3D."""
    print("=" * 60)
    print("Exemplo 1: Criar Modelo CNN 3D")
    print("=" * 60)
    
    # Criar modelo para RWF-2000 (2 classes)
    model_rwf2000 = create_cnn3d_model(
        model_name="r2plus1d_18",
        num_classes=2,
        pretrained=True,
        pretrained_dataset="kinetics400",
        device="cpu"
    )
    
    print(f"\nModelo RWF-2000:")
    print(f"  Modelo: R(2+1)D-18")
    print(f"  Classes: 2")
    print(f"  Parâmetros: {sum(p.numel() for p in model_rwf2000.parameters()):,}")


def example_forward_pass():
    """Exemplo de forward pass."""
    print("\n" + "=" * 60)
    print("Exemplo 2: Forward Pass")
    print("=" * 60)
    
    model = create_cnn3d_model(
        model_name="r2plus1d_18",
        num_classes=2,
        pretrained=False,  # Não carregar pesos para exemplo rápido
        device="cpu"
    )
    
    # Criar clipe de exemplo
    # Formato: (batch, T, C, H, W) - será convertido para (batch, C, T, H, W) internamente
    batch_size = 2
    num_frames = 16
    clip = torch.randn(batch_size, num_frames, 3, 112, 112)
    
    print(f"Input shape: {clip.shape}")  # (batch, T, C, H, W)
    
    # Forward pass
    with torch.no_grad():
        output = model(clip)
    
    print(f"Output shape: {output.shape}")  # (batch, num_classes)
    print(f"Probabilities: {torch.softmax(output, dim=1)}")


def example_load_pretrained():
    """Exemplo de carregar pesos pré-treinados."""
    print("\n" + "=" * 60)
    print("Exemplo 3: Carregar Pesos Pré-treinados")
    print("=" * 60)
    
    # Criar modelo
    model = create_cnn3d_model(
        model_name="r2plus1d_18",
        num_classes=2,
        pretrained=False,
        device="cpu"
    )
    
    # Carregar checkpoint de fine-tuning em RWF-2000
    checkpoint_path = str(p.CNN3D_WEIGHTS / "best_model.pth")
    
    try:
        model_with_pretrained = create_cnn3d_model(
            model_name="r2plus1d_18",
            num_classes=2,
            pretrained=False,
            checkpoint_path=checkpoint_path,
            device="cpu"
        )
        print(f"✓ Modelo carregado de: {checkpoint_path}")
    except FileNotFoundError:
        print(f"⚠ Checkpoint não encontrado: {checkpoint_path}")
        print("  Execute primeiro: python train_cnn3d.py")


def example_dataloader():
    """Exemplo de uso com DataLoaders."""
    print("\n" + "=" * 60)
    print("Exemplo 4: Usar com DataLoaders")
    print("=" * 60)
    
    try:
        # RWF-2000
        train_loader, val_loader = get_rwf2000_3d_dataloaders(
            dataset_root=str(p.RWF2000_ROOT),
            batch_size=2,
            num_frames=16,
            clip_size=(112, 112),
            num_workers=0
        )
        
        print(f"\nRWF-2000 DataLoaders:")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        # Obter um batch
        clip, label = next(iter(train_loader))
        print(f"\nBatch shape: {clip.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Label: {label.tolist()}")
        
    except FileNotFoundError:
        print(f"\n⚠ Dataset RWF-2000 não encontrado em {p.RWF2000_ROOT}")
        print("  Certifique-se de que o dataset está no local correto")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Exemplos de Uso do Módulo CNN 3D")
    print("=" * 60)
    print()
    
    try:
        example_create_model()
        example_forward_pass()
        example_load_pretrained()
        example_dataloader()
        
        print("\n" + "=" * 60)
        print("Todos os exemplos executados!")
        print("=" * 60)
    except ImportError as e:
        print(f"\nErro de importação: {e}")
        print("\nCertifique-se de ter:")
        print("  - torchvision >= 0.13.0 (para modelos 3D)")
        print("  - torch >= 2.0.0")
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        import traceback
        traceback.print_exc()

