"""
Exemplo de uso do módulo de Pose Estimation.

Este script demonstra como:
1. Carregar dados de pose usando PoseSequenceDataset
2. Criar DataLoaders para treinamento
3. Iterar sobre os dados
"""

import torch
from src.pose.pose_dataset import PoseSequenceDataset, get_pose_dataloaders
from src import paths as p


def example_basic_usage():
    """Exemplo básico de uso do PoseSequenceDataset."""
    print("=" * 60)
    print("Exemplo 1: Uso básico do PoseSequenceDataset")
    print("=" * 60)
    
    # Criar dataset
    dataset = PoseSequenceDataset(
        pose_data_root=str(p.POSE_ROOT),
        split="train",
        window_size=16,
        normalize=True,
        flatten=False,
        dataset_name="rwf2000"
    )
    
    print(f"Tamanho do dataset: {len(dataset)}")
    
    # Obter uma amostra
    pose_window, label = dataset[0]
    
    print(f"\nShape do pose_window: {pose_window.shape}")
    print(f"Label: {label.item()}")
    print(f"Tipo: {pose_window.dtype}")
    
    # pose_window shape: (window_size, num_joints, 3)
    # onde 3 = (x, y, visibility)


def example_dataloader():
    """Exemplo de uso com DataLoaders."""
    print("\n" + "=" * 60)
    print("Exemplo 2: Uso com DataLoaders")
    print("=" * 60)
    
    # Criar DataLoaders
    train_loader, val_loader, test_loader = get_pose_dataloaders(
        pose_data_root=str(p.POSE_ROOT),
        batch_size=8,
        window_size=16,
        normalize=True,
        flatten=False,
        dataset_name="rwf2000"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Iterar sobre um batch
    for batch_idx, (pose_windows, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Shape dos pose_windows: {pose_windows.shape}")
        print(f"  Shape dos labels: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        # pose_windows shape: (batch_size, window_size, num_joints, 3)
        # labels shape: (batch_size,)
        
        if batch_idx >= 2:  # Mostrar apenas 3 batches
            break


def example_flatten_mode():
    """Exemplo usando modo flatten (útil para alguns modelos)."""
    print("\n" + "=" * 60)
    print("Exemplo 3: Modo Flatten")
    print("=" * 60)
    
    dataset = PoseSequenceDataset(
        pose_data_root=str(p.POSE_ROOT),
        split="train",
        window_size=16,
        normalize=True,
        flatten=True,  # Flatten keypoints
        dataset_name="rwf2000"
    )
    
    pose_window, label = dataset[0]
    
    print(f"Shape do pose_window (flatten): {pose_window.shape}")
    print(f"  Esperado: (window_size, num_joints * 3) = (16, 99)")
    
    # Útil para modelos que esperam entrada 1D por timestep


def example_model_usage():
    """Exemplo de como usar os dados com um modelo LSTM simples."""
    print("\n" + "=" * 60)
    print("Exemplo 4: Integração com Modelo LSTM")
    print("=" * 60)
    
    import torch.nn as nn
    
    # Criar um modelo LSTM simples para pose
    class SimplePoseLSTM(nn.Module):
        def __init__(self, input_size=99, hidden_size=128, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x shape: (batch, window_size, num_joints * 3)
            lstm_out, _ = self.lstm(x)
            # Usar último output
            last_output = lstm_out[:, -1, :]
            output = self.fc(last_output)
            return output
    
    # Criar modelo
    model = SimplePoseLSTM(input_size=99, hidden_size=128, num_classes=2)
    
    # Obter um batch
    train_loader, _, _ = get_pose_dataloaders(
        pose_data_root=str(p.POSE_ROOT),
        batch_size=4,
        window_size=16,
        flatten=True,  # Usar flatten para LSTM
        dataset_name="rwf2000"
    )
    
    pose_windows, labels = next(iter(train_loader))
    
    print(f"Input shape: {pose_windows.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(pose_windows)
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {torch.softmax(output, dim=1)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Exemplos de Uso do Módulo de Pose Estimation")
    print("=" * 60)
    print(f"\nCertifique-se de ter executado run_preprocessing.py pose primeiro!")
    print(f"Os dados de pose devem estar em {p.POSE_ROOT}/\n")
    
    try:
        example_basic_usage()
        example_dataloader()
        example_flatten_mode()
        example_model_usage()
        
        print("\n" + "=" * 60)
        print("Todos os exemplos executados com sucesso!")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print(f"\nCertifique-se de:")
        print("  1. Executar run_preprocessing.py pose primeiro")
        print(f"  2. Verificar se os dados estão em {p.POSE_ROOT}/")
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        import traceback
        traceback.print_exc()

