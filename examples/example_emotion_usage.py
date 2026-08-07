"""
Exemplo de uso do módulo de Emotion Recognition.

Este script demonstra como:
1. Carregar dados de emoção usando EmotionSequenceDataset
2. Criar DataLoaders para treinamento
3. Iterar sobre os dados
4. Usar em modelos temporais
"""

import torch
from src.emotion.emotion_dataset import EmotionSequenceDataset, get_emotion_dataloaders
from src import paths as p


def example_basic_usage():
    """Exemplo básico de uso do EmotionSequenceDataset."""
    print("=" * 60)
    print("Exemplo 1: Uso básico do EmotionSequenceDataset")
    print("=" * 60)
    
    # Criar dataset
    dataset = EmotionSequenceDataset(
        emotion_data_root=str(p.EMOTION_ROOT),
        split="train",
        window_size=16,
        normalize=False,  # Emoções já são probabilidades
        dataset_name="rwf2000"
    )
    
    print(f"Tamanho do dataset: {len(dataset)}")
    
    # Obter uma amostra
    emotion_window, label = dataset[0]
    
    print(f"\nShape do emotion_window: {emotion_window.shape}")
    print(f"Label: {label.item()}")
    print(f"Tipo: {emotion_window.dtype}")
    print(f"Soma das probabilidades (primeiro frame): {emotion_window[0].sum().item():.4f}")
    
    # emotion_window shape: (window_size, num_emotions)
    # onde num_emotions = 8 (AffectNet)


def example_dataloader():
    """Exemplo de uso com DataLoaders."""
    print("\n" + "=" * 60)
    print("Exemplo 2: Uso com DataLoaders")
    print("=" * 60)
    
    # Criar DataLoaders
    train_loader, val_loader, test_loader = get_emotion_dataloaders(
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=8,
        window_size=16,
        normalize=False,
        dataset_name="rwf2000"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Iterar sobre um batch
    for batch_idx, (emotion_windows, labels) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Shape dos emotion_windows: {emotion_windows.shape}")
        print(f"  Shape dos labels: {labels.shape}")
        print(f"  Labels: {labels.tolist()}")
        
        # emotion_windows shape: (batch_size, window_size, num_emotions)
        # labels shape: (batch_size,)
        
        # Mostrar emoção dominante do primeiro vídeo
        first_video = emotion_windows[0]  # (window_size, num_emotions)
        avg_emotions = first_video.mean(dim=0)  # Média temporal
        dominant_emotion = avg_emotions.argmax().item()
        print(f"  Emoção dominante (primeiro vídeo): {dominant_emotion}")
        
        if batch_idx >= 2:  # Mostrar apenas 3 batches
            break


def example_model_usage():
    """Exemplo de como usar os dados com um modelo LSTM simples."""
    print("\n" + "=" * 60)
    print("Exemplo 3: Integração com Modelo LSTM")
    print("=" * 60)
    
    import torch.nn as nn
    
    # Criar um modelo LSTM simples para emoções
    class SimpleEmotionLSTM(nn.Module):
        def __init__(self, input_size=8, hidden_size=128, num_classes=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            # x shape: (batch, window_size, num_emotions)
            lstm_out, _ = self.lstm(x)
            # Usar último output
            last_output = lstm_out[:, -1, :]
            output = self.fc(last_output)
            return output
    
    # Criar modelo
    model = SimpleEmotionLSTM(input_size=8, hidden_size=128, num_classes=2)
    
    # Obter um batch
    train_loader, _, _ = get_emotion_dataloaders(
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=4,
        window_size=16,
        dataset_name="rwf2000"
    )
    
    emotion_windows, labels = next(iter(train_loader))
    
    print(f"Input shape: {emotion_windows.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(emotion_windows)
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {torch.softmax(output, dim=1)}")


def example_multimodal_integration():
    """Exemplo de como integrar emoções com outros modais."""
    print("\n" + "=" * 60)
    print("Exemplo 4: Integração Multimodal")
    print("=" * 60)
    
    # Simular features de diferentes modais
    emotion_features = torch.randn(4, 128)  # Features de emoção (batch, emotion_dim)
    pose_features = torch.randn(4, 256)    # Features de pose (batch, pose_dim)
    resnet_features = torch.randn(4, 512)  # Features de ResNet (batch, resnet_dim)
    
    print("Features de diferentes modais:")
    print(f"  Emoção: {emotion_features.shape}")
    print(f"  Pose: {pose_features.shape}")
    print(f"  ResNet: {resnet_features.shape}")
    
    # Concatenar features
    multimodal_features = torch.cat([
        emotion_features,
        pose_features,
        resnet_features
    ], dim=1)
    
    print(f"\nFeatures multimodais concatenadas: {multimodal_features.shape}")
    
    # Classificador final
    classifier = torch.nn.Linear(896, 2)  # 128 + 256 + 512 = 896
    output = classifier(multimodal_features)
    print(f"Saída do classificador: {output.shape}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Exemplos de Uso do Módulo de Emotion Recognition")
    print("=" * 60)
    print(f"\nCertifique-se de ter executado run_preprocessing.py emotion primeiro!")
    print(f"Os dados de emoção devem estar em {p.EMOTION_ROOT}/\n")
    
    try:
        example_basic_usage()
        example_dataloader()
        example_model_usage()
        example_multimodal_integration()
        
        print("\n" + "=" * 60)
        print("Todos os exemplos executados com sucesso!")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print(f"\nCertifique-se de:")
        print("  1. Executar run_preprocessing.py emotion primeiro")
        print(f"  2. Verificar se os dados estão em {p.EMOTION_ROOT}/")
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        import traceback
        traceback.print_exc()

