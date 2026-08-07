"""
Exemplo de uso do módulo de Fusão Multimodal.

Este script demonstra como:
1. Carregar dados multimodais
2. Criar e usar o modelo multimodal
3. Treinar e avaliar
"""

import torch
from src.models.multimodal_risk import create_multimodal_model
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.models.resnet_lstm import create_model as create_video_model
from src import paths as p


def example_load_data():
    """Exemplo de carregamento de dados multimodais."""
    print("=" * 60)
    print("Exemplo 1: Carregar Dados Multimodais")
    print("=" * 60)
    
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(
        video_data_root=str(p.PROCESSED_ROOT),
        pose_data_root=str(p.POSE_ROOT),
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=4,
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="flatten",
        dataset_name="rwf2000"
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Obter um batch
    video, pose, emotion, labels = next(iter(train_loader))
    
    print(f"\nShapes do batch:")
    print(f"  Video: {video.shape}")      # (batch, T, C, H, W) ou (batch, T, D_v)
    print(f"  Pose: {pose.shape}")        # (batch, T, D_p)
    print(f"  Emotion: {emotion.shape}")  # (batch, T, D_e)
    print(f"  Labels: {labels.shape}")    # (batch,)


def example_create_model():
    """Exemplo de criação de modelo multimodal."""
    print("\n" + "=" * 60)
    print("Exemplo 2: Criar Modelo Multimodal")
    print("=" * 60)
    
    # Criar modelo com late fusion
    model = create_multimodal_model(
        video_feature_dim=256,
        pose_feature_dim=99,
        emotion_feature_dim=8,
        num_frames=16,
        fusion_method="late",
        use_temporal_modeling=True,
        device="cpu"
    )
    
    print(f"Modelo criado:")
    print(f"  Fusion method: late")
    print(f"  Temporal modeling: True")
    print(f"  Parâmetros: {sum(p.numel() for p in model.parameters()):,}")
    
    # Testar forward pass
    batch_size = 2
    video_features = torch.randn(batch_size, 16, 256)  # (batch, T, D_v)
    pose_features = torch.randn(batch_size, 16, 99)    # (batch, T, D_p)
    emotion_features = torch.randn(batch_size, 16, 8)   # (batch, T, D_e)
    
    with torch.no_grad():
        output = model(video_features, pose_features, emotion_features)
    
    print(f"\nForward pass:")
    print(f"  Input shapes: video={video_features.shape}, pose={pose_features.shape}, emotion={emotion_features.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Probabilities: {torch.softmax(output, dim=1)}")


def example_different_fusion_methods():
    """Exemplo comparando diferentes métodos de fusão."""
    print("\n" + "=" * 60)
    print("Exemplo 3: Comparar Métodos de Fusão")
    print("=" * 60)
    
    batch_size = 2
    video_features = torch.randn(batch_size, 16, 256)
    pose_features = torch.randn(batch_size, 16, 99)
    emotion_features = torch.randn(batch_size, 16, 8)
    
    fusion_methods = ["early", "late", "attention"]
    
    for method in fusion_methods:
        model = create_multimodal_model(
            video_feature_dim=256,
            pose_feature_dim=99,
            emotion_feature_dim=8,
            num_frames=16,
            fusion_method=method,
            use_temporal_modeling=True,
            device="cpu"
        )
        
        with torch.no_grad():
            output = model(video_features, pose_features, emotion_features)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"\n{method.upper()} Fusion:")
        print(f"  Parâmetros: {num_params:,}")
        print(f"  Output shape: {output.shape}")


def example_training_loop():
    """Exemplo de loop de treinamento simplificado."""
    print("\n" + "=" * 60)
    print("Exemplo 4: Loop de Treinamento")
    print("=" * 60)
    
    # Criar modelo
    model = create_multimodal_model(
        video_feature_dim=256,
        pose_feature_dim=99,
        emotion_feature_dim=8,
        num_frames=16,
        fusion_method="late",
        use_temporal_modeling=True,
        device="cpu"
    )
    
    # Carregar modelo de vídeo
    video_model = create_video_model(
        num_frames=16,
        hidden_size=256,
        device="cpu"
    )
    video_model.eval()
    
    # Criar DataLoader
    train_loader, _, _ = get_multimodal_dataloaders(
        video_data_root=str(p.PROCESSED_ROOT),
        pose_data_root=str(p.POSE_ROOT),
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=2,
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="flatten",
        dataset_name="rwf2000"
    )
    
    # Loss e optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Um batch de exemplo
    video, pose, emotion, labels = next(iter(train_loader))
    
    # Extrair features de vídeo
    with torch.no_grad():
        video_features = video_model.get_features(video)  # (batch, D_v)
        video_features = video_features.unsqueeze(1).repeat(1, 16, 1)  # (batch, T, D_v)
    
    # Forward
    optimizer.zero_grad()
    outputs = model(video_features, pose, emotion)
    loss = criterion(outputs, labels)
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Predictions: {torch.softmax(outputs, dim=1)}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Exemplos de Uso do Módulo Multimodal")
    print("=" * 60)
    print("\nCertifique-se de ter:")
    print("  1. Dados processados (video, pose, emotion)")
    print("  2. Modelo de vídeo pré-treinado (opcional)")
    print()
    
    try:
        example_load_data()
        example_create_model()
        example_different_fusion_methods()
        example_training_loop()
        
        print("\n" + "=" * 60)
        print("Todos os exemplos executados!")
        print("=" * 60)
    except FileNotFoundError as e:
        print(f"\nErro: {e}")
        print(f"\nCertifique-se de ter executado:")
        print("  1. run_preprocessing.py frames (para vídeo)")
        print("  2. run_preprocessing.py pose (para pose)")
        print("  3. run_preprocessing.py emotion (para emoção)")
    except Exception as e:
        print(f"\nErro inesperado: {e}")
        import traceback
        traceback.print_exc()

