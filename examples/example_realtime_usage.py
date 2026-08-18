"""
Exemplos de uso do pipeline de inferência em tempo real.
"""

import torch
from src.inference.realtime_risk_detector import create_realtime_detector
from src.inference.multi_camera_detector import create_multi_camera_detector
from src import paths as p


def example_single_webcam():
    """Exemplo: Detecção com webcam única (backbone de vídeo CNN 3D, padrão)."""
    print("=" * 60)
    print("Exemplo 1: Webcam Única")
    print("=" * 60)
    
    detector = create_realtime_detector(
        multimodal_model_path="results/multimodal/best_model.pth",
        cnn3d_model_path="results/cnn3d/rwf2000/best_model.pth",
        emotion_model_path="results/emotion/best_model.pth",
        video_source="0",  # Webcam
        window_size=16,
        risk_threshold=0.8,
        consecutive_windows=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    detector.run(display=True)


def example_rtsp_stream():
    """Exemplo: Detecção com stream RTSP (CNN 3D, padrão)."""
    print("=" * 60)
    print("Exemplo 2: Stream RTSP")
    print("=" * 60)
    
    detector = create_realtime_detector(
        multimodal_model_path="results/multimodal/best_model.pth",
        cnn3d_model_path="results/cnn3d/rwf2000/best_model.pth",
        emotion_model_path="results/emotion/best_model.pth",
        video_source="rtsp://user:pass@192.168.1.100:554/stream",
        window_size=16,
        risk_threshold=0.8,
        consecutive_windows=3,
        device="cuda"
    )
    
    detector.run(display=True)


def example_resnet_lstm():
    """Exemplo: Usando ResNet-LSTM como backbone de vídeo (alternativo)."""
    print("=" * 60)
    print("Exemplo 3: ResNet-LSTM")
    print("=" * 60)
    
    detector = create_realtime_detector(
        multimodal_model_path="results/multimodal/best_model.pth",
        emotion_model_path="results/emotion/best_model.pth",
        video_source="0",
        window_size=16,
        risk_threshold=0.8,
        consecutive_windows=3,
        use_cnn3d=False,
        video_model_path="results/resnet_lstm/best_model.pth",
        device="cuda"
    )
    
    detector.run(display=True)


def example_multi_camera():
    """Exemplo: Múltiplas câmeras (CNN 3D, padrão)."""
    print("=" * 60)
    print("Exemplo 4: Múltiplas Câmeras")
    print("=" * 60)
    
    detector = create_multi_camera_detector(
        camera_sources=["0", "1", "rtsp://...", "rtsp://..."],
        multimodal_model_path="results/multimodal/best_model.pth",
        cnn3d_model_path="results/cnn3d/rwf2000/best_model.pth",
        emotion_model_path="results/emotion/best_model.pth",
        window_size=16,
        risk_threshold=0.8,
        consecutive_windows=3,
        device="cuda"
    )
    
    detector.run(display=True)


def example_custom_config():
    """Exemplo: Configuração customizada (CNN 3D, padrão)."""
    print("=" * 60)
    print("Exemplo 5: Configuração Customizada")
    print("=" * 60)
    
    detector = create_realtime_detector(
        multimodal_model_path="results/multimodal/best_model.pth",
        cnn3d_model_path="results/cnn3d/rwf2000/best_model.pth",
        emotion_model_path="results/emotion/best_model.pth",
        video_source="0",
        window_size=32,  # Janela maior
        overlap=16,  # Mais overlap
        frame_size=(160, 160),  # Resolução menor para performance
        risk_threshold=0.85,  # Threshold mais alto
        consecutive_windows=5,  # Mais janelas para alerta
        device="cuda"
    )
    
    detector.run(display=True)


if __name__ == "__main__":
    import torch
    
    print("\n" + "=" * 60)
    print("Exemplos de Uso - Pipeline de Inferência em Tempo Real")
    print("=" * 60)
    print("\nCertifique-se de ter:")
    print("  1. Modelos treinados (multimodal, video, emotion)")
    print("  2. Webcam conectada ou URL RTSP válida")
    print("  3. GPU disponível (recomendado)")
    print()
    
    try:
        # Descomente o exemplo que deseja executar:
        
        # example_single_webcam()
        # example_rtsp_stream()
        # example_resnet_lstm()
        # example_multi_camera()
        # example_custom_config()
        
        print("\n⚠ Descomente um dos exemplos acima para executar")
        print("\nOu use o script principal:")
        print("  python run_realtime_risk_detection.py --multimodal_model results/multimodal/best_model.pth")
        
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()

