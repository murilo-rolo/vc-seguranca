"""
Sistema de detecção de risco para múltiplas câmeras.

Este módulo estende o RealTimeRiskDetector para suportar múltiplas câmeras
simultaneamente, com processamento paralelo e gerenciamento centralizado de alertas.
"""

import cv2
import torch
import numpy as np
from typing import List, Dict, Optional
from threading import Thread, Lock
import time
from collections import deque

from .realtime_risk_detector import RealTimeRiskDetector


class MultiCameraRiskDetector:
    """
    Sistema de detecção de risco para múltiplas câmeras.
    
    Gerencia múltiplas instâncias de RealTimeRiskDetector, uma por câmera,
    com processamento paralelo e interface unificada.
    """
    
    def __init__(
        self,
        camera_sources: List[str],
        multimodal_model_path: str,
        video_model_path: Optional[str] = None,
        emotion_model_path: Optional[str] = None,
        window_size: int = 16,
        risk_threshold: float = 0.8,
        consecutive_windows: int = 3,
        use_cnn3d: bool = True,
        cnn3d_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa detector para múltiplas câmeras.
        
        Args:
            camera_sources: Lista de fontes de vídeo (webcam IDs ou URLs RTSP)
            multimodal_model_path: Caminho para modelo multimodal
            video_model_path: Caminho para modelo de vídeo
            emotion_model_path: Caminho para modelo de emoção
            window_size: Tamanho da janela temporal
            risk_threshold: Threshold de probabilidade
            consecutive_windows: Janelas consecutivas para alerta
            use_cnn3d: Se True (padrão), usa CNN 3D; se False, ResNet-LSTM
            cnn3d_model_path: Caminho para modelo CNN 3D
            device: Device para inferência
        """
        self.camera_sources = camera_sources
        self.num_cameras = len(camera_sources)
        
        # Criar detector para cada câmera
        self.detectors = []
        for i, source in enumerate(camera_sources):
            print(f"Criando detector para câmera {i+1}/{self.num_cameras}: {source}")
            detector = RealTimeRiskDetector(
                multimodal_model_path=multimodal_model_path,
                video_model_path=video_model_path,
                emotion_model_path=emotion_model_path,
                video_source=source,
                window_size=window_size,
                risk_threshold=risk_threshold,
                consecutive_windows=consecutive_windows,
                use_cnn3d=use_cnn3d,
                cnn3d_model_path=cnn3d_model_path,
                device=device
            )
            self.detectors.append(detector)
        
        # Status global
        self.global_alert = False
        self.camera_status = {}  # {camera_id: {'risk': float, 'alert': bool}}
        self.status_lock = Lock()
        
        # Threads de processamento
        self.threads = []
        self.running = False
    
    def _camera_worker(self, camera_id: int, detector: RealTimeRiskDetector):
        """Worker thread para processar uma câmera."""
        # Abrir captura
        source = self.camera_sources[camera_id]
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Erro ao abrir câmera {camera_id}: {source}")
            return
        
        frame_buffer = deque(maxlen=detector.window_size)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Processar frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, detector.frame_size)
            frame_buffer.append(frame_resized)
            
            # Processar janela quando completa
            risk_prob = 0.0
            if len(frame_buffer) == detector.window_size:
                frames_array = np.array(list(frame_buffer))
                risk_prob = detector._process_window(frames_array)
                detector._update_alert_status(risk_prob)
                
                # Atualizar status global
                with self.status_lock:
                    self.camera_status[camera_id] = {
                        'risk': risk_prob,
                        'alert': detector.alert_active
                    }
                    
                    # Verificar se alguma câmera tem alerta
                    self.global_alert = any(
                        status.get('alert', False) for status in self.camera_status.values()
                    )
                
                # Remover frames antigos (overlap)
                for _ in range(detector.overlap):
                    if len(frame_buffer) > 0:
                        frame_buffer.popleft()
            
            # Exibir frame (opcional)
            if display:
                fps = detector.fps_history[-1] if len(detector.fps_history) > 0 else 0.0
                frame_display = detector._draw_overlay(
                    frame,
                    risk_prob,
                    fps
                )
                cv2.imshow(f'Camera {camera_id}', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
        
        cap.release()
    
    def run(self, display: bool = True):
        """
        Executa detecção para todas as câmeras.
        
        Args:
            display: Se True, exibe vídeo de cada câmera
        """
        print("=" * 60)
        print(f"Iniciando detecção para {self.num_cameras} câmeras")
        print("Pressione 'q' para sair")
        print("=" * 60)
        
        self.running = True
        
        # Iniciar threads para cada câmera
        for i, detector in enumerate(self.detectors):
            thread = Thread(
                target=self._camera_worker,
                args=(i, detector),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        try:
            # Loop principal para exibição e controle
            while self.running:
                # Verificar tecla 'q'
                if display and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Exibir status global
                if self.global_alert:
                    print("⚠ ALERTA: Risco detectado em uma ou mais câmeras!")
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        finally:
            self.running = False
            
            # Aguardar threads terminarem
            for thread in self.threads:
                thread.join(timeout=2.0)
            
            if display:
                cv2.destroyAllWindows()
            
            print("Pipeline finalizado")


def create_multi_camera_detector(
    camera_sources: List[str],
    multimodal_model_path: str,
    **kwargs
) -> MultiCameraRiskDetector:
    """
    Função auxiliar para criar detector multi-câmera.
    
    Args:
        camera_sources: Lista de fontes de vídeo
        multimodal_model_path: Caminho para modelo multimodal
        **kwargs: Argumentos adicionais para RealTimeRiskDetector
    
    Returns:
        MultiCameraRiskDetector configurado
    """
    return MultiCameraRiskDetector(
        camera_sources=camera_sources,
        multimodal_model_path=multimodal_model_path,
        **kwargs
    )

