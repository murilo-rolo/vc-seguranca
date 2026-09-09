"""
Pipeline de inferência em tempo real para detecção de risco/violência.

Este módulo implementa um sistema completo de detecção em tempo real que:
1. Captura vídeo de webcam ou RTSP
2. Processa janelas temporais de frames
3. Extrai features de múltiplas modalidades (vídeo, pose, emoção)
4. Combina usando modelo multimodal
5. Gera alertas baseados em threshold e janelas consecutivas
6. Exibe resultado com overlay visual
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import time
from threading import Thread, Lock
import queue

from ..models.multimodal_risk import create_multimodal_model
from ..models.resnet_lstm import create_model as create_video_model
from ..models.cnn3d_risk import create_cnn3d_model
from ..pose.extract_pose import PoseExtractor
from ..emotion.extract_emotion import EmotionExtractor, FaceDetector
from ..models.emotion_cnn import create_emotion_model


class RealTimeRiskDetector:
    """
    Sistema de detecção de risco em tempo real.
    
    Pipeline:
    1. Captura frames de vídeo
    2. Acumula janelas temporais
    3. Extrai features (vídeo, pose, emoção)
    4. Combina com modelo multimodal
    5. Gera alertas
    6. Exibe resultado
    """
    
    def __init__(
        self,
        # Modelos
        multimodal_model_path: str,
        video_model_path: Optional[str] = None,
        emotion_model_path: Optional[str] = None,
        
        # Configuração de vídeo
        video_source: str = "0",  # "0" para webcam ou URL RTSP
        window_size: int = 16,
        overlap: int = 8,  # Sobreposição entre janelas
        
        # Configuração de processamento
        frame_size: Tuple[int, int] = (224, 224),  # Tamanho para processamento
        num_frames: int = 16,
        
        # Configuração de alertas
        risk_threshold: float = 0.8,
        consecutive_windows: int = 3,  # Número de janelas consecutivas para alerta
        
        # Configuração de modelos
        use_cnn3d: bool = True,  # Se True, usa CNN 3D (padrão), senão ResNet-LSTM
        cnn3d_model_path: Optional[str] = None,
        
        # Agregação de faces por frame
        face_aggregation: str = "mean",  # "mean" ou "max" — todas as faces do frame
        
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa o detector em tempo real.
        
        Args:
            multimodal_model_path: Caminho para modelo multimodal
            video_model_path: Caminho para modelo de vídeo (se usar ResNet-LSTM)
            emotion_model_path: Caminho para modelo de emoção
            video_source: Fonte de vídeo ("0" para webcam ou URL RTSP)
            window_size: Tamanho da janela temporal
            overlap: Sobreposição entre janelas
            frame_size: Tamanho dos frames para processamento
            num_frames: Número de frames por janela
            risk_threshold: Threshold de probabilidade para alerta
            consecutive_windows: Janelas consecutivas acima do threshold para alerta
            use_cnn3d: Se True (padrão), usa CNN 3D para vídeo; se False, ResNet-LSTM
            cnn3d_model_path: Caminho para modelo CNN 3D (se use_cnn3d=True)
            face_aggregation: Agregação de TODAS as faces por frame ("mean" ou "max")
            device: Device para inferência
        """
        if face_aggregation not in ("mean", "max"):
            raise ValueError(
                f"face_aggregation deve ser 'mean' ou 'max', recebido: {face_aggregation}"
            )
        self.video_source = video_source
        self.window_size = window_size
        self.overlap = overlap
        self.frame_size = frame_size
        self.num_frames = num_frames
        self.risk_threshold = risk_threshold
        self.consecutive_windows = consecutive_windows
        self.use_cnn3d = use_cnn3d
        self.face_aggregation = face_aggregation
        self.device = torch.device(device)
        
        # Buffer de frames
        self.frame_buffer = deque(maxlen=window_size)
        self.processed_windows = deque(maxlen=consecutive_windows)
        
        # Estatísticas
        self.fps_history = deque(maxlen=30)
        self.risk_history = deque(maxlen=30)
        self.alert_active = False
        self.alert_count = 0
        
        # Carregar modelos
        print("Carregando modelos...")
        self._load_models(
            multimodal_model_path,
            video_model_path,
            emotion_model_path,
            cnn3d_model_path
        )
        print("✓ Modelos carregados")
        
        # Inicializar extratores
        self._init_extractors()
        
        # Threading para processamento assíncrono
        self.processing_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing_thread = None
        self.running = False
    
    def _load_models(
        self,
        multimodal_model_path: str,
        video_model_path: Optional[str],
        emotion_model_path: Optional[str],
        cnn3d_model_path: Optional[str]
    ):
        """Carrega todos os modelos necessários."""
        # Modelo de vídeo primeiro — define D_v (necessário para o multimodal)
        if self.use_cnn3d:
            if cnn3d_model_path:
                # Backbone dirigido pela metadata do checkpoint (model_name/backbone),
                # com fallback para o default r2plus1d_18 (pesos Kinetics-400).
                checkpoint = torch.load(cnn3d_model_path, map_location=self.device)
                model_name = checkpoint.get('model_name') or checkpoint.get('backbone') or "r2plus1d_18"
                # Mesma tolerância de train_cnn3d.py: create_cnn3d_model carrega
                # o checkpoint RWF-2000 (2 classes) via state_dict completo.
                self.video_model = create_cnn3d_model(
                    model_name=model_name,
                    num_classes=2,
                    checkpoint_path=cnn3d_model_path,
                    num_frames=self.num_frames,
                    device=self.device
                )
                video_feature_dim = self.video_model.feature_dim  # 512
                self.video_model.eval()
            else:
                raise ValueError("cnn3d_model_path necessário quando use_cnn3d=True")
        else:
            if video_model_path:
                self.video_model = create_video_model(
                    num_frames=self.num_frames,
                    hidden_size=256,
                    num_layers=2,
                    dropout=0.3,
                    num_classes=2,
                    pretrained=True,
                    device=self.device
                )
                checkpoint = torch.load(video_model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.video_model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.video_model.load_state_dict(checkpoint)
            else:
                # Criar modelo básico sem checkpoint
                self.video_model = create_video_model(
                    num_frames=self.num_frames,
                    hidden_size=256,
                    device=self.device
                )
            self.video_model.eval()
            video_feature_dim = 256

        # Modelo multimodal
        multimodal_checkpoint = torch.load(multimodal_model_path, map_location=self.device)
        fusion_method = multimodal_checkpoint.get('fusion_method', 'cross_attention')
        use_temporal = multimodal_checkpoint.get('use_temporal_modeling', True)

        self.multimodal_model = create_multimodal_model(
            video_feature_dim=multimodal_checkpoint.get('video_feature_dim', video_feature_dim),
            pose_feature_dim=51,
            emotion_feature_dim=multimodal_checkpoint.get('emotion_feature_dim', 128),
            num_frames=self.num_frames,
            fusion_method=fusion_method,
            use_temporal_modeling=use_temporal,
            device=self.device
        )
        self.multimodal_model.load_state_dict(multimodal_checkpoint['model_state_dict'])
        self.multimodal_model.eval()
        
        # Modelo de emoção
        if emotion_model_path:
            self.emotion_model = create_emotion_model(
                num_emotions=8,
                checkpoint_path=emotion_model_path,
                device=self.device
            )
        else:
            # Criar modelo básico sem checkpoint
            self.emotion_model = create_emotion_model(
                num_emotions=8,
                pretrained=False,
                device=self.device
            )
        self.emotion_model.eval()
    
    def _init_extractors(self):
        """Inicializa extratores de pose e emoção."""
        # Extrator de pose
        self.pose_extractor = PoseExtractor(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        # Extrator de emoção
        face_detector = FaceDetector(method="mtcnn", device=self.device)
        self.emotion_extractor = EmotionExtractor(
            model=self.emotion_model,
            face_detector=face_detector,
            aggregation="mean",
            face_aggregation=self.face_aggregation
        )
    
    def _extract_video_features(self, frames: np.ndarray) -> torch.Tensor:
        """
        Extrai features de vídeo de uma janela de frames.

        Args:
            frames: Array (T, H, W, C) de frames

        Returns:
            Clip token (1, D_v) de features de vídeo
        """
        # Converter para tensor
        frames_tensor = torch.from_numpy(frames).float()
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        frames_tensor = frames_tensor / 255.0
        
        # Normalizar com ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std
        
        frames_tensor = frames_tensor.to(self.device)
        
        with torch.no_grad():
            if self.use_cnn3d:
                # CNN 3D espera (batch, C, T, H, W)
                frames_3d = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)
                # Clip token (1, D_v) — o backbone já modela o tempo internamente
                features = self.video_model.get_features(frames_3d)  # (1, D_v)
                return features
            else:
                # ResNet-LSTM
                frames_batch = frames_tensor.unsqueeze(0)  # (1, T, C, H, W)
                features = self.video_model.get_features(frames_batch)  # (1, D_v)
                # Clip token (1, D_v)
                return features
    
    def _extract_pose_features(self, frames: np.ndarray) -> torch.Tensor:
        """
        Extrai features de pose de uma janela de frames.
        
        Args:
            frames: Array (T, H, W, C) de frames
        
        Returns:
            Features de pose (T, num_joints, 3)
        """
        keypoints_list = []
        
        for frame in frames:
            keypoints = self.pose_extractor.extract_keypoints_from_frame(frame)
            if keypoints is None:
                # Se não detectar, usar zeros
                keypoints = np.zeros((17, 3))
            keypoints_list.append(keypoints)
        
        # Converter para tensor
        keypoints_array = np.array(keypoints_list)  # (T, 17, 3)
        keypoints_tensor = torch.from_numpy(keypoints_array).float()
        
        return keypoints_tensor
    
    def _extract_emotion_features(self, frames: np.ndarray) -> torch.Tensor:
        """
        Extrai features de emoção de uma janela de frames.

        Args:
            frames: Array (T, H, W, C) de frames

        Returns:
            Features de emoção (T, 128) — embeddings da penúltima camada
        """
        emotion_vectors = []
        
        for frame in frames:
            # Agrega TODAS as faces do frame (mean/max) em um único embedding (128,)
            emotion_vec = self.emotion_extractor.extract_from_frame(
                frame, face_aggregation=self.face_aggregation
            )
            if emotion_vec is None:
                # Se não detectar face, usar o embedding neutro (128-d)
                emotion_vec = self.emotion_extractor.neutral_embedding
            emotion_vectors.append(emotion_vec)
        
        # Converter para tensor
        emotion_array = np.array(emotion_vectors)  # (T, 128)
        emotion_tensor = torch.from_numpy(emotion_array).float()
        
        return emotion_tensor
    
    def _process_window(self, frames: np.ndarray) -> float:
        """
        Processa uma janela de frames e retorna probabilidade de risco.
        
        Args:
            frames: Array (T, H, W, C) de frames
        
        Returns:
            Probabilidade de risco (0-1)
        """
        # Extrair features
        video_features = self._extract_video_features(frames)  # (1, D_v)
        pose_features = self._extract_pose_features(frames)  # (T, 17, 3)
        emotion_features = self._extract_emotion_features(frames)  # (T, 128)
        
        # Converter para formato do modelo multimodal
        video_features = video_features.to(self.device)  # (1, D_v) clip token
        pose_features = pose_features.unsqueeze(0).to(self.device)  # (1, T, 17, 3)
        emotion_features = emotion_features.unsqueeze(0).to(self.device)  # (1, T, 128)
        
        # Flatten pose se necessário
        if len(pose_features.shape) == 4:
            pose_features = pose_features.view(1, self.num_frames, -1)  # (1, T, 51)
        
        # Inferência multimodal
        with torch.no_grad():
            logits = self.multimodal_model(
                video_features,
                pose_features,
                emotion_features
            )
            probs = F.softmax(logits, dim=1)
            risk_prob = probs[0, 1].item()  # Probabilidade da classe "violent"
        
        return risk_prob
    
    def _processing_worker(self):
        """Worker thread para processamento assíncrono."""
        while self.running:
            try:
                frames = self.processing_queue.get(timeout=1.0)
                if frames is None:
                    break
                
                # Processar janela
                risk_prob = self._process_window(frames)
                
                # Enviar resultado
                try:
                    self.result_queue.put_nowait(risk_prob)
                except queue.Full:
                    # Descartar resultado antigo
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.result_queue.put_nowait(risk_prob)
                
                self.processing_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Erro no worker: {e}")
                continue
    
    def _update_alert_status(self, risk_prob: float):
        """Atualiza status de alerta baseado em probabilidade."""
        self.risk_history.append(risk_prob)
        
        # Verificar se está acima do threshold
        if risk_prob >= self.risk_threshold:
            self.alert_count += 1
        else:
            self.alert_count = 0
        
        # Ativar alerta se K janelas consecutivas acima do threshold
        if self.alert_count >= self.consecutive_windows:
            self.alert_active = True
        else:
            self.alert_active = False
    
    def _draw_overlay(self, frame: np.ndarray, risk_prob: float, fps: float) -> np.ndarray:
        """
        Desenha overlay no frame com informações de risco.
        
        Args:
            frame: Frame original
            risk_prob: Probabilidade de risco
            fps: FPS atual
        
        Returns:
            Frame com overlay
        """
        frame_copy = frame.copy()
        H, W = frame_copy.shape[:2]
        
        # Cor baseada no risco
        if self.alert_active:
            color = (0, 0, 255)  # Vermelho
            status = "RISCO ALTO!"
        elif risk_prob >= self.risk_threshold * 0.7:
            color = (0, 165, 255)  # Laranja
            status = "Atenção"
        else:
            color = (0, 255, 0)  # Verde
            status = "Normal"
        
        # Fundo semi-transparente para texto
        overlay = frame_copy.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
        
        # Texto de status
        cv2.putText(
            frame_copy,
            f"Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        # Probabilidade
        cv2.putText(
            frame_copy,
            f"Risco: {risk_prob:.2%}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # FPS
        cv2.putText(
            frame_copy,
            f"FPS: {fps:.1f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Barra de risco
        bar_width = int(W * 0.3)
        bar_height = 20
        bar_x = W - bar_width - 10
        bar_y = 10
        
        # Fundo da barra
        cv2.rectangle(
            frame_copy,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50),
            -1
        )
        
        # Preenchimento baseado no risco
        fill_width = int(bar_width * risk_prob)
        cv2.rectangle(
            frame_copy,
            (bar_x, bar_y),
            (bar_x + fill_width, bar_y + bar_height),
            color,
            -1
        )
        
        return frame_copy
    
    def run(self, display: bool = True):
        """
        Executa o loop principal de detecção em tempo real.
        
        Args:
            display: Se True, exibe vídeo com overlay
        """
        # Abrir captura de vídeo
        if self.video_source.isdigit():
            cap = cv2.VideoCapture(int(self.video_source))
        else:
            cap = cv2.VideoCapture(self.video_source)  # RTSP URL
        
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir fonte de vídeo: {self.video_source}")
        
        # Configurar resolução (opcional, para melhor performance)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Iniciando detecção em tempo real...")
        print("Pressione 'q' para sair")
        
        # Iniciar thread de processamento
        self.running = True
        self.processing_thread = Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        frame_count = 0
        last_time = time.time()
        current_risk = 0.0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Erro ao ler frame ou fim do stream")
                    break
                
                # Converter BGR para RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Redimensionar para processamento
                frame_resized = cv2.resize(frame_rgb, self.frame_size)
                
                # Adicionar ao buffer
                self.frame_buffer.append(frame_resized)
                
                # Processar quando tiver janela completa
                if len(self.frame_buffer) == self.window_size:
                    # Tentar adicionar à fila de processamento
                    frames_array = np.array(list(self.frame_buffer))  # (T, H, W, C)
                    
                    try:
                        self.processing_queue.put_nowait(frames_array)
                    except queue.Full:
                        # Descartar frame antigo se fila cheia
                        try:
                            self.processing_queue.get_nowait()
                            self.processing_queue.put_nowait(frames_array)
                        except queue.Empty:
                            pass
                    
                    # Remover frames antigos (overlap)
                    for _ in range(self.overlap):
                        if len(self.frame_buffer) > 0:
                            self.frame_buffer.popleft()
                
                # Verificar resultados
                try:
                    current_risk = self.result_queue.get_nowait()
                    self._update_alert_status(current_risk)
                except queue.Empty:
                    pass
                
                # Calcular FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_time >= 1.0:
                    fps = frame_count / (current_time - last_time)
                    self.fps_history.append(fps)
                    frame_count = 0
                    last_time = current_time
                else:
                    fps = self.fps_history[-1] if len(self.fps_history) > 0 else 0.0
                
                # Desenhar overlay
                if display:
                    frame_display = self._draw_overlay(frame, current_risk, fps)
                    cv2.imshow('Risk Detection', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuário")
        finally:
            # Limpar recursos
            self.running = False
            cap.release()
            if display:
                cv2.destroyAllWindows()
            
            # Aguardar thread terminar
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            
            print("Pipeline finalizado")


def create_realtime_detector(
    multimodal_model_path: str,
    video_model_path: Optional[str] = None,
    emotion_model_path: Optional[str] = None,
    video_source: str = "0",
    window_size: int = 16,
    risk_threshold: float = 0.8,
    consecutive_windows: int = 3,
    use_cnn3d: bool = True,
    cnn3d_model_path: Optional[str] = None,
    face_aggregation: str = "mean",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> RealTimeRiskDetector:
    """
    Função auxiliar para criar detector em tempo real.
    
    Returns:
        RealTimeRiskDetector configurado
    """
    return RealTimeRiskDetector(
        multimodal_model_path=multimodal_model_path,
        video_model_path=video_model_path,
        emotion_model_path=emotion_model_path,
        video_source=video_source,
        window_size=window_size,
        risk_threshold=risk_threshold,
        consecutive_windows=consecutive_windows,
        use_cnn3d=use_cnn3d,
        cnn3d_model_path=cnn3d_model_path,
        face_aggregation=face_aggregation,
        device=device
    )

