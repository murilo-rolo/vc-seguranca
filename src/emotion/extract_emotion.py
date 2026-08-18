"""
Módulo para extração de emoções faciais de vídeos usando modelos FER.

Este módulo:
1. Detecta faces em frames de vídeo
2. Extrai e pré-processa faces detectadas
3. Extrai embeddings de emoção usando modelo FER
4. Agrega TODAS as faces detectadas por frame (mean/max) em um único embedding
5. Agrega embeddings ao longo do tempo
6. Salva vetores de emoção em formato .npy

Estrutura de dados:
- Emotion vectors shape: (num_frames, 128) onde cada linha é um vetor de embeddings (128-d)
- Frames sem face detectada usam o embedding neutro pré-computado
- Agregação por frame: média ou max pooling dos embeddings das faces (--face_aggregation)
- Agregação temporal: média ou max pooling dos embeddings por frame
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
import warnings

# Tentar importar detectores de face
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    warnings.warn("MTCNN não disponível. Usando Haar Cascade como fallback.")

try:
    from retinaface import RetinaFace
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False

from ..models.emotion_cnn import EmotionNet, create_emotion_model
from .neutral_embedding import compute_neutral_embedding


class FaceDetector:
    """
    Classe para detecção de faces em frames.
    
    Suporta múltiplos métodos:
    - MTCNN (recomendado, mais preciso)
    - RetinaFace (alternativa moderna)
    - Haar Cascade (fallback, mais lento mas sempre disponível)
    """
    
    def __init__(
        self,
        method: str = "mtcnn",
        min_face_size: int = 40,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa o detector de faces.
        
        Args:
            method: Método de detecção ("mtcnn", "retinaface", "haar")
            min_face_size: Tamanho mínimo da face em pixels
            device: Device para processamento (GPU recomendado)
        """
        self.method = method.lower()
        self.min_face_size = min_face_size
        self.device = device
        
        if self.method == "mtcnn":
            if HAS_MTCNN:
                self.detector = MTCNN(
                    image_size=224,
                    margin=0,
                    min_face_size=min_face_size,
                    thresholds=[0.6, 0.7, 0.7],
                    factor=0.709,
                    post_process=False,
                    device=device
                )
            else:
                warnings.warn("MTCNN não disponível. Usando Haar Cascade.")
                self.method = "haar"
                self._init_haar()
        elif self.method == "retinaface":
            if HAS_RETINAFACE:
                self.detector = None  # RetinaFace é chamado como função
            else:
                warnings.warn("RetinaFace não disponível. Usando Haar Cascade.")
                self.method = "haar"
                self._init_haar()
        else:  # haar
            self._init_haar()
    
    def _init_haar(self):
        """Inicializa detector Haar Cascade."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Não foi possível carregar Haar Cascade.")
    
    def detect_faces(
        self,
        frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detecta faces em um frame.
        
        Args:
            frame: Frame RGB (H, W, 3) como numpy array
        
        Returns:
            Lista de bounding boxes (x, y, w, h) das faces detectadas
        """
        if self.method == "mtcnn":
            return self._detect_mtcnn(frame)
        elif self.method == "retinaface":
            return self._detect_retinaface(frame)
        else:  # haar
            return self._detect_haar(frame)
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta faces usando MTCNN."""
        boxes, _ = self.detector.detect(frame)
        
        if boxes is None or len(boxes) == 0:
            return []
        
        # Converter para formato (x, y, w, h)
        faces = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1
            
            # Filtrar faces muito pequenas
            if w >= self.min_face_size and h >= self.min_face_size:
                faces.append((x1, y1, w, h))
        
        return faces
    
    def _detect_retinaface(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta faces usando RetinaFace."""
        detections = RetinaFace.detect_faces(frame)
        
        if not detections:
            return []
        
        faces = []
        for key, detection in detections.items():
            facial_area = detection['facial_area']
            x1, y1, x2, y2 = facial_area
            
            w = x2 - x1
            h = y2 - y1
            
            if w >= self.min_face_size and h >= self.min_face_size:
                faces.append((x1, y1, w, h))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detecta faces usando Haar Cascade."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def select_best_face(
        self,
        faces: List[Tuple[int, int, int, int]],
        frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Seleciona a melhor face quando múltiplas são detectadas.
        
        Estratégia: seleciona a face com maior área (mais visível).
        Alternativa: poderia selecionar a mais central.
        
        Args:
            faces: Lista de bounding boxes (x, y, w, h)
            frame_shape: Shape do frame (H, W)
        
        Returns:
            Bounding box da melhor face ou None
        """
        if len(faces) == 0:
            return None
        if len(faces) == 1:
            return faces[0]
        
        # Selecionar face com maior área
        best_face = max(faces, key=lambda f: f[2] * f[3])
        return best_face
    
    def extract_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (224, 224)
    ) -> Optional[np.ndarray]:
        """
        Extrai e redimensiona face do frame.
        
        Args:
            frame: Frame RGB (H, W, 3)
            bbox: Bounding box (x, y, w, h)
            target_size: Tamanho de saída (H, W)
        
        Returns:
            Face extraída e redimensionada (H, W, 3) ou None se erro
        """
        x, y, w, h = bbox
        
        # Garantir que bbox está dentro do frame
        H, W = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, W - x)
        h = min(h, H - y)
        
        if w <= 0 or h <= 0:
            return None
        
        # Extrair face
        face = frame[y:y+h, x:x+w]
        
        # Redimensionar
        face = cv2.resize(face, target_size)
        
        return face


class EmotionExtractor:
    """
    Classe para extração de emoções de vídeos.
    
    Pipeline:
    1. Detecta faces em cada frame
    2. Extrai e pré-processa faces
    3. Extrai embeddings de emoção (extract_features, 128-d) usando modelo FER
    4. Agrega embeddings ao longo do tempo
    """
    
    def __init__(
        self,
        model: EmotionNet,
        face_detector: FaceDetector,
        aggregation: str = "mean",  # agregação temporal (compatibilidade)
        face_aggregation: str = "mean",  # "mean" ou "max" — agregação das faces por frame
        batch_size: int = 32,
        neutral_val_dir: Optional[str] = None,
        neutral_cache_path: Optional[str] = None,
    ):
        """
        Inicializa o extrator de emoções.
        
        Args:
            model: Modelo EmotionNet pré-treinado
            face_detector: Detector de faces
            aggregation: Método de agregação temporal ("mean" ou "max")
            face_aggregation: Método de agregação de TODAS as faces detectadas
                em um frame ("mean" = centroide semântico; "max" = expressão mais
                saliente). Nenhuma face detectada -> embedding neutro (AD-004).
            batch_size: Tamanho do batch de faces processadas
            neutral_val_dir: Diretório com embeddings .npy do val (para o embedding neutro)
            neutral_cache_path: Caminho opcional do cache do embedding neutro
        """
        if face_aggregation not in ("mean", "max"):
            raise ValueError(
                f"face_aggregation deve ser 'mean' ou 'max', recebido: {face_aggregation}"
            )
        self.model = model
        self.face_detector = face_detector
        self.aggregation = aggregation
        self.face_aggregation = face_aggregation
        self.device = next(model.parameters()).device
        self.num_emotions = model.num_emotions
        self.batch_size = batch_size
        self.neutral_embedding = compute_neutral_embedding(
            model=model,
            val_dir=neutral_val_dir,
            cache_path=neutral_cache_path,
            device=str(self.device),
        )
    
    def extract_from_frame(
        self,
        frame: np.ndarray,
        face_aggregation: Optional[str] = None
    ) -> np.ndarray:
        """
        Extrai vetor de embeddings (128,) de um único frame agregando TODAS as faces.

        Quando N>=1 faces são detectadas, retorna a agregação (`mean` ou `max`)
        dos N embeddings de `extract_features`. Quando nenhuma face é detectada
        (ou nenhuma face pôde ser extraída), retorna o embedding neutro (AD-004).

        Args:
            frame: Frame RGB (H, W, 3)
            face_aggregation: Estratégia de agregação das faces ("mean" ou "max").
                None = usa self.face_aggregation.

        Returns:
            Vetor de embeddings (128,)
        """
        # Detectar faces
        faces = self.face_detector.detect_faces(frame)
        
        if len(faces) == 0:
            return self.neutral_embedding.copy()
        
        agg = face_aggregation if face_aggregation is not None else self.face_aggregation
        
        # Extrair o embedding de CADA face detectada
        embeddings: List[np.ndarray] = []
        
        for bbox in faces:
            face = self.face_detector.extract_face(frame, bbox, target_size=(224, 224))
            if face is None:
                continue
            
            # Pré-processar face para o modelo
            # Converter para tensor e normalizar
            face_tensor = torch.from_numpy(face).float()
            face_tensor = face_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            face_tensor = face_tensor / 255.0
            
            # Normalizar com ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            face_tensor = (face_tensor - mean) / std
            
            # Adicionar dimensão de batch
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # Extrair embedding da penúltima camada
            with torch.no_grad():
                embedding = self.model.extract_features(face_tensor)
                embeddings.append(embedding.cpu().numpy()[0])  # (128,)
        
        if len(embeddings) == 0:
            return self.neutral_embedding.copy()
        
        # Agregar todas as faces (mean = centroide; max = elemento a elemento)
        emb_stack = np.stack(embeddings, axis=0)  # (N, 128)
        
        if agg == "mean":
            return emb_stack.mean(axis=0)
        if agg == "max":
            return emb_stack.max(axis=0)
        raise ValueError(f"face_aggregation inválido: {agg} (use 'mean' ou 'max')")
    
    def extract_from_video(
        self,
        video_path: Path,
        num_frames: Optional[int] = None,
        aggregation: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """
        Extrai emoções de um vídeo completo.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            num_frames: Número de frames a processar (None = todos)
            aggregation: Método de agregação temporal (None = usar padrão)
        
        Returns:
            Array (num_frames, 128) com vetores de embeddings ou None se erro
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return None
        
        # Obter informações do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Determinar frames a processar
        if num_frames is None:
            frame_indices = list(range(total_frames))
        else:
            if total_frames < num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        num_selected_frames = len(frame_indices)
        
        # Array de saída (num_frames, 128)
        emotion_dim = self.neutral_embedding.shape[0]
        emotion_array = np.zeros((num_selected_frames, emotion_dim), dtype=np.float32)
        
        # Embedding neutro pré-computado para reutilização (frames sem face)
        neutral_vec = self.neutral_embedding
        
        # Processar em batches de frames para aproveitar batching em MTCNN e Emotion CNN
        batch_size = max(1, self.batch_size)
        
        for start in range(0, num_selected_frames, batch_size):
            end = min(start + batch_size, num_selected_frames)
            batch_indices = frame_indices[start:end]
            
            # Ler frames do vídeo
            frames_rgb: List[np.ndarray] = []
            frame_pos_map: List[int] = []  # posição no emotion_array para cada frame válido
            
            for local_idx, frame_idx in enumerate(batch_indices):
                pos = start + local_idx  # índice no emotion_array
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    # Se não conseguir ler, usar embedding neutro
                    emotion_array[pos] = neutral_vec
                    continue
                
                # Converter BGR para RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)
                frame_pos_map.append(pos)
            
            if not frames_rgb:
                continue
            
            # Detectar faces em batch quando possível (MTCNN), caso contrário por frame
            faces_per_frame: List[List[Tuple[int, int, int, int]]] = []
            
            if (
                self.face_detector.method == "mtcnn"
                and HAS_MTCNN
                and hasattr(self.face_detector, "detector")
                and self.face_detector.detector is not None
            ):
                # MTCNN aceita lista de imagens em batch
                boxes_list, _ = self.face_detector.detector.detect(frames_rgb)
                
                for frame_rgb, boxes in zip(frames_rgb, boxes_list):
                    if boxes is None or len(boxes) == 0:
                        faces_per_frame.append([])
                        continue
                    
                    valid_faces: List[Tuple[int, int, int, int]] = []
                    for box in boxes:
                        x1, y1, x2, y2 = box.astype(int)
                        w = x2 - x1
                        h = y2 - y1
                        if w >= self.face_detector.min_face_size and h >= self.face_detector.min_face_size:
                            valid_faces.append((x1, y1, w, h))
                    faces_per_frame.append(valid_faces)
            else:
                # Fallback: usar detecção por frame (RetinaFace / Haar ou MTCNN sem batching)
                for frame_rgb in frames_rgb:
                    faces = self.face_detector.detect_faces(frame_rgb)
                    faces_per_frame.append(faces)
            
            # Preparar batch com TODAS as faces de todos os frames do batch
            face_tensors: List[torch.Tensor] = []
            face_tensor_frame: List[int] = []  # posição em emotion_array por face
            
            # Valores de normalização (no device da CPU; serão movidos com o tensor)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            
            # Embeddings por frame (cada frame pode ter N faces)
            frame_embeddings: Dict[int, List[np.ndarray]] = {pos: [] for pos in frame_pos_map}
            
            for frame_rgb, faces, pos in zip(frames_rgb, faces_per_frame, frame_pos_map):
                for bbox in faces:
                    face = self.face_detector.extract_face(frame_rgb, bbox, target_size=(224, 224))
                    if face is None:
                        continue
                    
                    # Pré-processar face para o modelo (igual a extract_from_frame)
                    face_tensor = torch.from_numpy(face).float()
                    face_tensor = face_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
                    face_tensor = face_tensor / 255.0
                    face_tensor = (face_tensor - mean) / std
                    face_tensors.append(face_tensor)
                    face_tensor_frame.append(pos)
            
            if face_tensors:
                # Criar batch tensor e enviar para device
                batch_tensor = torch.stack(face_tensors, dim=0).to(self.device)
                
                with torch.no_grad():
                    emb_batch = self.model.extract_features(batch_tensor)
                    emb_batch = emb_batch.cpu().numpy()  # (B, 128)
                
                # Distribuir embeddings por frame (um frame pode ter várias faces)
                for emb, pos in zip(emb_batch, face_tensor_frame):
                    frame_embeddings[pos].append(emb)
            
            # Agregar por frame (mean/max sobre as faces) — linhas independentes
            for pos in frame_pos_map:
                embs = frame_embeddings[pos]
                if len(embs) == 0:
                    emotion_array[pos] = neutral_vec
                    continue
                emb_stack = np.stack(embs, axis=0)  # (N, 128)
                if self.face_aggregation == "mean":
                    emotion_array[pos] = emb_stack.mean(axis=0)
                else:
                    emotion_array[pos] = emb_stack.max(axis=0)
        
        cap.release()
        
        # Agregar se solicitado (opcional - pode ser feito depois)
        agg_method = aggregation if aggregation is not None else self.aggregation
        
        if agg_method == "mean":
            # Já temos média por frame, mas podemos agregar temporalmente
            # Por enquanto, retornamos todos os frames
            pass
        elif agg_method == "max":
            # Max pooling temporal (opcional)
            pass
        
        return emotion_array


def extract_emotions_from_video(
    video_path: Path,
    model: EmotionNet,
    num_frames: Optional[int] = None,
    face_detector_method: str = "mtcnn",
    aggregation: str = "mean",
    face_aggregation: str = "mean"
) -> Optional[np.ndarray]:
    """
    Função auxiliar para extrair emoções de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        model: Modelo EmotionNet pré-treinado
        num_frames: Número de frames a processar
        face_detector_method: Método de detecção de faces
        aggregation: Método de agregação temporal
        face_aggregation: Agregação das faces por frame ("mean" ou "max")
    
    Returns:
        Array (num_frames, 128) com vetores de embeddings
    """
    face_detector = FaceDetector(method=face_detector_method)
    extractor = EmotionExtractor(
        model, face_detector,
        aggregation=aggregation,
        face_aggregation=face_aggregation,
        batch_size=256
    )
    
    return extractor.extract_from_video(video_path, num_frames=num_frames)


def process_videos_for_emotion(
    input_dir: Path,
    output_dir: Path,
    model: EmotionNet,
    num_frames: Optional[int] = None,
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mov"),
    face_detector_method: str = "mtcnn",
    aggregation: str = "mean",
    face_aggregation: str = "mean"
):
    """
    Processa todos os vídeos de um diretório e salva vetores de emoção.
    
    Args:
        input_dir: Diretório com vídeos de entrada
        output_dir: Diretório de saída para salvar arquivos .npy
        model: Modelo EmotionNet pré-treinado
        num_frames: Número de frames a processar por vídeo
        video_extensions: Extensões de vídeo aceitas
        face_detector_method: Método de detecção de faces
        aggregation: Método de agregação temporal
        face_aggregation: Agregação das faces por frame ("mean" ou "max")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Listar todos os vídeos
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f"*{ext}")))
        video_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if len(video_files) == 0:
        print(f"Nenhum vídeo encontrado em {input_dir}")
        return
    
    print(f"Processando {len(video_files)} vídeos de {input_dir.name}...")
    
    # Criar extrator uma vez (reutilizar para eficiência)
    face_detector = FaceDetector(method=face_detector_method)
    extractor = EmotionExtractor(
        model, face_detector,
        aggregation=aggregation,
        face_aggregation=face_aggregation
    )
    
    success_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc=f"Extraindo emoções de {input_dir.name}"):
        try:
            # Extrair emoções
            emotions = extractor.extract_from_video(
                video_path=video_path,
                num_frames=num_frames
            )
            
            if emotions is None:
                print(f"  Aviso: Não foi possível extrair emoções de {video_path.name}")
                error_count += 1
                continue
            
            # Salvar como .npy
            video_id = video_path.stem
            output_path = output_dir / f"{video_id}.npy"
            np.save(output_path, emotions)
            success_count += 1
            
        except Exception as e:
            print(f"  Erro ao processar {video_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessamento concluído:")
    print(f"  - Sucesso: {success_count}")
    print(f"  - Erros: {error_count}")
    print(f"  - Total: {len(video_files)}")


def process_dataset_for_emotion(
    dataset_root: str,
    output_root: str,
    model: EmotionNet,
    dataset_name: str,  # "rwf2000"
    num_frames: Optional[int] = None,
    face_detector_method: str = "mtcnn",
    aggregation: str = "mean",
    face_aggregation: str = "mean"
):
    """
    Processa um dataset completo (RWF-2000) para extrair emoções.
    
    Estrutura esperada:
    - RWF-2000: dataset/RWF-2000/train/<Fight|NonFight>/<video>.avi
    
    Args:
        dataset_root: Raiz do dataset (ex: "dataset/RWF-2000")
        output_root: Raiz de saída (ex: "data/emotion")
        model: Modelo EmotionNet pré-treinado
        dataset_name: Nome do dataset ("rwf2000")
        num_frames: Número de frames a processar por vídeo
        face_detector_method: Método de detecção de faces
        aggregation: Método de agregação temporal
        face_aggregation: Agregação das faces por frame ("mean" ou "max")
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    
    if dataset_name.lower() == "rwf2000":
        # Processar RWF-2000: estrutura Fight/NonFight
        train_dir = dataset_path / "train"
        val_dir = dataset_path / "val"
        
        for split_dir, split_name in [(train_dir, "train"), (val_dir, "val")]:
            if not split_dir.exists():
                print(f"Diretório não encontrado: {split_dir}")
                continue
            
            # Processar Fight (violent)
            fight_dir = split_dir / "Fight"
            if fight_dir.exists():
                output_fight_dir = output_path / "rwf2000" / split_name / "violent"
                print(f"\nProcessando RWF-2000 - {split_name}/Fight...")
                process_videos_for_emotion(
                    input_dir=fight_dir,
                    output_dir=output_fight_dir,
                    model=model,
                    num_frames=num_frames,
                    face_detector_method=face_detector_method,
                    aggregation=aggregation,
                    face_aggregation=face_aggregation
                )
            
            # Processar NonFight (non_violent)
            nonfight_dir = split_dir / "NonFight"
            if nonfight_dir.exists():
                output_nonfight_dir = output_path / "rwf2000" / split_name / "non_violent"
                print(f"\nProcessando RWF-2000 - {split_name}/NonFight...")
                process_videos_for_emotion(
                    input_dir=nonfight_dir,
                    output_dir=output_nonfight_dir,
                    model=model,
                    num_frames=num_frames,
                    face_detector_method=face_detector_method,
                    aggregation=aggregation,
                    face_aggregation=face_aggregation
                )
    
    else:
        raise ValueError(f"Dataset não suportado: {dataset_name}. Use 'rwf2000'")
    
    print("\nProcessamento do dataset concluído!")

