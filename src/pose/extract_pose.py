"""
Módulo para extração de keypoints de pose de vídeos usando MediaPipe.

Este módulo:
1. Extrai keypoints de pose de cada frame de um vídeo
2. Trata múltiplas pessoas (seleciona a maior ou mais central)
3. Normaliza coordenadas para [0, 1] baseado na resolução do frame
4. Salva keypoints em formato .npy para reutilização
5. Lida com frames sem detecção de pose (interpolação ou padding)

Estrutura de dados:
- Keypoints shape: (num_frames, num_joints, 3) onde 3 = (x, y, visibility)
- MediaPipe Pose retorna 33 keypoints por pessoa

## Mudanças na API do MediaPipe

### Compatibilidade com MediaPipe 0.10+

Este módulo foi atualizado para suportar tanto a **API antiga** (`solutions`) quanto a **nova API** (`tasks`) do MediaPipe.

#### Problema Identificado

A partir do **MediaPipe 0.10.0**, a API mudou significativamente:
- **Antes (0.9.x)**: `mediapipe.solutions.pose` → `mp.solutions.pose.Pose()`
- **Depois (0.10+)**: `mediapipe.tasks.python.vision` → `PoseLandmarker.create_from_options()`

O erro `AttributeError: module 'mediapipe' has no attribute 'solutions'` ocorria porque:
- O módulo `solutions` foi removido na versão 0.10+
- A nova API usa `tasks` e requer arquivos de modelo baixados

#### Solução Implementada

O código agora:

1. **Detecta automaticamente** qual API está disponível:
   ```python
   # Tenta importar nova API (0.10+)
   try:
       from mediapipe.tasks.python.vision import PoseLandmarker
       NEW_API_AVAILABLE = True
   except ImportError:
       NEW_API_AVAILABLE = False
   
   # Tenta detectar API antiga (< 0.10)
   OLD_API_AVAILABLE = hasattr(mp, 'solutions')
   ```

2. **Usa a nova API quando disponível**:
   - Baixa automaticamente o modelo necessário do repositório oficial do MediaPipe
   - Salva o modelo em `%TEMP%/mediapipe_models/` para reutilização
   - Configura `PoseLandmarker` com as opções apropriadas

3. **Faz fallback para API antiga** se necessário:
   - Se a nova API falhar e a antiga estiver disponível, usa automaticamente
   - Emite um aviso informativo ao usuário

#### Modelos Suportados

A nova API suporta três níveis de complexidade:
- **0 (Lite)**: `pose_landmarker_lite.task` - Mais rápido, menos preciso
- **1 (Full)**: `pose_landmarker_full.task` - Balanceado (padrão)
- **2 (Heavy)**: `pose_landmarker_heavy.task` - Mais lento, mais preciso

Os modelos são baixados automaticamente na primeira execução.

#### Exemplo de Uso

```python
# Funciona com ambas as versões do MediaPipe
extractor = PoseExtractor(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1  # 0=lite, 1=full, 2=heavy
)

keypoints = extractor.extract_from_video(video_path, num_frames=16)
```

#### Notas Importantes

- **Primeira execução**: O modelo será baixado automaticamente (~5-10 MB)
- **Execuções subsequentes**: O modelo baixado é reutilizado
- **Compatibilidade**: Funciona com MediaPipe 0.9.x e 0.10+
- **Performance**: A nova API pode ter pequenas diferenças de performance

#### Troubleshooting

Se encontrar problemas:

1. **Erro de download**: Verifique conexão com internet na primeira execução
2. **Erro de permissão**: Verifique permissões de escrita em `%TEMP%`
3. **Versão incompatível**: Considere fazer downgrade: `pip install mediapipe==0.9.3.1`

#### Referências

- [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)
- [MediaPipe Python API Migration](https://developers.google.com/mediapipe/solutions/guide#legacy)
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
import warnings

# Detectar qual API do MediaPipe está disponível
NEW_API_AVAILABLE = False
OLD_API_AVAILABLE = False

# Tentar nova API (MediaPipe 0.10+)
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
    from mediapipe.tasks.python.core import base_options
    from mediapipe.tasks.python.vision.core import image as mp_image_module
    NEW_API_AVAILABLE = True
except ImportError:
    NEW_API_AVAILABLE = False

# Tentar API antiga (MediaPipe < 0.10)
try:
    OLD_API_AVAILABLE = hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose')
except:
    OLD_API_AVAILABLE = False


class PoseExtractor:
    """
    Classe para extrair keypoints de pose de vídeos usando MediaPipe.
    
    MediaPipe Pose detecta 33 keypoints por pessoa:
    - 0-10: Face
    - 11-16: Torso superior
    - 17-22: Braços
    - 23-28: Torso inferior
    - 29-32: Pernas
    
    Suporta tanto a API antiga (solutions) quanto a nova (tasks).
    """
    
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        """
        Inicializa o extrator de pose.
        
        Args:
            min_detection_confidence: Confiança mínima para detecção inicial
            min_tracking_confidence: Confiança mínima para rastreamento
            model_complexity: Complexidade do modelo (0, 1 ou 2)
                             0 = mais rápido, 2 = mais preciso
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self.use_new_api = NEW_API_AVAILABLE
        
        if self.use_new_api:
            # Nova API (MediaPipe 0.10+)
            self._init_new_api()
        elif OLD_API_AVAILABLE:
            # API antiga (MediaPipe < 0.10)
            self._init_old_api()
        else:
            raise RuntimeError(
                "MediaPipe não está instalado corretamente. "
                "Instale com: pip install mediapipe"
            )
        
        # MediaPipe Pose tem 33 keypoints
        self.num_joints = 33
    
    def _init_new_api(self):
        """Inicializa usando a nova API do MediaPipe (tasks)."""
        try:
            import os
            import urllib.request
            
            # A nova API do MediaPipe requer um arquivo de modelo
            # URLs dos modelos do MediaPipe
            model_urls = {
                0: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
                1: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
                2: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
            }
            
            # Nome do modelo baseado na complexidade
            model_names = {
                0: 'pose_landmarker_lite.task',
                1: 'pose_landmarker_full.task',
                2: 'pose_landmarker_heavy.task'
            }
            
            model_name = model_names.get(self.model_complexity, 'pose_landmarker_full.task')
            model_url = model_urls.get(self.model_complexity, model_urls[1])
            
            # Diretório para salvar modelos
            import tempfile
            model_dir = os.path.join(tempfile.gettempdir(), 'mediapipe_models')
            os.makedirs(model_dir, exist_ok=True)
            model_asset_path = os.path.join(model_dir, model_name)
            
            # Baixar modelo se não existir
            if not os.path.exists(model_asset_path):
                print(f"Baixando modelo do MediaPipe: {model_name}...")
                try:
                    urllib.request.urlretrieve(model_url, model_asset_path)
                    print(f"Modelo baixado com sucesso: {model_asset_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Erro ao baixar modelo do MediaPipe: {e}\n"
                        f"URL: {model_url}\n"
                        "Verifique sua conexão com a internet ou faça download manual do modelo."
                    )
            
            # Configurar opções base
            base_opts = base_options.BaseOptions(
                model_asset_path=model_asset_path
            )
            
            # Configurar opções do PoseLandmarker
            options = PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_pose_presence_confidence=self.min_tracking_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_segmentation_masks=False
            )
            
            # Criar o landmarker
            self.pose_landmarker = PoseLandmarker.create_from_options(options)
            self.frame_timestamp_ms = 0
            
        except Exception as e:
            # Se falhar, tentar usar API antiga se disponível
            if OLD_API_AVAILABLE:
                warnings.warn(
                    f"Erro ao inicializar nova API do MediaPipe: {e}\n"
                    "Fazendo fallback para API antiga (solutions)."
                )
                self.use_new_api = False
                self._init_old_api()
            else:
                raise RuntimeError(
                    f"Erro ao inicializar MediaPipe: {e}\n"
                    "Tente fazer downgrade: pip install mediapipe==0.9.3.1"
                )
    
    def _init_old_api(self):
        """Inicializa usando a API antiga do MediaPipe (solutions)."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # False para vídeos (tracking)
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def extract_keypoints_from_frame(
        self,
        frame: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Extrai keypoints de pose de um único frame.
        
        Args:
            frame: Frame RGB (H, W, 3) como numpy array
        
        Returns:
            Array de shape (num_joints, 3) com (x, y, visibility) ou None se não detectar
            Coordenadas x, y são normalizadas [0, 1] baseadas na resolução do frame
        """
        # MediaPipe espera RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Converter BGR para RGB se necessário (OpenCV usa BGR)
            if frame.dtype == np.uint8:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
        else:
            frame_rgb = frame
        
        if self.use_new_api:
            # Nova API (MediaPipe 0.10+)
            return self._extract_keypoints_new_api(frame_rgb)
        else:
            # API antiga (MediaPipe < 0.10)
            return self._extract_keypoints_old_api(frame_rgb)
    
    def _extract_keypoints_new_api(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extrai keypoints usando a nova API do MediaPipe."""
        try:
            # Converter numpy array para MPImage
            # A nova API espera ImageFormat como enum
            mp_image = mp_image_module.Image(
                image_format=mp_image_module.ImageFormat.SRGB, 
                data=frame_rgb
            )
            
            # Processar frame (timestamp em milissegundos)
            detection_result = self.pose_landmarker.detect_for_video(
                mp_image, self.frame_timestamp_ms
            )
            
            # Atualizar timestamp para próximo frame (assumindo ~30 FPS)
            self.frame_timestamp_ms += 33
            
            # Se não detectar pose, retornar None
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None
            
            # Pegar primeira pessoa detectada (pode ter múltiplas)
            landmarks = detection_result.pose_landmarks[0]
            keypoints = np.zeros((self.num_joints, 3))
            
            # A nova API retorna landmarks como lista de objetos PoseLandmark
            for i, landmark in enumerate(landmarks):
                # x, y são normalizados [0, 1] pelo MediaPipe
                # visibility indica confiança da detecção
                keypoints[i] = [landmark.x, landmark.y, landmark.visibility]
            
            return keypoints
        except Exception as e:
            warnings.warn(f"Erro ao extrair keypoints (nova API): {e}")
            return None
    
    def _extract_keypoints_old_api(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extrai keypoints usando a API antiga do MediaPipe."""
        # Processar frame
        results = self.pose.process(frame_rgb)
        
        # Se não detectar pose, retornar None
        if not results.pose_landmarks:
            return None
        
        # Extrair keypoints
        landmarks = results.pose_landmarks.landmark
        keypoints = np.zeros((self.num_joints, 3))
        
        for i, landmark in enumerate(landmarks):
            # x, y são normalizados [0, 1] pelo MediaPipe
            # visibility indica confiança da detecção
            keypoints[i] = [landmark.x, landmark.y, landmark.visibility]
        
        return keypoints
    
    def select_best_person(
        self,
        all_keypoints: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Seleciona a melhor pessoa quando múltiplas são detectadas.
        
        Estratégia: seleciona a pessoa com maior área de bounding box
        (mais visível) ou mais central no frame.
        
        Args:
            all_keypoints: Lista de arrays (num_joints, 3) para cada pessoa detectada
        
        Returns:
            Array (num_joints, 3) da melhor pessoa ou None
        """
        if len(all_keypoints) == 0:
            return None
        if len(all_keypoints) == 1:
            return all_keypoints[0]
        
        # Calcular área de bounding box para cada pessoa
        best_person = None
        max_area = 0
        
        for keypoints in all_keypoints:
            # Calcular bounding box baseado nos keypoints visíveis
            visible_mask = keypoints[:, 2] > 0.5  # visibility > 0.5
            if visible_mask.sum() == 0:
                continue
            
            visible_points = keypoints[visible_mask, :2]  # apenas x, y
            x_min, y_min = visible_points.min(axis=0)
            x_max, y_max = visible_points.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            
            if area > max_area:
                max_area = area
                best_person = keypoints
        
        return best_person if best_person is not None else all_keypoints[0]
    
    def interpolate_missing_frames(
        self,
        keypoints_sequence: List[Optional[np.ndarray]],
        method: str = "linear"
    ) -> np.ndarray:
        """
        Interpola keypoints para frames onde não houve detecção.
        
        Args:
            keypoints_sequence: Lista de keypoints (pode conter None)
            method: Método de interpolação ("linear" ou "forward_fill")
        
        Returns:
            Array (num_frames, num_joints, 3) com todos os frames preenchidos
        """
        num_frames = len(keypoints_sequence)
        
        # Converter lista para array, preenchendo None com NaN
        keypoints_array = np.full((num_frames, self.num_joints, 3), np.nan)
        
        for i, kp in enumerate(keypoints_sequence):
            if kp is not None:
                keypoints_array[i] = kp
        
        # Interpolar frames faltantes
        if method == "linear":
            # Interpolação linear ao longo do tempo
            for joint_idx in range(self.num_joints):
                for coord_idx in range(3):
                    values = keypoints_array[:, joint_idx, coord_idx]
                    # Encontrar índices válidos
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 0:
                        # Interpolar apenas se houver valores válidos
                        if valid_mask.sum() == 1:
                            # Se só um valor válido, preencher tudo com ele
                            keypoints_array[:, joint_idx, coord_idx] = values[valid_mask][0]
                        else:
                            # Interpolação linear
                            valid_indices = np.where(valid_mask)[0]
                            valid_values = values[valid_indices]
                            # Interpolar para todos os índices
                            keypoints_array[:, joint_idx, coord_idx] = np.interp(
                                np.arange(num_frames),
                                valid_indices,
                                valid_values
                            )
        elif method == "forward_fill":
            # Preencher com último valor válido
            for joint_idx in range(self.num_joints):
                for coord_idx in range(3):
                    values = keypoints_array[:, joint_idx, coord_idx]
                    # Forward fill
                    last_valid = None
                    for i in range(num_frames):
                        if not np.isnan(values[i]):
                            last_valid = values[i]
                        elif last_valid is not None:
                            keypoints_array[i, joint_idx, coord_idx] = last_valid
        
        # Se ainda houver NaN, preencher com zeros (último recurso)
        keypoints_array = np.nan_to_num(keypoints_array, nan=0.0)
        
        return keypoints_array
    
    def extract_from_video(
        self,
        video_path: Path,
        num_frames: Optional[int] = None,
        normalize_coords: bool = True,
        interpolate_missing: bool = True
    ) -> Optional[np.ndarray]:
        """
        Extrai keypoints de pose de um vídeo completo.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            num_frames: Número de frames a processar (None = todos)
            normalize_coords: Se True, mantém coordenadas normalizadas [0, 1]
            interpolate_missing: Se True, interpola frames sem detecção
        
        Returns:
            Array (num_frames, num_joints, 3) com keypoints ou None se erro
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return None
        
        # Obter informações do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Determinar frames a processar
        if num_frames is None:
            frame_indices = list(range(total_frames))
        else:
            if total_frames < num_frames:
                # Se vídeo tem menos frames, processar todos
                frame_indices = list(range(total_frames))
            else:
                # Selecionar frames uniformemente espaçados
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
        
        keypoints_sequence = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                keypoints_sequence.append(None)
                continue
            
            # Extrair keypoints
            keypoints = self.extract_keypoints_from_frame(frame)
            keypoints_sequence.append(keypoints)
        
        cap.release()
        
        # Interpolar frames faltantes se solicitado
        if interpolate_missing:
            keypoints_array = self.interpolate_missing_frames(
                keypoints_sequence,
                method="linear"
            )
        else:
            # Preencher None com zeros
            num_frames_actual = len(keypoints_sequence)
            keypoints_array = np.zeros((num_frames_actual, self.num_joints, 3))
            for i, kp in enumerate(keypoints_sequence):
                if kp is not None:
                    keypoints_array[i] = kp
        
        return keypoints_array
    
    def __del__(self):
        """Libera recursos do MediaPipe."""
        if self.use_new_api:
            if hasattr(self, 'pose_landmarker'):
                try:
                    self.pose_landmarker.close()
                except:
                    pass
        else:
            if hasattr(self, 'pose'):
                try:
                    self.pose.close()
                except:
                    pass


def extract_pose_from_video(
    video_path: Path,
    num_frames: Optional[int] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    normalize_coords: bool = True,
    interpolate_missing: bool = True
) -> Optional[np.ndarray]:
    """
    Função auxiliar para extrair keypoints de pose de um vídeo.
    
    Args:
        video_path: Caminho para o arquivo de vídeo
        num_frames: Número de frames a processar (None = todos)
        min_detection_confidence: Confiança mínima para detecção
        min_tracking_confidence: Confiança mínima para rastreamento
        model_complexity: Complexidade do modelo (0, 1 ou 2)
        normalize_coords: Se True, mantém coordenadas normalizadas [0, 1]
        interpolate_missing: Se True, interpola frames sem detecção
    
    Returns:
        Array (num_frames, num_joints, 3) com keypoints ou None se erro
    """
    extractor = PoseExtractor(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=model_complexity
    )
    
    return extractor.extract_from_video(
        video_path=video_path,
        num_frames=num_frames,
        normalize_coords=normalize_coords,
        interpolate_missing=interpolate_missing
    )


def process_videos_for_pose(
    input_dir: Path,
    output_dir: Path,
    num_frames: Optional[int] = None,
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mov"),
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1,
    normalize_coords: bool = True,
    interpolate_missing: bool = True
):
    """
    Processa todos os vídeos de um diretório e salva keypoints de pose.
    
    Args:
        input_dir: Diretório com vídeos de entrada
        output_dir: Diretório de saída para salvar arquivos .npy
        num_frames: Número de frames a processar por vídeo (None = todos)
        video_extensions: Extensões de vídeo aceitas
        min_detection_confidence: Confiança mínima para detecção
        min_tracking_confidence: Confiança mínima para rastreamento
        model_complexity: Complexidade do modelo (0, 1 ou 2)
        normalize_coords: Se True, mantém coordenadas normalizadas [0, 1]
        interpolate_missing: Se True, interpola frames sem detecção
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
    extractor = PoseExtractor(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_complexity=model_complexity
    )
    
    success_count = 0
    error_count = 0
    
    for video_path in tqdm(video_files, desc=f"Extraindo pose de {input_dir.name}"):
        try:
            # Extrair keypoints
            keypoints = extractor.extract_from_video(
                video_path=video_path,
                num_frames=num_frames,
                normalize_coords=normalize_coords,
                interpolate_missing=interpolate_missing
            )
            
            if keypoints is None:
                print(f"  Aviso: Não foi possível extrair pose de {video_path.name}")
                error_count += 1
                continue
            
            # Salvar como .npy
            video_id = video_path.stem
            output_path = output_dir / f"{video_id}.npy"
            np.save(output_path, keypoints)
            success_count += 1
            
        except Exception as e:
            print(f"  Erro ao processar {video_path.name}: {str(e)}")
            error_count += 1
    
    print(f"\nProcessamento concluído:")
    print(f"  - Sucesso: {success_count}")
    print(f"  - Erros: {error_count}")
    print(f"  - Total: {len(video_files)}")


def process_dataset_for_pose(
    dataset_root: str,
    output_root: str,
    dataset_name: str,  # "rwf2000"
    num_frames: Optional[int] = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_complexity: int = 1
):
    """
    Processa um dataset completo para extrair pose.

    Estrutura esperada:
    - RWF-2000: dataset/RWF-2000/train/<Fight|NonFight>/<video>.avi

    Args:
        dataset_root: Raiz do dataset (ex: "dataset/RWF-2000")
        output_root: Raiz de saída (ex: "data/pose")
        dataset_name: Nome do dataset ("rwf2000")
        num_frames: Número de frames a processar por vídeo
        min_detection_confidence: Confiança mínima para detecção
        min_tracking_confidence: Confiança mínima para rastreamento
        model_complexity: Complexidade do modelo
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)
    
    if dataset_name.lower() != "rwf2000":
        raise ValueError(f"Dataset não suportado: {dataset_name}. Use 'rwf2000'")

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
            process_videos_for_pose(
                input_dir=fight_dir,
                output_dir=output_fight_dir,
                num_frames=num_frames,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity
            )
        
        # Processar NonFight (non_violent)
        nonfight_dir = split_dir / "NonFight"
        if nonfight_dir.exists():
            output_nonfight_dir = output_path / "rwf2000" / split_name / "non_violent"
            print(f"\nProcessando RWF-2000 - {split_name}/NonFight...")
            process_videos_for_pose(
                input_dir=nonfight_dir,
                output_dir=output_nonfight_dir,
                num_frames=num_frames,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                model_complexity=model_complexity
            )
    
    print("\nProcessamento do dataset concluído!")

