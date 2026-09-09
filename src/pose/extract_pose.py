"""
Módulo para extração de keypoints de pose de vídeos usando Ultralytics YOLO26.

Este módulo:
1. Extrai keypoints de pose de cada frame de um vídeo
2. Trata múltiplas pessoas (seleciona a maior ou mais central)
3. Retorna coordenadas pixel [x, y, confidence] por keypoint
4. Salva keypoints em formato .npy para reutilização
5. Lida com frames sem detecção de pose (interpolação ou padding)

Estrutura de dados:
- Keypoints shape: (num_frames, num_joints, 3) onde 3 = (x, y, confidence)
- YOLO26 pose detecta 17 keypoints por pessoa (COCO)

## Mudanças em Relação ao MediaPipe

Este módulo substitui o MediaPipe pelo Ultralytics YOLO26:
- 17 keypoints em vez de 33 (COCO keypoints)
- Coordenadas em pixel em vez de normalizadas [0, 1]
- Confiança por keypoint em vez de visibility
- Modelos YOLO26-pose: nano, small, medium, large, extra-large

Modelos YOLO26 suportados:
- **0 (nano)**: `yolo26n-pose` - Mais rápido, menos preciso
- **1 (small)**: `yolo26s-pose` - Balanceado (padrão)
- **2 (medium)**: `yolo26m-pose` - Mais lento, mais preciso

Exemplo de uso:
```python
extractor = PoseExtractor(model_complexity=1)
keypoints = extractor.extract_from_video(video_path, num_frames=16)
```
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
import warnings

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class PoseExtractor:
    """
    Classe para extrair keypoints de pose de vídeos usando Ultralytics YOLO26.

    YOLO26 pose detecta 17 keypoints por pessoa (COCO dataset):
    - 0: nose
    - 1: left_eye, 2: right_eye
    - 3: left_ear, 4: right_ear
    - 5: left_shoulder, 6: right_shoulder
    - 7: left_elbow, 8: right_elbow
    - 9: left_wrist, 10: right_wrist
    - 11: left_hip, 12: right_hip
    - 13: left_knee, 14: right_knee
    - 15: left_ankle, 16: right_ankle

    Suporta modelos YOLO26n-pose, YOLO26s-pose, YOLO26m-pose.
    """

    YOLO26_MODELS = {
        0: "yolo26n-pose",
        1: "yolo26s-pose",
        2: "yolo26m-pose",
    }

    def __init__(
        self,
        model_complexity: int = 1,
        conf: float = 0.5,
        iou: float = 0.7,
    ):
        """
        Inicializa o extrator de pose com YOLO26.

        Args:
            model_complexity: Complexidade do modelo (0=nano, 1=small, 2=medium)
            conf: Confiança mínima para detecção (0.0-1.0)
            iou: IoU mínima para NMS (0.0-1.0)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError(
                "Ultralytics não está instalado. "
                "Instale com: pip install ultralytics"
            )

        self.model_complexity = model_complexity
        self.conf = conf
        self.iou = iou
        self.num_joints = 17

        model_name = self.YOLO26_MODELS.get(model_complexity, "yolo26s-pose")
        self.model = YOLO(model_name)

    def extract_keypoints_from_frame(
        self,
        frame: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Extrai keypoints de pose de um único frame.

        Args:
            frame: Frame RGB (H, W, 3) como numpy array

        Returns:
            Array de shape (num_joints, 3) com (x, y, confidence) ou None se não detectar
            Coordenadas x, y são em pixels baseadas na resolução do frame
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            if frame.dtype == np.uint8:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
        else:
            frame_bgr = frame

        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        if not results or len(results) == 0:
            return None

        result = results[0]

        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # PEGAR primeira pessoa detectada (pode ter múltiplas)
        keypoints = result.keypoints.data

        if len(keypoints) == 0:
            return None

        # keypoints shape: (num_instances, 17, 3) com (x, y, confidence)
        person_keypoints = keypoints[0]  # (17, 3)

        # Converter para array numpy (17, 3)
        result_kp = np.zeros((self.num_joints, 3))
        for i in range(self.num_joints):
            result_kp[i] = [
                person_keypoints[i, 0].item(),  # x (pixel)
                person_keypoints[i, 1].item(),  # y (pixel)
                person_keypoints[i, 2].item(),  # confidence
            ]

        return result_kp

    def select_best_person(
        self,
        all_keypoints: List[np.ndarray],
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

        best_person = None
        max_area = 0

        for keypoints in all_keypoints:
            visible_mask = keypoints[:, 2] > 0.5
            if visible_mask.sum() == 0:
                continue

            visible_points = keypoints[visible_mask, :2]
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
        method: str = "linear",
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

        keypoints_array = np.full((num_frames, self.num_joints, 3), np.nan)

        for i, kp in enumerate(keypoints_sequence):
            if kp is not None:
                keypoints_array[i] = kp

        if method == "linear":
            for joint_idx in range(self.num_joints):
                for coord_idx in range(3):
                    values = keypoints_array[:, joint_idx, coord_idx]
                    valid_mask = ~np.isnan(values)
                    if valid_mask.sum() > 0:
                        if valid_mask.sum() == 1:
                            keypoints_array[:, joint_idx, coord_idx] = values[valid_mask][0]
                        else:
                            valid_indices = np.where(valid_mask)[0]
                            valid_values = values[valid_indices]
                            keypoints_array[:, joint_idx, coord_idx] = np.interp(
                                np.arange(num_frames),
                                valid_indices,
                                valid_values,
                            )
        elif method == "forward_fill":
            for joint_idx in range(self.num_joints):
                for coord_idx in range(3):
                    values = keypoints_array[:, joint_idx, coord_idx]
                    last_valid = None
                    for i in range(num_frames):
                        if not np.isnan(values[i]):
                            last_valid = values[i]
                        elif last_valid is not None:
                            keypoints_array[i, joint_idx, coord_idx] = last_valid

        keypoints_array = np.nan_to_num(keypoints_array, nan=0.0)

        return keypoints_array

    def extract_from_video(
        self,
        video_path: Path,
        num_frames: Optional[int] = None,
        normalize_coords: bool = True,
        interpolate_missing: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Extrai keypoints de pose de um vídeo completo.

        Args:
            video_path: Caminho para o arquivo de vídeo
            num_frames: Número de frames a processar (None = todos)
            normalize_coords: Ignorado (YOLO26 retorna pixel coords)
            interpolate_missing: Se True, interpola frames sem detecção

        Returns:
            Array (num_frames, num_joints, 3) com keypoints ou None se erro
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"Erro ao abrir vídeo: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if total_frames == 0:
            cap.release()
            return None

        if num_frames is None:
            frame_indices = list(range(total_frames))
        else:
            if total_frames < num_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

        keypoints_sequence = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                keypoints_sequence.append(None)
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            keypoints = self.extract_keypoints_from_frame(frame_rgb)
            keypoints_sequence.append(keypoints)

        cap.release()

        if interpolate_missing:
            keypoints_array = self.interpolate_missing_frames(
                keypoints_sequence,
                method="linear",
            )
        else:
            num_frames_actual = len(keypoints_sequence)
            keypoints_array = np.zeros((num_frames_actual, self.num_joints, 3))
            for i, kp in enumerate(keypoints_sequence):
                if kp is not None:
                    keypoints_array[i] = kp

        return keypoints_array

    def __del__(self):
        """Libera recursos do modelo."""
        if hasattr(self, 'model'):
            try:
                del self.model
            except:
                pass


def extract_pose_from_video(
    video_path: Path,
    num_frames: Optional[int] = None,
    model_complexity: int = 1,
    conf: float = 0.5,
    iou: float = 0.7,
    normalize_coords: bool = True,
    interpolate_missing: bool = True,
) -> Optional[np.ndarray]:
    """
    Função auxiliar para extrair keypoints de pose de um vídeo.

    Args:
        video_path: Caminho para o arquivo de vídeo
        num_frames: Número de frames a processar (None = todos)
        model_complexity: Complexidade do modelo (0=nano, 1=small, 2=medium)
        conf: Confiança mínima para detecção
        iou: IoU mínima para NMS
        normalize_coords: Ignorado (YOLO26 retorna pixel coords)
        interpolate_missing: Se True, interpola frames sem detecção

    Returns:
        Array (num_frames, num_joints, 3) com keypoints ou None se erro
    """
    extractor = PoseExtractor(
        model_complexity=model_complexity,
        conf=conf,
        iou=iou,
    )

    return extractor.extract_from_video(
        video_path=video_path,
        num_frames=num_frames,
        normalize_coords=normalize_coords,
        interpolate_missing=interpolate_missing,
    )


def process_videos_for_pose(
    input_dir: Path,
    output_dir: Path,
    num_frames: Optional[int] = None,
    video_extensions: Tuple[str, ...] = (".avi", ".mp4", ".mov"),
    model_complexity: int = 1,
    conf: float = 0.5,
    iou: float = 0.7,
    normalize_coords: bool = True,
    interpolate_missing: bool = True,
):
    """
    Processa todos os vídeos de um diretório e salva keypoints de pose.

    Args:
        input_dir: Diretório com vídeos de entrada
        output_dir: Diretório de saída para salvar arquivos .npy
        num_frames: Número de frames a processar por vídeo (None = todos)
        video_extensions: Extensões de vídeo aceitas
        model_complexity: Complexidade do modelo (0, 1 ou 2)
        conf: Confiança mínima para detecção
        iou: IoU mínima para NMS
        normalize_coords: Ignorado (YOLO26 retorna pixel coords)
        interpolate_missing: Se True, interpola frames sem detecção
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = []
    for ext in video_extensions:
        video_files.extend(list(input_dir.glob(f"*{ext}")))
        video_files.extend(list(input_dir.glob(f"*{ext.upper()}")))

    if len(video_files) == 0:
        print(f"Nenhum vídeo encontrado em {input_dir}")
        return

    print(f"Processando {len(video_files)} vídeos de {input_dir.name}...")

    extractor = PoseExtractor(
        model_complexity=model_complexity,
        conf=conf,
        iou=iou,
    )

    success_count = 0
    error_count = 0

    for video_path in tqdm(video_files, desc=f"Extraindo pose de {input_dir.name}"):
        try:
            keypoints = extractor.extract_from_video(
                video_path=video_path,
                num_frames=num_frames,
                normalize_coords=normalize_coords,
                interpolate_missing=interpolate_missing,
            )

            if keypoints is None:
                print(f"  Aviso: Não foi possível extrair pose de {video_path.name}")
                error_count += 1
                continue

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
    dataset_name: str,
    num_frames: Optional[int] = None,
    model_complexity: int = 1,
    conf: float = 0.5,
    iou: float = 0.7,
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
        model_complexity: Complexidade do modelo (0, 1 ou 2)
        conf: Confiança mínima para detecção
        iou: IoU mínima para NMS
    """
    dataset_path = Path(dataset_root)
    output_path = Path(output_root)

    if dataset_name.lower() != "rwf2000":
        raise ValueError(f"Dataset não suportado: {dataset_name}. Use 'rwf2000'")

    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    for split_dir, split_name in [(train_dir, "train"), (val_dir, "val")]:
        if not split_dir.exists():
            print(f"Diretório não encontrado: {split_dir}")
            continue

        fight_dir = split_dir / "Fight"
        if fight_dir.exists():
            output_fight_dir = output_path / "rwf2000" / split_name / "violent"
            print(f"\nProcessando RWF-2000 - {split_name}/Fight...")
            process_videos_for_pose(
                input_dir=fight_dir,
                output_dir=output_fight_dir,
                num_frames=num_frames,
                model_complexity=model_complexity,
                conf=conf,
                iou=iou,
            )

        nonfight_dir = split_dir / "NonFight"
        if nonfight_dir.exists():
            output_nonfight_dir = output_path / "rwf2000" / split_name / "non_violent"
            print(f"\nProcessando RWF-2000 - {split_name}/NonFight...")
            process_videos_for_pose(
                input_dir=nonfight_dir,
                output_dir=output_nonfight_dir,
                num_frames=num_frames,
                model_complexity=model_complexity,
                conf=conf,
                iou=iou,
            )

    print("\nProcessamento do dataset concluído!")
