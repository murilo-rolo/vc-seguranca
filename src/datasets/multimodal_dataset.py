"""
Dataset Multimodal para detecção de risco/violência.

Este módulo implementa MultimodalSurveillanceDataset que carrega:
- Video Features (frames processados)
- Pose Features (keypoints de pose)
- Emotion Features (vetores de emoção)

Todas as modalidades são carregadas de forma sincronizada para o mesmo vídeo.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import random
import numpy as np

from src import paths as p


class MultimodalSurveillanceDataset(Dataset):
    """
    Dataset multimodal para detecção de risco.
    
    Carrega três modalidades sincronizadas:
    - Video: Frames processados (T, C, H, W) ou features (T, D_v)
    - Pose: Keypoints de pose (T, num_joints, 3) ou (T, D_p)
    - Emotion: Vetores de emoção (T, num_emotions)
    
    Todas as modalidades devem estar alinhadas temporalmente (mesmo número de frames).
    """
    
    def __init__(
        self,
        # Caminhos dos dados
        video_data_root: str = None,      # Frames ou features de vídeo
        pose_data_root: str = None,       # Keypoints de pose
        emotion_data_root: str = None,    # Vetores de emoção
        
        # Configuração
        split: str = "train",
        num_frames: int = 16,
        window_size: int = 16,  # Tamanho da janela temporal (pode ser diferente de num_frames)
        
        # Modos de carregamento
        video_mode: str = "frames",  # "frames" ou "features"
        pose_mode: str = "keypoints",  # "keypoints" ou "flatten"
        
        # Transformações
        transform: Optional[Callable] = None,
        
        # Divisão de dados
        use_original_split: bool = True,
        val_test_split_ratio: float = 0.5,
        seed: int = 42,
        
        # Dataset
        dataset_name: str = "rwf2000"
    ):
        """
        Inicializa o dataset multimodal.
        
        Args:
            video_data_root: Raiz dos dados de vídeo
            pose_data_root: Raiz dos dados de pose
            emotion_data_root: Raiz dos dados de emoção
            split: "train", "val" ou "test"
            num_frames: Número de frames esperados (para vídeo)
            window_size: Tamanho da janela temporal para todas as modalidades
            video_mode: "frames" (carrega frames) ou "features" (carrega features pré-extraídas)
            pose_mode: "keypoints" (mantém shape 3D) ou "flatten" (achata para 2D)
            transform: Transformações a aplicar
            use_original_split: Se True (padrão), usa divisão original do RWF-2000.
                               Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
            val_test_split_ratio: Se use_original_split=True, divide o val original em val e test
                                 usando esta proporção (padrão: 0.5 = 50/50)
            seed: Seed para reprodutibilidade
            dataset_name: Nome do dataset ("rwf2000")
        """
        if video_data_root is None:
            video_data_root = str(p.PROCESSED_ROOT)
        if pose_data_root is None:
            pose_data_root = str(p.POSE_ROOT)
        if emotion_data_root is None:
            emotion_data_root = str(p.EMOTION_ROOT)
        self.video_data_root = Path(video_data_root)
        self.pose_data_root = Path(pose_data_root)
        self.emotion_data_root = Path(emotion_data_root)
        self.split = split
        self.num_frames = num_frames
        self.window_size = window_size
        self.video_mode = video_mode
        self.pose_mode = pose_mode
        self.transform = transform
        self.dataset_name = dataset_name.lower()
        self.use_original_split = use_original_split
        self.val_test_split_ratio = val_test_split_ratio
        self.seed = seed
        
        # Carregar lista de amostras
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(
                f"Nenhuma amostra encontrada para o split '{split}' "
                f"(video: {video_data_root}, pose: {pose_data_root}, emotion: {emotion_data_root})"
            )
        
        print(f"MultimodalSurveillanceDataset {split}: {len(self.samples)} amostras carregadas")
        print(f"  Video mode: {video_mode}")
        print(f"  Pose mode: {pose_mode}")
        print(f"  Window size: {window_size}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """
        Carrega lista de amostras (video_id, label) preservando a divisão original.
        
        Garante que todas as modalidades existem para cada vídeo.
        
        Returns:
            Lista de tuplas (video_id, label)
        """
        samples = []
        
        if self.dataset_name == "rwf2000":
            # Estrutura: data/{processed,pose,emotion}/rwf2000/{split}/{violent|non_violent}/
            
            # Usar pose como referência (geralmente tem menos vídeos processados)
            pose_base = self.pose_data_root / "rwf2000"
            
            if self.use_original_split:
                # Preservar divisão original: usar apenas o split solicitado
                if self.split == "train":
                    original_split = "train"
                elif self.split in ["val", "test"]:
                    original_split = "val"
                else:
                    raise ValueError(f"Split inválido: {self.split}")
                
                split_pose_dir = pose_base / original_split
                if not split_pose_dir.exists():
                    return []
                
                # Vídeos violentos (label=1)
                violent_pose_dir = split_pose_dir / "violent"
                if violent_pose_dir.exists():
                    for npy_file in violent_pose_dir.glob("*.npy"):
                        video_id = npy_file.stem
                        
                        # Verificar se todas as modalidades existem
                        if self._check_all_modalities_exist(video_id, original_split, 1):
                            samples.append((video_id, 1))
                
                # Vídeos não violentos (label=0)
                non_violent_pose_dir = split_pose_dir / "non_violent"
                if non_violent_pose_dir.exists():
                    for npy_file in non_violent_pose_dir.glob("*.npy"):
                        video_id = npy_file.stem
                        
                        if self._check_all_modalities_exist(video_id, original_split, 0):
                            samples.append((video_id, 0))
                
                # Se solicitamos val ou test, dividir o val original
                if original_split == "val" and len(samples) > 0:
                    random.seed(self.seed)
                    random.shuffle(samples)
                    
                    total_val = len(samples)
                    val_end = int(total_val * self.val_test_split_ratio)
                    
                    if self.split == "val":
                        samples = samples[:val_end]
                    else:  # test
                        samples = samples[val_end:]
            else:
                # Modo legado: mistura train e val (DEPRECADO)
                import warnings
                warnings.warn(
                    "⚠️  AVISO: Usando divisão aleatória pode causar DATA LEAKAGE!",
                    UserWarning
                )
                
                for split_name in ["train", "val"]:
                    split_pose_dir = pose_base / split_name
                    if not split_pose_dir.exists():
                        continue
                    
                    violent_pose_dir = split_pose_dir / "violent"
                    if violent_pose_dir.exists():
                        for npy_file in violent_pose_dir.glob("*.npy"):
                            video_id = npy_file.stem
                            if self._check_all_modalities_exist(video_id, split_name, 1):
                                samples.append((video_id, 1))
                    
                    non_violent_pose_dir = split_pose_dir / "non_violent"
                    if non_violent_pose_dir.exists():
                        for npy_file in non_violent_pose_dir.glob("*.npy"):
                            video_id = npy_file.stem
                            if self._check_all_modalities_exist(video_id, split_name, 0):
                                samples.append((video_id, 0))
                
                # Embaralhar e dividir aleatoriamente (CAUSA DATA LEAKAGE)
                random.seed(self.seed)
                random.shuffle(samples)
                
                total = len(samples)
                train_end = int(total * 0.7)
                val_end = train_end + int(total * 0.15)
                
                if self.split == "train":
                    return samples[:train_end]
                elif self.split == "val":
                    return samples[train_end:val_end]
                else:  # test
                    return samples[val_end:]
        
        return samples
    
    def _check_all_modalities_exist(
        self,
        video_id: str,
        split_name: str,
        label: int
    ) -> bool:
        """
        Verifica se todas as modalidades existem para um vídeo.
        
        Args:
            video_id: ID do vídeo
            split_name: Nome do split ("train" ou "val")
            label: Label (0 ou 1)
        
        Returns:
            True se todas as modalidades existem
        """
        class_name = "violent" if label == 1 else "non_violent"
        
        # Verificar pose
        pose_path = self.pose_data_root / "rwf2000" / split_name / class_name / f"{video_id}.npy"
        if not pose_path.exists():
            return False
        
        # Verificar emoção
        emotion_path = self.emotion_data_root / "rwf2000" / split_name / class_name / f"{video_id}.npy"
        if not emotion_path.exists():
            return False
        
        # Verificar vídeo
        if self.video_mode == "frames":
            # Procurar frame_sequence.pt
            video_path = self.video_data_root / class_name / video_id / "frame_sequence.pt"
            if not video_path.exists():
                return False
        else:  # features
            # Procurar features.pt
            video_path = self.video_data_root / class_name / video_id / "features.pt"
            if not video_path.exists():
                return False
        
        return True
    
    def _load_video_features(
        self,
        video_id: str,
        label: int
    ) -> torch.Tensor:
        """
        Carrega features de vídeo.
        
        Args:
            video_id: ID do vídeo
            label: Label (0 ou 1)
        
        Returns:
            Tensor de vídeo (T, C, H, W) ou (T, D_v)
        """
        class_name = "violent" if label == 1 else "non_violent"
        
        if self.video_mode == "frames":
            # Carregar frames processados
            video_path = self.video_data_root / class_name / video_id / "frame_sequence.pt"
            frames = torch.load(video_path, map_location='cpu')
            
            # frames shape: (num_frames, C, H, W)
            # Garantir que tem num_frames
            if frames.shape[0] < self.num_frames:
                last_frame = frames[-1:].repeat(self.num_frames - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            elif frames.shape[0] > self.num_frames:
                frames = frames[:self.num_frames]
            
            return frames
        else:
            # Carregar features pré-extraídas
            video_path = self.video_data_root / class_name / video_id / "features.pt"
            features = torch.load(video_path, map_location='cpu')
            
            # features shape: (num_frames, D_v) ou (D_v,)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)  # (1, D_v)
            
            return features
    
    def _load_pose_features(
        self,
        video_id: str,
        split_name: str,
        label: int
    ) -> torch.Tensor:
        """
        Carrega features de pose.
        
        Args:
            video_id: ID do vídeo
            split_name: Nome do split
            label: Label (0 ou 1)
        
        Returns:
            Tensor de pose (T, num_joints, 3) ou (T, D_p)
        """
        class_name = "violent" if label == 1 else "non_violent"
        pose_path = self.pose_data_root / "rwf2000" / split_name / class_name / f"{video_id}.npy"
        
        keypoints = np.load(pose_path)
        
        # keypoints shape: (num_frames, num_joints, 3)
        if self.pose_mode == "flatten":
            # Flatten para (num_frames, num_joints * 3)
            keypoints = keypoints.reshape(keypoints.shape[0], -1)
        
        # Converter para tensor
        keypoints = torch.from_numpy(keypoints).float()
        
        return keypoints
    
    def _load_emotion_features(
        self,
        video_id: str,
        split_name: str,
        label: int
    ) -> torch.Tensor:
        """
        Carrega features de emoção.
        
        Args:
            video_id: ID do vídeo
            split_name: Nome do split
            label: Label (0 ou 1)
        
        Returns:
            Tensor de emoção (T, num_emotions)
        """
        class_name = "violent" if label == 1 else "non_violent"
        emotion_path = self.emotion_data_root / "rwf2000" / split_name / class_name / f"{video_id}.npy"
        
        emotions = np.load(emotion_path)
        
        # emotions shape: (num_frames, num_emotions)
        # Converter para tensor
        emotions = torch.from_numpy(emotions).float()
        
        return emotions
    
    def _create_windows(
        self,
        video: torch.Tensor,
        pose: torch.Tensor,
        emotion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cria janelas temporais alinhadas para todas as modalidades.
        
        Args:
            video: Features de vídeo (T_v, ...)
            pose: Features de pose (T_p, ...)
            emotion: Features de emoção (T_e, ...)
        
        Returns:
            Tupla (video_window, pose_window, emotion_window) com shape (window_size, ...)
        """
        # Determinar T mínimo
        T_v = video.shape[0]
        T_p = pose.shape[0]
        T_e = emotion.shape[0]
        T_min = min(T_v, T_p, T_e)
        
        # Truncar para T_min
        if T_v > T_min:
            video = video[:T_min]
        if T_p > T_min:
            pose = pose[:T_min]
        if T_e > T_min:
            emotion = emotion[:T_min]
        
        # Se T_min < window_size, fazer padding
        if T_min < self.window_size:
            # Repetir último frame
            if len(video.shape) == 4:  # (T, C, H, W)
                padding = video[-1:].repeat(self.window_size - T_min, 1, 1, 1)
            else:  # (T, D)
                padding = video[-1:].repeat(self.window_size - T_min, 1)
            video = torch.cat([video, padding], dim=0)
            
            if len(pose.shape) == 3:  # (T, num_joints, 3)
                padding = pose[-1:].repeat(self.window_size - T_min, 1, 1)
            else:  # (T, D)
                padding = pose[-1:].repeat(self.window_size - T_min, 1)
            pose = torch.cat([pose, padding], dim=0)
            
            padding = emotion[-1:].repeat(self.window_size - T_min, 1)
            emotion = torch.cat([emotion, padding], dim=0)
        else:
            # Selecionar janela aleatória (ou primeira se val/test)
            if self.split == "train" and T_min > self.window_size:
                start_idx = random.randint(0, T_min - self.window_size)
                video = video[start_idx:start_idx + self.window_size]
                pose = pose[start_idx:start_idx + self.window_size]
                emotion = emotion[start_idx:start_idx + self.window_size]
            else:
                video = video[:self.window_size]
                pose = pose[:self.window_size]
                emotion = emotion[:self.window_size]
        
        return video, pose, emotion
    
    def __len__(self) -> int:
        """Retorna o tamanho do dataset."""
        return len(self.samples)
    
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna uma amostra do dataset.
        
        Args:
            idx: Índice da amostra
        
        Returns:
            Tupla (video_features, pose_features, emotion_features, label)
        """
        video_id, label = self.samples[idx]
        
        # Determinar split original (train ou val)
        if self.split == "train":
            split_name = "train"
        else:  # val ou test - ambos vêm do split val original
            split_name = "val"
        
        # Carregar todas as modalidades
        video = self._load_video_features(video_id, label)
        pose = self._load_pose_features(video_id, split_name, label)
        emotion = self._load_emotion_features(video_id, split_name, label)
        
        # Criar janelas alinhadas
        video, pose, emotion = self._create_windows(video, pose, emotion)
        
        # Aplicar transformações
        if self.transform is not None:
            video, pose, emotion = self.transform(video, pose, emotion)
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return video, pose, emotion, label_tensor


def get_multimodal_dataloaders(
    video_data_root: str = None,
    pose_data_root: str = None,
    emotion_data_root: str = None,
    batch_size: int = 8,
    num_frames: int = 16,
    window_size: int = 16,
    video_mode: str = "frames",
    pose_mode: str = "keypoints",
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_original_split: bool = True,
    val_test_split_ratio: float = 0.5,
    seed: int = 42,
    dataset_name: str = "rwf2000"
):
    """
    Cria DataLoaders para treino, validação e teste multimodal.
    
    IMPORTANTE: Por padrão, agora usa a divisão original do RWF-2000 (train/val)
    para evitar data leakage. O split 'test' é criado dividindo o split 'val' original.
    
    Returns:
        Tupla (train_loader, val_loader, test_loader)
    """
    if video_data_root is None:
        video_data_root = str(p.PROCESSED_ROOT)
    if pose_data_root is None:
        pose_data_root = str(p.POSE_ROOT)
    if emotion_data_root is None:
        emotion_data_root = str(p.EMOTION_ROOT)
    from torch.utils.data import DataLoader
    
    # Criar datasets
    train_dataset = MultimodalSurveillanceDataset(
        video_data_root=video_data_root,
        pose_data_root=pose_data_root,
        emotion_data_root=emotion_data_root,
        split="train",
        num_frames=num_frames,
        window_size=window_size,
        video_mode=video_mode,
        pose_mode=pose_mode,
        transform=train_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    val_dataset = MultimodalSurveillanceDataset(
        video_data_root=video_data_root,
        pose_data_root=pose_data_root,
        emotion_data_root=emotion_data_root,
        split="val",
        num_frames=num_frames,
        window_size=window_size,
        video_mode=video_mode,
        pose_mode=pose_mode,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    test_dataset = MultimodalSurveillanceDataset(
        video_data_root=video_data_root,
        pose_data_root=pose_data_root,
        emotion_data_root=emotion_data_root,
        split="test",
        num_frames=num_frames,
        window_size=window_size,
        video_mode=video_mode,
        pose_mode=pose_mode,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    # Criar DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

