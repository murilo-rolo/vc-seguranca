"""
Datasets para CNN 3D: RWF-2000.

Este módulo implementa datasets para treinamento de modelos CNN 3D:
- RWF2000Video3DDataset: Para fine-tuning em RWF-2000 (2 classes)
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import random
import cv2
import numpy as np
from torchvision import transforms

from src import paths as p


class RWF2000Video3DDataset(Dataset):
    """
    Dataset RWF-2000 para CNN 3D (fine-tuning).
    
    Carrega vídeos do RWF-2000 e retorna clipes 3D para detecção de violência.
    
    Estrutura esperada:
    dataset/RWF-2000/
    ├── train/
    │   ├── Fight/
    │   └── NonFight/
    └── val/
        ├── Fight/
        └── NonFight/
    """
    
    def __init__(
        self,
        dataset_root: str = None,
        split: str = "train",
        num_frames: int = 16,
        clip_size: Tuple[int, int] = (112, 112),
        transform: Optional[Callable] = None,
        sample_stride: int = 1
    ):
        """
        Inicializa o dataset RWF-2000.
        
        Args:
            dataset_root: Raiz do dataset RWF-2000
            split: "train" ou "val"
            num_frames: Número de frames por clipe
            clip_size: Tamanho do clipe (H, W)
            transform: Transformações a aplicar
            sample_stride: Stride para amostragem de frames
        """
        if dataset_root is None:
            dataset_root = str(p.RWF2000_ROOT)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.num_frames = num_frames
        self.clip_size = clip_size
        self.transform = transform
        self.sample_stride = sample_stride
        
        # Carregar amostras
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"Nenhuma amostra encontrada para split '{split}' em {dataset_root}")
        
        print(f"RWF2000Video3DDataset {split}: {len(self.samples)} amostras")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Carrega lista de amostras (video_path, label)."""
        samples = []
        split_dir = self.dataset_root / self.split
        
        if not split_dir.exists():
            return samples
        
        # Vídeos violentos (label=1)
        fight_dir = split_dir / "Fight"
        if fight_dir.exists():
            for video_file in fight_dir.glob("*.avi"):
                samples.append((video_file, 1))
        
        # Vídeos não violentos (label=0)
        nonfight_dir = split_dir / "NonFight"
        if nonfight_dir.exists():
            for video_file in nonfight_dir.glob("*.avi"):
                samples.append((video_file, 0))
        
        return samples
    
    def _load_video_clip(self, video_path: Path) -> torch.Tensor:
        """
        Carrega clipe de vídeo.
        
        Args:
            video_path: Caminho para o vídeo
        
        Returns:
            Tensor (T, C, H, W) com frames do clipe
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Erro ao abrir vídeo: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Vídeo vazio: {video_path}")
        
        # Selecionar frames uniformemente espaçados
        if total_frames < self.num_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frame = np.zeros((self.clip_size[0], self.clip_size[1], 3), dtype=np.uint8)
                    frames.append(frame)
                continue
            
            # Converter BGR para RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar
            frame = cv2.resize(frame, self.clip_size)
            
            frames.append(frame)
        
        cap.release()
        
        # Converter para tensor
        frames_array = np.array(frames)  # (T, H, W, C)
        frames_tensor = torch.from_numpy(frames_array).float()
        
        # Converter de (T, H, W, C) para (T, C, H, W)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)
        
        # Normalizar para [0, 1]
        frames_tensor = frames_tensor / 255.0
        
        return frames_tensor
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna uma amostra.
        
        Returns:
            Tupla (clip, label) onde clip é (T, C, H, W) e label é 0 ou 1
        """
        video_path, label = self.samples[idx]
        
        # Carregar clipe
        clip = self._load_video_clip(video_path)
        
        # Aplicar transformações
        if self.transform is not None:
            clip = self.transform(clip)
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return clip, label_tensor


def get_rwf2000_3d_dataloaders(
    dataset_root: str = None,
    batch_size: int = 8,
    num_frames: int = 16,
    clip_size: Tuple[int, int] = (112, 112),
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None
):
    """
    Cria DataLoaders para RWF-2000 (fine-tuning).
    
    Returns:
        Tupla (train_loader, val_loader)
    """
    if dataset_root is None:
        dataset_root = str(p.RWF2000_ROOT)
    from torch.utils.data import DataLoader
    
    train_dataset = RWF2000Video3DDataset(
        dataset_root=dataset_root,
        split="train",
        num_frames=num_frames,
        clip_size=clip_size,
        transform=train_transform
    )
    
    val_dataset = RWF2000Video3DDataset(
        dataset_root=dataset_root,
        split="val",
        num_frames=num_frames,
        clip_size=clip_size,
        transform=val_transform
    )
    
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
    
    return train_loader, val_loader

