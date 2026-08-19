"""
Dataset PyTorch para sequências de keypoints de pose.

Este módulo implementa PoseSequenceDataset que:
1. Lê arquivos .npy com keypoints de pose
2. Cria janelas temporais de tamanho fixo (ex: 16 ou 32 frames)
3. Retorna tensores prontos para modelos temporais (LSTM, Transformer, CNN3D)
4. Suporta normalização e augmentations de pose
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import numpy as np
import random

from src import paths as p


class PoseSequenceDataset(Dataset):
    """
    Dataset para sequências de keypoints de pose.
    
    Lê arquivos .npy com keypoints e retorna janelas temporais de tamanho fixo.
    
    Formato dos dados:
    - Input: arquivo .npy com shape (T, num_joints, 3) onde 3 = (x, y, visibility)
    - Output: tensor (window_size, num_joints, 3) ou (window_size, num_joints * 3)
    """
    
    def __init__(
        self,
        pose_data_root: str,
        split: str = "train",
        window_size: int = 16,
        stride: int = 1,
        normalize: bool = True,
        flatten: bool = False,
        transform: Optional[Callable] = None,
        use_original_split: bool = True,
        val_test_split_ratio: float = 0.5,
        seed: int = 42,
        dataset_name: str = "rwf2000"  # "rwf2000"
    ):
        """
        Inicializa o dataset de pose.
        
        IMPORTANTE: Por padrão, agora preserva a divisão original do RWF-2000 
        (train/val) para evitar data leakage.
        
        Args:
            pose_data_root: Raiz dos dados de pose (ex: "data/pose")
            split: "train", "val" ou "test"
            window_size: Tamanho da janela temporal (número de frames)
            stride: Stride para criar janelas (1 = todas as janelas possíveis)
            normalize: Se True, normaliza keypoints para média 0 e std 1
            flatten: Se True, retorna (window_size, num_joints * 3) ao invés de (window_size, num_joints, 3)
            transform: Transformações a aplicar (augmentations)
            use_original_split: Se True (padrão), usa divisão original do RWF-2000.
                               Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
            val_test_split_ratio: Se use_original_split=True, divide o val original em val e test
                                 usando esta proporção (padrão: 0.5 = 50/50)
            seed: Seed para reprodutibilidade
dataset_name: Nome do dataset ("rwf2000")
        """
        self.pose_data_root = Path(pose_data_root)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
        self.flatten = flatten
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
                f"em {pose_data_root} (dataset: {dataset_name})"
            )
        
        # Calcular estatísticas para normalização se necessário
        if normalize:
            self.mean, self.std = self._compute_statistics()
        else:
            self.mean = None
            self.std = None
        
        print(f"PoseSequenceDataset {split}: {len(self.samples)} amostras carregadas")
        if normalize:
            print(f"  Normalização: mean={self.mean.shape}, std={self.std.shape}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Carrega lista de arquivos .npy e seus labels preservando a divisão original.
        
        Returns:
            Lista de tuplas (caminho_para_npy, label)
        """
        samples = []
        
        if self.dataset_name == "rwf2000":
            # Estrutura: data/pose/rwf2000/<split>/<violent|non_violent>/<video>.npy
            base_dir = self.pose_data_root / "rwf2000"
            
            if self.use_original_split:
                # Preservar divisão original: usar apenas o split solicitado
                # train -> train original
                # val/test -> val original (depois dividimos)
                if self.split == "train":
                    original_split = "train"
                elif self.split in ["val", "test"]:
                    original_split = "val"
                else:
                    raise ValueError(f"Split inválido: {self.split}")
                
                split_dir = base_dir / original_split
                if not split_dir.exists():
                    return []
                
                # Vídeos violentos (label=1)
                violent_dir = split_dir / "violent"
                if violent_dir.exists():
                    for npy_file in violent_dir.glob("*.npy"):
                        samples.append((npy_file, 1))
                
                # Vídeos não violentos (label=0)
                non_violent_dir = split_dir / "non_violent"
                if non_violent_dir.exists():
                    for npy_file in non_violent_dir.glob("*.npy"):
                        samples.append((npy_file, 0))
                
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
                    split_dir = base_dir / split_name
                    if not split_dir.exists():
                        continue
                    
                    violent_dir = split_dir / "violent"
                    if violent_dir.exists():
                        for npy_file in violent_dir.glob("*.npy"):
                            samples.append((npy_file, 1))
                    
                    non_violent_dir = split_dir / "non_violent"
                    if non_violent_dir.exists():
                        for npy_file in non_violent_dir.glob("*.npy"):
                            samples.append((npy_file, 0))
                
                # Embaralhar e dividir aleatoriamente (CAUSA DATA LEAKAGE)
                random.seed(self.seed)
                random.shuffle(samples)
                
                total = len(samples)
                train_end = int(total * 0.7)
                val_end = train_end + int(total * 0.15)
                
                if self.split == "train":
                    samples = samples[:train_end]
                elif self.split == "val":
                    samples = samples[train_end:val_end]
                else:  # test
                    samples = samples[val_end:]
        
        else:
            raise ValueError(f"Dataset não suportado: {self.dataset_name}. Use 'rwf2000'")
        
        return samples
    
    def _compute_statistics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula média e desvio padrão dos keypoints para normalização.
        
        Returns:
            Tupla (mean, std) com shape (num_joints, 3) ou (num_joints * 3,)
        """
        print("Calculando estatísticas para normalização...")
        
        all_keypoints = []
        
        # Amostrar alguns arquivos para calcular estatísticas
        sample_size = min(100, len(self.samples))
        sampled_indices = random.sample(range(len(self.samples)), sample_size)
        
        for idx in sampled_indices:
            npy_path, _ = self.samples[idx]
            try:
                keypoints = np.load(npy_path)
                # keypoints shape: (T, num_joints, 3)
                all_keypoints.append(keypoints)
            except Exception as e:
                print(f"  Aviso: Erro ao carregar {npy_path}: {str(e)}")
                continue
        
        if len(all_keypoints) == 0:
            # Se não conseguir carregar, retornar zeros
            return np.zeros((33, 3)), np.ones((33, 3))
        
        # Concatenar todos os keypoints
        all_keypoints = np.concatenate(all_keypoints, axis=0)  # (total_frames, num_joints, 3)
        
        # Calcular média e std por joint e coordenada
        mean = np.mean(all_keypoints, axis=0)  # (num_joints, 3)
        std = np.std(all_keypoints, axis=0)  # (num_joints, 3)
        
        # Evitar divisão por zero
        std = np.where(std < 1e-6, 1.0, std)
        
        return mean, std
    
    def _create_windows(
        self,
        keypoints: np.ndarray
    ) -> List[np.ndarray]:
        """
        Cria janelas temporais de keypoints.
        
        Args:
            keypoints: Array (T, num_joints, 3)
        
        Returns:
            Lista de janelas, cada uma com shape (window_size, num_joints, 3)
        """
        T = keypoints.shape[0]
        
        if T < self.window_size:
            # Se sequência é menor que a janela, repetir último frame
            padding = np.repeat(keypoints[-1:], self.window_size - T, axis=0)
            window = np.concatenate([keypoints, padding], axis=0)
            return [window]
        
        # Criar janelas com stride
        windows = []
        for start_idx in range(0, T - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = keypoints[start_idx:end_idx]
            windows.append(window)
        
        # Se não criou nenhuma janela (edge case), criar uma com os últimos frames
        if len(windows) == 0:
            window = keypoints[-self.window_size:]
            windows.append(window)
        
        return windows
    
    def __len__(self) -> int:
        """Retorna o número total de janelas no dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna uma amostra do dataset.
        
        Args:
            idx: Índice da amostra
        
        Returns:
            Tupla (pose_window, label)
            - pose_window: (window_size, num_joints, 3) ou (window_size, num_joints * 3)
            - label: tensor escalar com label
        """
        npy_path, label = self.samples[idx]
        
        # Carregar keypoints
        try:
            keypoints = np.load(npy_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar {npy_path}: {str(e)}")
        
        # Validar shape
        if len(keypoints.shape) != 3 or keypoints.shape[2] != 3:
            raise ValueError(
                f"Keypoints devem ter shape (T, num_joints, 3), "
                f"mas têm {keypoints.shape}"
            )
        
        # Criar janelas
        windows = self._create_windows(keypoints)
        
        # Selecionar uma janela aleatória (ou a primeira se for val/test)
        if self.split == "train" and len(windows) > 1:
            window = random.choice(windows)
        else:
            window = windows[0]  # Primeira janela (ou única)
        
        # Normalizar se solicitado
        if self.normalize and self.mean is not None and self.std is not None:
            window = (window - self.mean) / self.std
        
        # Aplicar transformações (augmentations)
        if self.transform is not None:
            window = self.transform(window)
        
        # Converter para tensor
        window_tensor = torch.from_numpy(window).float()
        
        # Flatten se solicitado
        if self.flatten:
            # (window_size, num_joints, 3) -> (window_size, num_joints * 3)
            window_tensor = window_tensor.view(self.window_size, -1)
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return window_tensor, label_tensor


def get_pose_dataloaders(
    pose_data_root: str = None,
    batch_size: int = 8,
    window_size: int = 16,
    stride: int = 1,
    normalize: bool = True,
    flatten: bool = False,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_original_split: bool = True,
    val_test_split_ratio: float = 0.5,
    seed: int = 42,
    dataset_name: str = "rwf2000"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria DataLoaders para treino, validação e teste de pose.
    
    IMPORTANTE: Por padrão, agora usa a divisão original do RWF-2000 (train/val)
    para evitar data leakage. O split 'test' é criado dividindo o split 'val' original.
    
    Args:
        pose_data_root: Raiz dos dados de pose
        batch_size: Tamanho do batch
        window_size: Tamanho da janela temporal
        stride: Stride para criar janelas
        normalize: Se True, normaliza keypoints
        flatten: Se True, retorna keypoints flatten
        num_workers: Número de workers para carregamento
        train_transform: Transformações para treino
        val_transform: Transformações para validação/teste
        use_original_split: Se True (padrão), usa divisão original do RWF-2000.
                           Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
        val_test_split_ratio: Se use_original_split=True, divide o val original em val e test
                             usando esta proporção (padrão: 0.5 = 50/50)
        seed: Seed para reprodutibilidade
        dataset_name: Nome do dataset ("rwf2000")
    
    Returns:
        Tupla (train_loader, val_loader, test_loader)
    """
    if pose_data_root is None:
        pose_data_root = str(p.POSE_ROOT)

    # Criar datasets
    train_dataset = PoseSequenceDataset(
        pose_data_root=pose_data_root,
        split="train",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        flatten=flatten,
        transform=train_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    val_dataset = PoseSequenceDataset(
        pose_data_root=pose_data_root,
        split="val",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        flatten=flatten,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    test_dataset = PoseSequenceDataset(
        pose_data_root=pose_data_root,
        split="test",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        flatten=flatten,
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

