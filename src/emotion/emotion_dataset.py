"""
Dataset PyTorch para sequências de vetores de emoção.

Este módulo implementa EmotionSequenceDataset que:
1. Lê arquivos .npy com vetores de emoção
2. Cria janelas temporais de tamanho fixo (ex: 16 ou 32 frames)
3. Retorna tensores prontos para modelos temporais (LSTM, Transformer, CNN3D)
4. Suporta normalização e augmentations de emoção
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import numpy as np
import random

from src import paths as p


class EmotionSequenceDataset(Dataset):
    """
    Dataset para sequências de vetores de emoção.
    
    Lê arquivos .npy com vetores de emoção e retorna janelas temporais de tamanho fixo.
    
    Formato dos dados:
    - Input: arquivo .npy com shape (T, num_emotions) onde cada linha é um vetor de probabilidades
    - Output: tensor (window_size, num_emotions)
    """
    
    def __init__(
        self,
        emotion_data_root: str,
        split: str = "train",
        window_size: int = 16,
        stride: int = 1,
        normalize: bool = False,  # Emoções já são probabilidades [0, 1]
        transform: Optional[Callable] = None,
        use_original_split: bool = True,
        val_test_split_ratio: float = 0.5,
        seed: int = 42,
        dataset_name: str = "rwf2000"  # "rwf2000"
    ):
        """
        Inicializa o dataset de emoção.
        
        IMPORTANTE: Por padrão, agora preserva a divisão original do RWF-2000 
        (train/val) para evitar data leakage.
        
        Args:
            emotion_data_root: Raiz dos dados de emoção (ex: "data/emotion")
            split: "train", "val" ou "test"
            window_size: Tamanho da janela temporal (número de frames)
            stride: Stride para criar janelas (1 = todas as janelas possíveis)
            normalize: Se True, normaliza vetores (geralmente não necessário, já são probabilidades)
            transform: Transformações a aplicar (augmentations)
            use_original_split: Se True (padrão), usa divisão original do RWF-2000.
                               Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
            val_test_split_ratio: Se use_original_split=True, divide o val original em val e test
                                 usando esta proporção (padrão: 0.5 = 50/50)
            seed: Seed para reprodutibilidade
            dataset_name: Nome do dataset ("rwf2000")
        """
        self.emotion_data_root = Path(emotion_data_root)
        self.split = split
        self.window_size = window_size
        self.stride = stride
        self.normalize = normalize
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
                f"em {emotion_data_root} (dataset: {dataset_name})"
            )
        
        # Obter num_emotions do primeiro arquivo
        if len(self.samples) > 0:
            first_file = self.samples[0][0]
            try:
                sample_data = np.load(first_file)
                self.num_emotions = sample_data.shape[1] if len(sample_data.shape) > 1 else sample_data.shape[0]
            except:
                self.num_emotions = 128  # Padrão para embeddings 128-d
        else:
            self.num_emotions = 128
        
        print(f"EmotionSequenceDataset {split}: {len(self.samples)} amostras carregadas")
        print(f"  Número de emoções: {self.num_emotions}")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Carrega lista de arquivos .npy e seus labels preservando a divisão original.
        
        Returns:
            Lista de tuplas (caminho_para_npy, label)
        """
        samples = []
        
        if self.dataset_name == "rwf2000":
            # Estrutura: data/emotion/rwf2000/<split>/<violent|non_violent>/<video>.npy
            base_dir = self.emotion_data_root / "rwf2000"
            
            if self.use_original_split:
                # Preservar divisão original: usar apenas o split solicitado
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
            raise ValueError(f"Dataset não suportado: {self.dataset_name}")
        
        return samples
    
    def _create_windows(
        self,
        emotions: np.ndarray
    ) -> List[np.ndarray]:
        """
        Cria janelas temporais de vetores de emoção.
        
        Args:
            emotions: Array (T, num_emotions)
        
        Returns:
            Lista de janelas, cada uma com shape (window_size, num_emotions)
        """
        T = emotions.shape[0]
        
        if T < self.window_size:
            # Se sequência é menor que a janela, repetir último vetor
            padding = np.repeat(emotions[-1:], self.window_size - T, axis=0)
            window = np.concatenate([emotions, padding], axis=0)
            return [window]
        
        # Criar janelas com stride
        windows = []
        for start_idx in range(0, T - self.window_size + 1, self.stride):
            end_idx = start_idx + self.window_size
            window = emotions[start_idx:end_idx]
            windows.append(window)
        
        # Se não criou nenhuma janela (edge case), criar uma com os últimos frames
        if len(windows) == 0:
            window = emotions[-self.window_size:]
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
            Tupla (emotion_window, label)
            - emotion_window: (window_size, num_emotions)
            - label: tensor escalar com label
        """
        npy_path, label = self.samples[idx]
        
        # Carregar vetores de emoção
        try:
            emotions = np.load(npy_path)
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar {npy_path}: {str(e)}")
        
        # Validar shape
        if len(emotions.shape) != 2 or emotions.shape[1] != self.num_emotions:
            # Se for 1D, expandir
            if len(emotions.shape) == 1 and emotions.shape[0] == self.num_emotions:
                emotions = emotions.reshape(1, -1)
            else:
                raise ValueError(
                    f"Vetores de emoção devem ter shape (T, num_emotions), "
                    f"mas têm {emotions.shape}"
                )
        
        # Criar janelas
        windows = self._create_windows(emotions)
        
        # Selecionar uma janela aleatória (ou a primeira se for val/test)
        if self.split == "train" and len(windows) > 1:
            window = random.choice(windows)
        else:
            window = windows[0]  # Primeira janela (ou única)
        
        # Normalizar se solicitado (geralmente não necessário para probabilidades)
        if self.normalize:
            # Normalizar cada vetor para soma = 1 (já deve estar, mas garantir)
            window = window / (window.sum(axis=1, keepdims=True) + 1e-8)
        
        # Aplicar transformações (augmentations)
        if self.transform is not None:
            window = self.transform(window)
        
        # Converter para tensor
        window_tensor = torch.from_numpy(window).float()
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return window_tensor, label_tensor


def get_emotion_dataloaders(
    emotion_data_root: str = None,
    batch_size: int = 8,
    window_size: int = 16,
    stride: int = 1,
    normalize: bool = False,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_original_split: bool = True,
    val_test_split_ratio: float = 0.5,
    seed: int = 42,
    dataset_name: str = "rwf2000"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Cria DataLoaders para treino, validação e teste de emoção.
    
    IMPORTANTE: Por padrão, agora usa a divisão original do RWF-2000 (train/val)
    para evitar data leakage. O split 'test' é criado dividindo o split 'val' original.
    
    Args:
        emotion_data_root: Raiz dos dados de emoção
        batch_size: Tamanho do batch
        window_size: Tamanho da janela temporal
        stride: Stride para criar janelas
        normalize: Se True, normaliza vetores (geralmente não necessário)
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
    if emotion_data_root is None:
        emotion_data_root = str(p.EMOTION_ROOT)

    # Criar datasets
    train_dataset = EmotionSequenceDataset(
        emotion_data_root=emotion_data_root,
        split="train",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        transform=train_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    val_dataset = EmotionSequenceDataset(
        emotion_data_root=emotion_data_root,
        split="val",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed,
        dataset_name=dataset_name
    )
    
    test_dataset = EmotionSequenceDataset(
        emotion_data_root=emotion_data_root,
        split="test",
        window_size=window_size,
        stride=stride,
        normalize=normalize,
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

