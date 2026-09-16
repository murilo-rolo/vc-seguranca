"""
Dataset para classificação de vídeos de segurança (violência vs não-violência).

Implementa a classe SurveillanceRiskDataset que lê sequências de frames
pré-processadas e retorna tensores prontos para treinamento.

CORRIGIDO: Agora preserva a divisão train/val original do RWF-2000 para evitar
data leakage e garantir métricas válidas.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Callable
import random
import warnings

from src import paths as p


class SurveillanceRiskDataset(Dataset):
    """
    Dataset para classificação de risco em vídeos de segurança.
    
    Lê sequências de frames pré-processadas de data/processed e retorna
    (tensor_de_frames, label) onde:
    - tensor_de_frames: (num_frames, C, H, W)
    - label: 0 (non-violent) ou 1 (violent)
    
    IMPORTANTE: Este dataset agora preserva a divisão train/val original do 
    RWF-2000 para evitar vazamento de dados (data leakage). O split 'test' 
    é criado a partir do split 'val' original.
    """
    
    def __init__(
        self,
        processed_data_root: str = None,
        split: str = "train",
        num_frames: int = 16,
        transform: Optional[Callable] = None,
        use_original_split: bool = True,
        val_test_split_ratio: float = 0.5,
        seed: int = 42
    ):
        """
        Inicializa o dataset.
        
        Args:
            processed_data_root: Raiz dos dados processados
            split: "train", "val" ou "test"
            num_frames: Número de frames esperados por vídeo
            transform: Transformações a aplicar (augmentations)
            use_original_split: Se True, usa a divisão original do RWF-2000 (train/val).
                               Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
            val_test_split_ratio: Se use_original_split=True, divide o val original em
                                 val e test usando esta proporção (padrão: 0.5 = 50/50)
            seed: Seed para reprodutibilidade
        """
        if processed_data_root is None:
            processed_data_root = str(p.PROCESSED_ROOT)
        self.processed_data_root = Path(processed_data_root)
        self.split = split
        self.num_frames = num_frames
        self.transform = transform
        self.use_original_split = use_original_split
        self.val_test_split_ratio = val_test_split_ratio
        self.seed = seed
        
        # Carregar lista de vídeos
        self.samples = self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"Nenhuma amostra encontrada para o split '{split}' em {processed_data_root}")
        
        print(f"Dataset {split}: {len(self.samples)} amostras carregadas")
        if use_original_split:
            print(f"  [OK] Usando divisao original do RWF-2000 (train/val)")
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """
        Carrega lista de amostras (caminho, label) preservando a divisão original.
        
        Tenta primeiro usar a estrutura que preserva train/val do RWF-2000.
        Se não encontrar, tenta usar diretamente do dataset original.
        Se não encontrar nenhuma, avisa e usa estrutura antiga (DEPRECADO).
        
        Returns:
            Lista de tuplas (caminho_para_frames, label)
        """
        # Tentar encontrar estrutura que preserve splits originais
        # Estrutura nova: data/processed/train|val/violent|non_violent/
        if self.use_original_split:
            test_dir = self.processed_data_root / "train"
            if test_dir.exists():
                return self._load_samples_from_rwf2000_structure(self.processed_data_root)

            # Estrutura alternativa: data/processed/rwf2000/train|val/violent|non_violent/
            rwf2000_path = self.processed_data_root / "rwf2000"
            if rwf2000_path.exists():
                return self._load_samples_from_rwf2000_structure(rwf2000_path)

        # Fallback: estrutura antiga (DEPRECADO - causa data leakage)
        if not self.use_original_split:
            warnings.warn(
                "⚠️  AVISO: Usando divisão aleatória pode causar DATA LEAKAGE! "
                "Os resultados podem estar inflados. Recomenda-se usar use_original_split=True.",
                UserWarning
            )
        return self._load_samples_from_legacy_structure()
    
    def _load_samples_from_rwf2000_structure(self, base_path: Path) -> List[Tuple[Path, int]]:
        """
        Carrega amostras preservando a divisão original train/val do RWF-2000.
        
        Args:
            base_path: Caminho base (ex: data/processed/rwf2000)
        
        Returns:
            Lista de tuplas (caminho_para_frames, label)
        """
        samples = []
        
        # Mapear split solicitado para split original
        if self.split == "train":
            original_split = "train"
        elif self.split in ["val", "test"]:
            # Ambos val e test vêm do split 'val' original
            original_split = "val"
        else:
            raise ValueError(f"Split inválido: {self.split}. Deve ser 'train', 'val' ou 'test'.")
        
        split_dir = base_path / original_split
        if not split_dir.exists():
            # Se não encontrar, tentar estrutura sem rwf2000
            split_dir = base_path / original_split if original_split in ["train", "val"] else base_path / "val"
            if not split_dir.exists():
                return []
        
        # Carregar vídeos violentos (label=1)
        violent_dir = split_dir / "violent"
        if violent_dir.exists():
            for video_dir in violent_dir.iterdir():
                if video_dir.is_dir():
                    frame_path = video_dir / "frame_sequence.pt"
                    if frame_path.exists():
                        samples.append((frame_path, 1, original_split))
        
        # Carregar vídeos não violentos (label=0)
        non_violent_dir = split_dir / "non_violent"
        if non_violent_dir.exists():
            for video_dir in non_violent_dir.iterdir():
                if video_dir.is_dir():
                    frame_path = video_dir / "frame_sequence.pt"
                    if frame_path.exists():
                        samples.append((frame_path, 0, original_split))
        
        # Se solicitamos val ou test, precisamos dividir o val original
        if original_split == "val" and len(samples) > 0:
            # Embaralhar com seed para reprodutibilidade
            random.seed(self.seed)
            random.shuffle(samples)
            
            # Dividir val em val e test
            total_val = len(samples)
            val_end = int(total_val * self.val_test_split_ratio)
            
            if self.split == "val":
                samples = samples[:val_end]
            else:  # test
                samples = samples[val_end:]
        
        # Remover o terceiro elemento (split) e retornar apenas (path, label)
        return [(path, label) for path, label, _ in samples]
    
    def _load_samples_from_legacy_structure(self) -> List[Tuple[Path, int]]:
        """
        Carrega amostras da estrutura legada (sem preservar splits originais).
        
        DEPRECADO: Esta estrutura mistura train e val, causando data leakage.
        
        Returns:
            Lista de tuplas (caminho_para_frames, label)
        """
        samples = []
        
        # Carregar vídeos violentos (label=1)
        violent_dir = self.processed_data_root / "violent"
        if violent_dir.exists():
            for video_dir in violent_dir.iterdir():
                if video_dir.is_dir():
                    frame_path = video_dir / "frame_sequence.pt"
                    if frame_path.exists():
                        samples.append((frame_path, 1))
        
        # Carregar vídeos não violentos (label=0)
        non_violent_dir = self.processed_data_root / "non_violent"
        if non_violent_dir.exists():
            for video_dir in non_violent_dir.iterdir():
                if video_dir.is_dir():
                    frame_path = video_dir / "frame_sequence.pt"
                    if frame_path.exists():
                        samples.append((frame_path, 0))
        
        # Embaralhar com seed fixa
        random.seed(self.seed)
        random.shuffle(samples)
        
        # Dividir em splits (ESTE É O PROBLEMA - mistura train e val original)
        total = len(samples)
        train_end = int(total * 0.7)  # 70% treino
        val_end = train_end + int(total * 0.15)  # 15% val
        
        if self.split == "train":
            return samples[:train_end]
        elif self.split == "val":
            return samples[train_end:val_end]
        else:  # test
            return samples[val_end:]
    
    def __len__(self) -> int:
        """Retorna o tamanho do dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna uma amostra do dataset.
        
        Args:
            idx: Índice da amostra
        
        Returns:
            Tupla (frames_tensor, label_tensor)
            - frames_tensor: (num_frames, C, H, W)
            - label_tensor: tensor escalar com label (0 ou 1)
        """
        frame_path, label = self.samples[idx]
        
        # Carregar tensor de frames com tratamento de erro
        try:
            frames = torch.load(frame_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Erro ao carregar {frame_path}: {str(e)}")
        
        # Validar shape do tensor
        if len(frames.shape) != 4:
            raise ValueError(f"Tensor deve ter 4 dimensões (num_frames, C, H, W), mas tem {len(frames.shape)}")
        
        # Verificar número de frames
        if frames.shape[0] != self.num_frames:
            # Se tiver menos frames, repetir o último
            # Se tiver mais, pegar os primeiros N
            if frames.shape[0] < self.num_frames:
                last_frame = frames[-1:].repeat(self.num_frames - frames.shape[0], 1, 1, 1)
                frames = torch.cat([frames, last_frame], dim=0)
            else:
                frames = frames[:self.num_frames]
        
        # Aplicar transformações (augmentations) se for treino
        if self.transform is not None:
            try:
                frames = self.transform(frames)
            except Exception as e:
                print(f"Erro ao aplicar transformação: {str(e)}")
                # Continuar sem transformação se houver erro
        
        # Converter label para tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return frames, label_tensor


def get_dataloaders(
    processed_data_root: str = None,
    batch_size: int = 8,
    num_frames: int = 16,
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    use_original_split: bool = True,
    val_test_split_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Cria DataLoaders para treino, validação e teste.
    
    IMPORTANTE: Por padrão, agora usa a divisão original do RWF-2000 (train/val)
    para evitar data leakage. O split 'test' é criado dividindo o split 'val' original.
    
    Args:
        processed_data_root: Raiz dos dados processados
        batch_size: Tamanho do batch
        num_frames: Número de frames por vídeo
        num_workers: Número de workers para carregamento
        train_transform: Transformações para treino
        val_transform: Transformações para validação/teste
        use_original_split: Se True (padrão), usa divisão original do RWF-2000.
                           Se False, usa divisão aleatória (DEPRECADO - causa data leakage).
        val_test_split_ratio: Proporção para dividir val original em val e test (padrão: 0.5)
        seed: Seed para reprodutibilidade
    
    Returns:
        Tupla (train_loader, val_loader, test_loader)
    """
    if processed_data_root is None:
        processed_data_root = str(p.PROCESSED_ROOT)

    # Criar datasets
    train_dataset = SurveillanceRiskDataset(
        processed_data_root=processed_data_root,
        split="train",
        num_frames=num_frames,
        transform=train_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed
    )
    
    val_dataset = SurveillanceRiskDataset(
        processed_data_root=processed_data_root,
        split="val",
        num_frames=num_frames,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed
    )
    
    test_dataset = SurveillanceRiskDataset(
        processed_data_root=processed_data_root,
        split="test",
        num_frames=num_frames,
        transform=val_transform,
        use_original_split=use_original_split,
        val_test_split_ratio=val_test_split_ratio,
        seed=seed
    )
    
    # Criar DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader

