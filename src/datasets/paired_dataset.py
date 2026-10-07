"""
Dataset pareado RWF-2000 (vídeo) × AffectNet (face) via dataset/paired/.

Cada amostra é um diretório <split>/<cell>/<pair_id>/ com symlinks:
    frames.pt (ou features.pt), pose.npy (opcional), emotion.npy

Label = rótulo do VÍDEO; o rótulo da face é covariável (guardado em
sample["face_label"], acessível via get_metadata()).

Estrutura gerada por src/preprocessing/build_paired_dataset.py.
"""

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from src import paths as p
from src.datasets.multimodal_dataset import MultimodalSurveillanceDataset
from src.preprocessing.build_paired_dataset import CELLS, MANIFEST_NAME

DEFAULT_POSE_JOINTS = 133


def paired_root_exists(paired_root: Optional[Path] = None) -> bool:
    root = Path(paired_root) if paired_root else p.PAIRED_ROOT
    return (root / MANIFEST_NAME).exists()


class PairedSurveillanceDataset(MultimodalSurveillanceDataset):
    """
    Dataset multimodal que lê pares (vídeo, face) de dataset/paired/.

    Subclasse de MultimodalSurveillanceDataset para reaproveitar
    _create_windows, transforms e formatos de saída; o carregamento por
    par sobrescreve toda a resolução de caminhos por split original
    (a partição do RWF-2000 é irrelevante: o symlink guarda o caminho).
    """

    def __init__(
        self,
        split: str = "train",
        paired_root: Optional[str] = None,
        cells: Optional[List[str]] = None,
        num_frames: int = 16,
        window_size: int = 16,
        video_mode: str = "frames",
        pose_mode: str = "keypoints",
        transform: Optional[Callable] = None,
        seed: int = 42,
    ):
        self.paired_root = Path(paired_root) if paired_root else p.PAIRED_ROOT
        self.requested_cells = list(cells) if cells else None
        self.seed = seed
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split inválido: {split}")
        if self.requested_cells:
            unknown = set(self.requested_cells) - set(CELLS)
            if unknown:
                raise ValueError(f"Células desconhecidas: {sorted(unknown)}")

        super().__init__(
            video_data_root=str(p.PROCESSED_ROOT),
            pose_data_root=str(p.POSE_ROOT),
            emotion_data_root=str(p.EMOTION_ROOT),
            split=split,
            num_frames=num_frames,
            window_size=window_size,
            video_mode=video_mode,
            pose_mode=pose_mode,
            transform=transform,
            seed=seed,
        )

    # ── Carregamento ────────────────────────────────────────────────────────

    def _load_samples(self) -> List[Dict]:
        split_dir = self.paired_root / self.split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Split '{self.split}' não encontrado em {split_dir}. "
                "Gere o dataset com: python -m src.preprocessing.build_paired_dataset"
            )
        self._pose_default_cache = None

        samples: List[Dict] = []
        skipped = 0
        for cell_dir in sorted(split_dir.glob("*")):
            if not cell_dir.is_dir():
                continue
            cell = cell_dir.name
            if cell not in CELLS:
                continue
            if self.requested_cells and cell not in self.requested_cells:
                continue
            video_class, face_class = CELLS[cell]
            for pair_dir in sorted(cell_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                frames = pair_dir / (
                    "frames.pt" if self.video_mode == "frames" else "features.pt"
                )
                emotion = pair_dir / "emotion.npy"
                if not (frames.exists() and emotion.exists()):
                    skipped += 1
                    continue
                pair_id = pair_dir.name
                video_id, sep, face_id = pair_id.rpartition("__")
                if not sep:
                    video_id, face_id = pair_id, ""
                samples.append({
                    "pair_dir": pair_dir,
                    "pair_id": pair_id,
                    "video_id": video_id,
                    "face_id": face_id,
                    "cell": cell,
                    "video_label": 1 if video_class == "violent" else 0,
                    "face_label": 1 if face_class == "violent" else 0,
                    "has_pose": (pair_dir / "pose.npy").exists(),
                })

        if skipped:
            print(f"⚠️  {skipped} pares ignorados em '{self.split}' "
                  f"(symlinks quebrados em {split_dir})")
        return samples

    def _pose_default(self) -> torch.Tensor:
        """Zeros com nº de juntas coerente com o primeiro pose.npy encontrado."""
        if getattr(self, "_pose_default_cache", None) is None:
            joints = DEFAULT_POSE_JOINTS
            for sample in self.samples:
                pose_path = sample["pair_dir"] / "pose.npy"
                if pose_path.exists():
                    joints = np.load(pose_path, mmap_mode="r").shape[1]
                    break
            self._pose_default_cache = torch.zeros(self.num_frames, joints, 3)
        return self._pose_default_cache

    def get_metadata(self) -> List[Dict]:
        """Metadados de cada amostra (cell, video_label, face_label, ids)."""
        return [
            {k: (str(v) if isinstance(v, Path) else v) for k, v in s.items()}
            for s in self.samples
        ]

    def get_cell_counts(self) -> Dict[str, int]:
        return dict(Counter(s["cell"] for s in self.samples))

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        pair_dir = sample["pair_dir"]

        if self.video_mode == "frames":
            video = torch.load(pair_dir / "frames.pt", map_location="cpu")
            if video.shape[0] < self.num_frames:
                last = video[-1:].repeat(self.num_frames - video.shape[0], 1, 1, 1)
                video = torch.cat([video, last], dim=0)
            elif video.shape[0] > self.num_frames:
                video = video[: self.num_frames]
        else:
            video = torch.load(pair_dir / "features.pt", map_location="cpu")
            if len(video.shape) == 1:
                video = video.unsqueeze(0)

        pose_path = pair_dir / "pose.npy"
        if pose_path.exists():
            pose = torch.from_numpy(np.load(pose_path)).float()
        else:
            pose = self._pose_default()
        if self.pose_mode == "flatten":
            pose = pose.reshape(pose.shape[0], -1)

        emotion = torch.from_numpy(np.load(pair_dir / "emotion.npy")).float()
        if emotion.ndim == 1:
            emotion = emotion.reshape(1, -1)

        video, pose, emotion = self._create_windows(video, pose, emotion)

        if self.transform is not None:
            video, pose, emotion = self.transform(video, pose, emotion)

        label = torch.tensor(sample["video_label"], dtype=torch.long)
        return video, pose, emotion, label


def get_paired_dataloaders(
    paired_root: Optional[str] = None,
    batch_size: int = 8,
    num_frames: int = 16,
    window_size: int = 16,
    video_mode: str = "frames",
    pose_mode: str = "keypoints",
    num_workers: int = 4,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    seed: int = 42,
    test_cells: Optional[List[str]] = None,
):
    """
    Cria DataLoaders (train, val, test) do dataset pareado.

    test_cells: filtra o split de teste por célula (impact study);
                None = todas as células.
    """
    from torch.utils.data import DataLoader

    root = Path(paired_root) if paired_root else p.PAIRED_ROOT
    if not paired_root_exists(root):
        raise FileNotFoundError(
            f"Dataset pareado não encontrado em {root}. "
            "Gere com: python -m src.preprocessing.build_paired_dataset"
        )

    def _make(split: str, transform, cells=None):
        return PairedSurveillanceDataset(
            split=split,
            paired_root=str(root),
            cells=cells,
            num_frames=num_frames,
            window_size=window_size,
            video_mode=video_mode,
            pose_mode=pose_mode,
            transform=transform,
            seed=seed,
        )

    train_dataset = _make("train", train_transform)
    val_dataset = _make("val", val_transform)
    test_dataset = _make("test", val_transform, cells=test_cells)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader
