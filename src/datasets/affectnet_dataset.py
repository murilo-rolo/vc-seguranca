"""
Dataset de emoção (AffectNet) para classificação binária violent / non_violent.

Dois layouts suportados:
- folder mode (`dataset/balanced-affectnet/`): splits train/val/test com
  pastas por classe; labels = índice das pastas ordenadas.
- CSV mode (AffectNet legado, `dataset/AffectNet/` com labels.csv + Train/Test);
  `min_confidence` filtra por coluna relFCs.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src import paths as p

AFFECTNET_CLASSES = {
    'violent': 0,
    'non_violent': 1
}


def resolve_affectnet_root(dataset_path: Optional[str] = None) -> Path:
    """Resolve a raiz do dataset de emoção (preferência: balanced-affectnet)."""
    if dataset_path is not None:
        return Path(dataset_path)
    if (p.DATASET_ROOT / "balanced-affectnet").exists():
        return p.DATASET_ROOT / "balanced-affectnet"
    return p.AFFECTNET_ROOT


def resolve_affectnet_splits(root: Path) -> Tuple[str, str]:
    """Splits (treino, validação) conforme o layout do dataset de emoção."""
    if (root / "train").is_dir():
        return "train", "val"
    return "Train", "Test"


class AffectNetDataset(Dataset):
    """Dataset binário de emoções faciais (violent / non_violent)."""

    def __init__(
        self,
        data_root: Path,
        split: str = "Train",
        transform: Optional[transforms.Compose] = None,
        min_confidence: float = 0.0,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.min_confidence = min_confidence

        csv_path = self.data_root / 'labels.csv'
        if csv_path.exists():
            self.class_to_idx = AFFECTNET_CLASSES.copy()
            self.samples = self._load_from_csv(csv_path)
            source = "CSV"
        else:
            class_names = self._derive_class_names_from_folders() or list(AFFECTNET_CLASSES.keys())
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            self.samples = self._load_from_folders()
            source = "pastas"

        self.class_names = list(self.class_to_idx.keys())
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        print(f"AffectNet {split}: {len(self.samples)} amostras carregadas ({source})")
        if len(self.samples) == 0:
            print(f"  ⚠️  Nenhuma imagem encontrada em {self.data_root / split}/")
        else:
            counts = {}
            for _, label in self.samples:
                counts[label] = counts.get(label, 0) + 1
            dist = ', '.join(f'{self.idx_to_class[k]}: {v}' for k, v in sorted(counts.items()))
            print(f"  Distribuição: {dist}")

    def _derive_class_names_from_folders(self) -> Optional[List[str]]:
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            return None
        folders = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
        return folders or None

    def _load_from_csv(self, csv_path: Path) -> List[Tuple[Path, int]]:
        samples = []
        split_dir = self.data_root / self.split
        with open(csv_path, 'r') as f:
            for row in csv.DictReader(f):
                if float(row['relFCs']) < self.min_confidence:
                    continue
                img_path = split_dir / row['pth']
                label = self.class_to_idx.get(row['label'])
                if img_path.exists() and label is not None:
                    samples.append((img_path, label))
        return samples

    def _load_from_folders(self) -> List[Tuple[Path, int]]:
        samples = []
        split_dir = self.data_root / self.split
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                class_dir = split_dir / class_name.capitalize()
            if class_dir.exists():
                for ext in ('.jpg', '.jpeg', '.png'):
                    samples.extend((img, class_idx) for img in class_dir.glob(f"*{ext}"))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Transformações ImageNet-normalized (augmentation no treino)."""
    if not is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    aug_list = [
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ]
    try:
        from torchvision.transforms import RandAugment
        aug_list.insert(0, RandAugment(num_ops=2, magnitude=9))
    except ImportError:
        pass
    aug_list += [
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(aug_list)
