"""
DataLoaders de teste para avaliação.

Um ponto único para todos os loaders usados por run_evaluation e
run_cross_label_evaluation (sub-modelos, cnn3d, multimodal/paired).
"""

from typing import Tuple

from torch.utils.data import DataLoader

from src import paths as p
from src.datasets.video3d_dataset import get_rwf2000_3d_dataloaders
from src.pose.pose_dataset import get_pose_dataloaders

BINARY_CLASS_NAMES = ["Non-Violent", "Violent"]


def get_per_model_test_loader(
    model_type: str,
    batch_size: int,
) -> Tuple[DataLoader, list, int]:
    """
    Retorna (test_loader, class_names, num_classes) para um sub-modelo.

    Suporta: video (clipes 3D RWF-2000), pose (keypoints flatten),
    emotion (faces do balanced-affectnet).
    """
    if model_type == "video":
        # Clipes 3D do RWF-2000 (permutação (B,T,C,H,W)->(B,C,T,H,W) na inferência)
        _, _, test_loader = get_rwf2000_3d_dataloaders(
            dataset_root=str(p.RWF2000_ROOT),
            batch_size=batch_size,
            num_frames=16,
            clip_size=(112, 112),
        )
        return test_loader, list(BINARY_CLASS_NAMES), 2

    if model_type == "pose":
        # Sequências de keypoints flatten (B, T, 99) + label binário
        _, _, test_loader = get_pose_dataloaders(
            pose_data_root=str(p.POSE_ROOT),
            batch_size=batch_size,
            window_size=16,
            flatten=True,
            num_workers=0,
        )
        return test_loader, list(BINARY_CLASS_NAMES), 2

    if model_type == "emotion":
        # Imagens de face (B, 3, 224, 224) + label de 2 classes (balanced-affectnet)
        from src.datasets.affectnet_dataset import AffectNetDataset, get_transforms

        data_root = p.DATASET_ROOT / "balanced-affectnet"
        if not data_root.exists():
            raise FileNotFoundError(
                f"Dataset de emoção não encontrado em {data_root}. "
                "Baixe o balanced-affectnet (F3) antes de avaliar o sub-modelo de emoção."
            )
        dataset = AffectNetDataset(
            data_root, split="test", transform=get_transforms(is_train=False)
        )
        if len(dataset) == 0:
            raise RuntimeError(
                f"Nenhuma imagem de teste em {data_root / 'test'}. "
                "Baixe o balanced-affectnet (F3) antes de avaliar o sub-modelo de emoção."
            )
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return test_loader, list(dataset.class_names), len(dataset.class_names)

    raise ValueError(f"Sub-modelo não suportado: {model_type}")


def get_full_model_test_loader(model_type: str, batch_size: int) -> DataLoader:
    """Retorna o dataloader de teste dos modelos completos (multimodal/cnn3d)."""
    if model_type == "multimodal":
        from src.datasets.multimodal_dataset import get_multimodal_dataloaders
        from src.datasets.paired_dataset import (
            get_paired_dataloaders,
            paired_root_exists,
        )

        if paired_root_exists():
            print(f"✓ Dataset pareado: {p.PAIRED_ROOT}")
            _, _, test_loader = get_paired_dataloaders(
                batch_size=batch_size,
                num_frames=16,
                window_size=16,
                video_mode="frames",
                pose_mode="keypoints",
            )
        else:
            print("⚠️  dataset/paired não encontrado — usando CSV legado "
                  "(gerar com: python -m src.preprocessing.build_paired_dataset)")
            _, _, test_loader = get_multimodal_dataloaders(
                video_data_root=str(p.PROCESSED_ROOT),
                pose_data_root=str(p.POSE_ROOT),
                emotion_data_root=str(p.EMOTION_ROOT),
                batch_size=batch_size,
                num_frames=16,
                window_size=16,
                video_mode="frames",
                pose_mode="keypoints",
                index_csv=str(p.PIPELINE_CSV_PATH) if p.PIPELINE_CSV_PATH.exists() else None,
            )
        return test_loader

    if model_type == "cnn3d":
        # Clipes 3D do RWF-2000 em formato frame-last (B, T, C, H, W);
        # a permutação para (B, C, T, H, W) é feita na inferência.
        _, _, test_loader = get_rwf2000_3d_dataloaders(
            dataset_root=str(p.RWF2000_ROOT),
            batch_size=batch_size,
            num_frames=16,
            clip_size=(112, 112),
        )
        return test_loader

    raise ValueError(f"Modelo completo não suportado: {model_type}")
