"""
Módulo de datasets para o projeto de detecção de violência em vídeos.
"""

from .surveillance_dataset import SurveillanceRiskDataset, get_dataloaders
from .multimodal_dataset import MultimodalSurveillanceDataset, get_multimodal_dataloaders
from .video3d_dataset import (
    RWF2000Video3DDataset,
    get_rwf2000_3d_dataloaders
)
from .paired_dataset import (
    PairedSurveillanceDataset,
    get_paired_dataloaders,
    paired_root_exists,
)
from .affectnet_dataset import (
    AffectNetDataset,
    get_transforms,
    resolve_affectnet_root,
    resolve_affectnet_splits,
)
from src.preprocessing import load_index

__all__ = [
    "SurveillanceRiskDataset",
    "get_dataloaders",
    "MultimodalSurveillanceDataset",
    "get_multimodal_dataloaders",
    "RWF2000Video3DDataset",
    "get_rwf2000_3d_dataloaders",
    "PairedSurveillanceDataset",
    "get_paired_dataloaders",
    "paired_root_exists",
    "AffectNetDataset",
    "get_transforms",
    "resolve_affectnet_root",
    "resolve_affectnet_splits",
    "load_index",
]
