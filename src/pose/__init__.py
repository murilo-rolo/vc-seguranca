"""
Módulo de Pose Estimation para detecção de keypoints em vídeos.

Este módulo fornece:
- Extração de keypoints usando Ultralytics YOLO26
- Dataset PyTorch para sequências de pose
- Utilitários para processamento de pose data
"""

from .extract_pose import extract_pose_from_video, process_videos_for_pose, process_dataset_for_pose
from .pose_dataset import PoseSequenceDataset, get_pose_dataloaders

__all__ = [
    'extract_pose_from_video',
    'process_videos_for_pose',
    'PoseSequenceDataset',
    'get_pose_dataloaders'
]

