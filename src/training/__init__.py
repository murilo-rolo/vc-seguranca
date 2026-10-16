"""
Módulo de treinamento para o projeto de detecção de violência em vídeos.
"""

from .config import add_config_args, load_config, resolve_config
from .loop import train_loop
from .utils import create_dataloader, run_epoch

__all__ = [
    "add_config_args",
    "load_config",
    "resolve_config",
    "train_loop",
    "create_dataloader",
    "run_epoch",
]
