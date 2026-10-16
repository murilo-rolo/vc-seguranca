"""Utilitários de tensores compartilhados entre treino e avaliação."""

from typing import Any

import torch


def move_to_device(x: Any, device: torch.device) -> Any:
    """Move recursivamente tensores (inclusive dentro de listas/tuplas) para o device."""
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(item, device) for item in x)
    return x
