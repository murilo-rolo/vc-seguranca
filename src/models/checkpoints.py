"""
Carregamento flexível de checkpoints dos modelos.

Os checkpoints do projeto podem estar em três formatos:
- dict com 'model_state_dict' (checkpoint completo de treino)
- dict com 'state_dict' (formato alternativo)
- state_dict puro (sem wrapper)
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extrai o state_dict de um objeto de checkpoint em qualquer formato."""
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_state(
    model: nn.Module,
    checkpoint_path: str,
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Carrega pesos de um checkpoint em `model`.

    Returns:
        O objeto checkpoint completo (dict) ou None se o arquivo não existir.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_state_dict(checkpoint))
    return checkpoint if isinstance(checkpoint, dict) else None
