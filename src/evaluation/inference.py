"""
Inferência compartilhada pelos módulos de avaliação.

O mesmo ciclo (separar inputs/label do batch → mover para device →
forward → softmax/argmax) estava duplicado em metrics, robustness,
limitations, performance e nos scripts de avaliação. Aqui fica a única
implementação.
"""

from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch

from src.tensor_utils import move_to_device

PrepareFn = Optional[Callable[[Any], Any]]


def unpack_batch(batch: Any) -> Tuple[Any, Optional[torch.Tensor]]:
    """Separa (inputs, labels) de um batch. Sem label => labels=None."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0:-1], batch[-1]
    return batch, None


def forward_batch(model: torch.nn.Module, inputs: Any) -> torch.Tensor:
    """Forward padrão: Tensor vai como argumento único, tupla/lista desempacota."""
    if isinstance(inputs, torch.Tensor):
        return model(inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    raise ValueError(f"Formato de input não suportado: {type(inputs)}")


def labels_to_numpy(labels: Any) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        return labels.cpu().numpy()
    return np.array(labels)


def collect_predictions(
    model: torch.nn.Module,
    dataloader,
    device,
    prepare_inputs: PrepareFn = None,
    keep_full_probs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Percorre o dataloader e acumula (y_true, y_pred, y_proba).

    Args:
        prepare_inputs: hook aplicado aos inputs antes do forward
            (ex.: permutação frame-last -> channel-first do CNN3D).
        keep_full_probs: True mantém a matriz completa (N, K); False guarda
            apenas a probabilidade da classe positiva (coluna 1).

    Returns:
        (y_true, y_pred, y_proba)
    """
    model.eval()
    preds, probs, labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            inputs, batch_labels = unpack_batch(batch)
            if prepare_inputs is not None:
                inputs = prepare_inputs(inputs)
            inputs = move_to_device(inputs, device)

            outputs = forward_batch(model, inputs)
            batch_probs = torch.softmax(outputs, dim=1)

            preds.append(torch.argmax(outputs, dim=1).cpu().numpy())
            probs.append(
                batch_probs.cpu().numpy()
                if keep_full_probs
                else batch_probs[:, 1].cpu().numpy()
            )
            if batch_labels is not None:
                labels.append(labels_to_numpy(batch_labels))

    if not labels:
        raise ValueError("Labels não fornecidos no dataloader")

    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)
