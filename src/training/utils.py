"""
Funções compartilhadas para treinamento de modelos.

Centraliza o loop de treino/validação (run_epoch) e a criação de
DataLoaders para evitar duplicação entre scripts de treinamento.
"""

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from src.tensor_utils import move_to_device


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_train: bool = True,
    optimizer: Optional[optim.Optimizer] = None,
    desc: str = "Epoch",
    model_hook: Optional[Callable] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    return_metrics: bool = False,
    grad_clip: Optional[float] = None,
):
    """
    Executa uma época de treino ou validação.

    Args:
        model_hook: pré-processa o batch e retorna (inputs, labels).
        scaler: GradScaler para mixed precision.
        return_metrics: retorna também dict com f1/precision/recall (macro)
            e confusion_matrix (lista de listas).
        grad_clip: norma máxima dos gradientes (None/0 = desativado).

    Returns:
        (loss, accuracy) ou (loss, accuracy, metrics).
    """
    model.train(is_train)

    running_loss, correct, total = 0.0, 0, 0
    all_labels, all_preds = [], []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            if model_hook:
                inputs, labels = model_hook(batch)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0:-1], batch[-1]
            else:
                raise ValueError(
                    "Batch não reconhecido. Use model_hook para extrair inputs/labels."
                )

            inputs = move_to_device(inputs, device)
            labels = move_to_device(labels, device)

            if is_train:
                optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if grad_clip is not None and grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

            pbar.set_postfix(
                loss=f"{running_loss / (pbar.n + 1):.4f}",
                acc=f"{100.0 * correct / total:.2f}%",
            )

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    if not return_metrics:
        return epoch_loss, epoch_acc

    from sklearn.metrics import (
        confusion_matrix, f1_score, precision_score, recall_score,
    )
    labels_idx = list(range(max(max(all_labels), max(all_preds)) + 1))
    metrics = {
        'f1_score': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'precision': float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=labels_idx).tolist(),
    }
    return epoch_loss, epoch_acc, metrics


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    sampler: Optional[Sampler] = None,
) -> DataLoader:
    """Cria um DataLoader com configurações padronizadas."""
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
