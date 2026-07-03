"""
Funções compartilhadas para treinamento de modelos.

Centraliza loops de treino/validação e criação de DataLoaders
para evitar duplicação entre scripts de treinamento.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional, Callable


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
        model: Modelo a ser treinado/validado
        loader: DataLoader
        criterion: Função de loss
        device: Device (cuda/cpu)
        is_train: True para treino, False para validação
        optimizer: Otimizador (obrigatório se is_train=True)
        desc: Descrição para a barra de progresso
        model_hook: Função opcional para pré-processar batch antes do forward.
                    Recebe (batch) e retorna (inputs, labels).
        scaler: GradScaler para mixed precision (opcional)
        return_metrics: Se True, retorna também dict com F1, precision, recall, confusion_matrix
        grad_clip: Valor máximo da norma L2 dos gradientes (opcional, ex: 1.0)

    Returns:
        Se return_metrics=False: Tupla (loss_média, accuracy)
        Se return_metrics=True: Tupla (loss_média, accuracy, metrics_dict)
    """
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.set_grad_enabled(is_train):
        pbar = tqdm(loader, desc=desc)
        for batch in pbar:
            if model_hook:
                inputs, labels = model_hook(batch)
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                raise ValueError(
                    "Batch não reconhecido. Use model_hook para extrair inputs/labels."
                )

            inputs = inputs.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if is_train:
                scaler.scale(loss).backward()
                if grad_clip is not None and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(predicted.cpu().tolist())

            current_loss = running_loss / (pbar.n + 1)
            current_acc = 100.0 * correct / total
            pbar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.2f}%")

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total

    if return_metrics:
        from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
        metrics = {
            'f1_score': float(f1_score(all_labels, all_preds, average='macro')),
            'precision': float(precision_score(all_labels, all_preds, average='macro')),
            'recall': float(recall_score(all_labels, all_preds, average='macro')),
            'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=range(8)).tolist(),
        }
        return epoch_loss, epoch_acc, metrics

    return epoch_loss, epoch_acc


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


def save_checkpoint(
    state: dict,
    output_path: Path,
    filename: str = "best_model.pth",
    history: Optional[dict] = None,
):
    """Salva checkpoint do modelo e opcionalmente histórico."""
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / filename
    torch.save(state, model_path)
    print(f"  Modelo salvo em: {model_path}")

    if history:
        import json
        history_path = output_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)


def setup_device(device: Optional[str] = None) -> torch.device:
    """Configura e retorna o device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def set_seed(seed: int = 42):
    """Configura seed para reprodutibilidade."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
