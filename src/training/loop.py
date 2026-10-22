"""
Loop de treinamento compartilhado pelos scripts train_*.py.

Centraliza: épocas (treino+validação), scheduler, histórico, checkpoint do
melhor modelo (com a config efetiva gravada dentro), early stopping e
persistência do training_history.json.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

from .utils import run_epoch

HISTORY_KEYS = ("train_loss", "train_acc", "val_loss", "val_acc")


def train_loop(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    *,
    criterion,
    optimizer,
    device,
    epochs: int,
    output_dir: Path,
    experiments_dir: Path,
    scheduler=None,
    config: Optional[Dict[str, Any]] = None,
    model_hook: Optional[Callable] = None,
    scaler=None,
    grad_clip: Optional[float] = None,
    return_metrics: bool = False,
    early_stop_patience: Optional[int] = None,
    start_epoch: int = 1,
    best_val_acc: float = -1.0,
    checkpoint_extra: Optional[Dict[str, Any]] = None,
    checkpoint_model: Optional[torch.nn.Module] = None,
    monitor_loader=None,
    on_epoch_start: Optional[Callable[[int], None]] = None,
) -> Tuple[float, Dict[str, list]]:
    """
    Executa o treino e retorna (best_score, history).

    - Salva `output_dir/best_model.pth` quando a val accuracy melhora
      (checkpoint inclui a config efetiva em 'hyperparameters'). `checkpoint_model`
      permite gravar os pesos de um sub-módulo (ex.: só a fusão multimodal).
    - monitor_loader (opcional): val alternativo avaliado a cada época (ex.:
      val com emotion-swap forçado). Quando presente, o critério de best
      checkpoint e de early stopping passa a ser
      monitor_score = (val_acc + monitor_acc)/2 — selecionando modelos
      invariantes à face e não só os do val congruente. O histórico ganha
      'val_acc_swap' e 'monitor_score'; o checkpoint grava ambos.
    - Com return_metrics=True, grava também confusion_matrix.npy por época
      vencedora em `experiments_dir`.
    - on_epoch_start (opcional): hook chamado com o índice da época antes do
      treino (ex.: destravar o backbone do fine-tune em 2 estágios).
    - Ao final grava `experiments_dir/training_history.json` (inclui 'config').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {key: [] for key in HISTORY_KEYS}
    if return_metrics:
        history.update({"val_f1": [], "val_precision": [], "val_recall": []})
    if monitor_loader is not None:
        history.update({"val_acc_swap": [], "monitor_score": []})

    epochs_no_improve = 0
    metric_name = "monitor_score" if monitor_loader is not None else "val_acc"

    for epoch in range(start_epoch, epochs + 1):
        if on_epoch_start is not None:
            on_epoch_start(epoch)
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device,
            is_train=True, optimizer=optimizer, scaler=scaler,
            grad_clip=grad_clip, model_hook=model_hook,
            desc=f"Epoch {epoch} [Train]",
        )
        if return_metrics:
            val_loss, val_acc, val_metrics = run_epoch(
                model, val_loader, criterion, device,
                is_train=False, return_metrics=True, model_hook=model_hook,
                desc=f"Epoch {epoch} [Val]",
            )
            print(
                f"         ↳ F1: {val_metrics['f1_score']:.4f} | "
                f"Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f}"
            )
        else:
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, device,
                is_train=False, model_hook=model_hook,
                desc=f"Epoch {epoch} [Val]",
            )

        monitor_acc = None
        monitor_score = val_acc
        if monitor_loader is not None:
            _, monitor_acc = run_epoch(
                model, monitor_loader, criterion, device,
                is_train=False, model_hook=model_hook,
                desc=f"Epoch {epoch} [Val-swapped]",
            )
            monitor_score = (val_acc + monitor_acc) / 2
            print(
                f"         ↳ Val-swapped: {monitor_acc:.2f}% | "
                f"Monitor: {monitor_score:.2f}%"
            )

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if monitor_loader is not None:
            history['val_acc_swap'].append(monitor_acc)
            history['monitor_score'].append(monitor_score)
        if return_metrics:
            history['val_f1'].append(val_metrics['f1_score'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])

        if monitor_score > best_val_acc:
            best_val_acc = monitor_score
            epochs_no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': (checkpoint_model or model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'hyperparameters': config or {},
            }
            if monitor_loader is not None:
                checkpoint['val_acc_swap'] = monitor_acc
                checkpoint['monitor_score'] = monitor_score
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            if checkpoint_extra:
                checkpoint.update(checkpoint_extra)
            torch.save(checkpoint, output_dir / 'best_model.pth')

            if return_metrics:
                cm = np.array(val_metrics['confusion_matrix'])
                np.save(experiments_dir / 'confusion_matrix.npy', cm)
            if monitor_loader is not None:
                print(
                    f"\n✓ Melhor modelo salvo! Val: {val_acc:.2f}% | "
                    f"Swap: {monitor_acc:.2f}% | Monitor: {monitor_score:.2f}%"
                )
            else:
                print(f"\n✓ Melhor modelo salvo! Val Acc: {val_acc:.2f}%")
        else:
            print()
            epochs_no_improve += 1

        if early_stop_patience is not None and epochs_no_improve >= early_stop_patience:
            print(
                f"\n⏹ Early stopping na época {epoch} "
                f"({metric_name} não melhorou por {early_stop_patience} épocas)"
            )
            break

    history['config'] = config or {}
    with open(experiments_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return best_val_acc, history
