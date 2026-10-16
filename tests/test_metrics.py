"""Testes das métricas com dados sintéticos (sem datasets reais)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import MetricsCalculator, calculate_metrics

BINARY_CLASS_NAMES = ["Non-Violent", "Violent"]


def _binario(perfect: bool):
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=64)
    if perfect:
        y_pred = y_true.copy()
        y_proba = np.eye(2)[y_true][:, 1]
    else:
        y_pred = 1 - y_true
        y_proba = rng.random(64)
    return y_true, y_pred, y_proba


def test_calculate_metrics_perfect():
    y_true, y_pred, y_proba = _binario(perfect=True)
    m = calculate_metrics(y_true, y_pred, y_proba, BINARY_CLASS_NAMES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["f1_score"]["macro"] == pytest.approx(1.0)
    assert m["confusion_matrix"]["true_positive"] + m["confusion_matrix"]["true_negative"] == 64
    assert m["auc_roc"] == pytest.approx(1.0)


def test_calculate_metrics_single_class_present():
    # Células do impact study podem conter uma única classe.
    y_true = np.zeros(16, dtype=int)
    y_pred = np.zeros(16, dtype=int)
    y_proba = np.zeros(16)
    m = calculate_metrics(y_true, y_pred, y_proba, BINARY_CLASS_NAMES)
    assert m["accuracy"] == pytest.approx(1.0)
    assert "Non-Violent" in m["precision"]


class _DummyBinaryModel(torch.nn.Module):
    """Modelo que 'aprende' a identidade: logit alto quando x>0."""

    def forward(self, x):
        # x: (B, 1) — usa o valor como logit da classe positiva
        pos = x.view(-1, 1)
        return torch.cat([-pos, pos], dim=1)


def _loader_com_labels(perfect: bool = True):
    rng = np.random.default_rng(1)
    labels = torch.from_numpy(rng.integers(0, 2, size=32)).float()
    if not perfect:
        labels = 1 - labels
    inputs = labels.view(-1, 1) * 2 - 1  # classe 0 -> -1, classe 1 -> +1
    return DataLoader(TensorDataset(inputs, labels.long()), batch_size=8)


def test_metrics_calculator_binary():
    model = _DummyBinaryModel().eval()
    calc = MetricsCalculator(
        model, _loader_com_labels(), device="cpu",
        class_names=BINARY_CLASS_NAMES, num_classes=2,
    )
    metrics, y_true, y_pred, y_proba = calc.evaluate()
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert y_true.shape == y_pred.shape == y_proba.shape == (32,)
    assert set(np.unique(y_pred)) <= {0, 1}


class _DummyMultiModel(torch.nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n = n_classes

    def forward(self, x):
        # x: (B,) com o label alvo embutido — logit 1 na classe certa
        B = x.shape[0]
        logits = torch.zeros(B, self.n)
        logits[torch.arange(B), x.long()] = 10.0
        return logits


def test_metrics_calculator_multiclass():
    labels = torch.arange(3).repeat(4)  # 12 amostras, 3 classes
    inputs = labels.float()
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4)
    model = _DummyMultiModel(3).eval()
    calc = MetricsCalculator(
        model, loader, device="cpu",
        class_names=["a", "b", "c"], num_classes=3,
    )
    metrics, y_true, y_pred, y_proba = calc.evaluate()
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["num_classes"] == 3
    assert y_proba.shape == (12, 3)  # multiclasse mantém matriz completa
