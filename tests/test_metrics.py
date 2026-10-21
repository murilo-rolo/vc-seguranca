"""Testes das métricas com dados sintéticos (sem datasets reais)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.inference import collect_predictions
from src.evaluation.metrics import MetricsCalculator, calculate_metrics
from src.models.cnn3d_risk import ensure_channels_first

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


class _DummyVideoModel(torch.nn.Module):
    """Recebe clipes (B, C, T, H, W); logit positivo quando os pixels são 1."""

    def forward(self, x):
        assert x.dim() == 5 and x.shape[1] == 3, \
            f"esperado (B,C,T,H,W), veio {tuple(x.shape)}"
        s = x.mean(dim=(1, 2, 3, 4))
        return torch.stack([-s, s], dim=1)


def _video_loader_frame_last(batch_size: int = 2):
    """Clipes frame-last (B, T, C, H, W) como o RWF-2000 3D devolve."""
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    clips = torch.zeros(len(labels), 4, 3, 8, 8)
    clips[labels == 1] = 1.0
    return DataLoader(TensorDataset(clips, labels), batch_size=batch_size)


def test_collect_predictions_prepare_inputs_receives_tensor():
    # Regressão: o hook recebe cada tensor, não o container [tensor]
    # (bug: ensure_channels_first([clips]) -> 'list' object has no attribute 'dim').
    calls = []

    def prepare(x):
        assert isinstance(x, torch.Tensor), f"hook recebeu {type(x)}"
        calls.append(tuple(x.shape))
        return ensure_channels_first(x)

    model = _DummyVideoModel().eval()
    y_true, y_pred, y_proba = collect_predictions(
        model, _video_loader_frame_last(), "cpu", prepare_inputs=prepare
    )
    assert len(calls) == 4  # um por batch (8 amostras / batch_size=2)
    assert calls[0] == (2, 4, 3, 8, 8)  # frame-last antes da permuta
    assert (y_pred == y_true).all()
    assert y_proba.shape == (8,)


def test_metrics_calculator_video_frame_last():
    model = _DummyVideoModel().eval()
    calc = MetricsCalculator(
        model, _video_loader_frame_last(), device="cpu",
        prepare_inputs=ensure_channels_first,
    )
    metrics, y_true, y_pred, y_proba = calc.evaluate()
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert y_true.shape == y_pred.shape == y_proba.shape == (8,)
