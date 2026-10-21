"""Testes da calibração de limiar (threshold) e sua aplicação na avaliação."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.inference import collect_predictions
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.threshold import DEFAULT_THRESHOLD, calibrate_threshold


# ---------------------------------------------------------------------------
# calibrate_threshold
# ---------------------------------------------------------------------------


def test_calibrate_perfect_separation():
    y_true = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.6, 0.9]
    result = calibrate_threshold(y_true, scores)
    assert result["recall"] == 1.0
    assert result["spec"] == 1.0
    assert 0.4 < result["threshold"] <= 0.6
    assert result["accuracy"] == 1.0


def test_calibrate_maximin_with_overlap():
    # Sobreposição parcial: melhor min(recall, spec) = 0.75 em t ∈ (0.4, 0.5].
    y_true = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.2, 0.3, 0.4, 0.9, 0.1, 0.5, 0.6, 0.7]
    result = calibrate_threshold(y_true, scores)
    assert min(result["recall"], result["spec"]) == pytest.approx(0.75)
    assert 0.4 < result["threshold"] <= 0.5


def test_calibrate_single_class_fallback():
    result = calibrate_threshold([1, 1, 1], [0.2, 0.6, 0.9])
    assert result["threshold"] == DEFAULT_THRESHOLD
    assert result["recall"] is None and result["spec"] is None


def test_calibrate_rejects_size_mismatch():
    with pytest.raises(ValueError, match="tamanhos divergentes"):
        calibrate_threshold([0, 1], [0.1])


# ---------------------------------------------------------------------------
# aplicação: collect_predictions / MetricsCalculator
# ---------------------------------------------------------------------------


class _FixedLogits(torch.nn.Module):
    """Logits fixos indexados pela linha de entrada.

    P(classe 1) = sigmoid(0.5)≈0.622 (linhas 0,1) ou sigmoid(1.5)≈0.818 (2,3).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "logits", torch.tensor([[0.0, 0.5], [0.0, 0.5], [0.0, 1.5], [0.0, 1.5]])
        )

    def forward(self, x):
        return self.logits[x[:, 0].long()]


def _fixed_loader():
    inputs = torch.arange(4).float().view(-1, 1)  # índice da linha de logits
    labels = torch.tensor([1, 0, 1, 0])
    return DataLoader(TensorDataset(inputs, labels), batch_size=2)


def test_collect_predictions_argmax_sem_threshold():
    _, y_pred, y_proba = collect_predictions(_FixedLogits(), _fixed_loader(), "cpu")
    # argmax sempre escolhe a classe 1 (logit maior)
    assert (y_pred == 1).all()
    assert y_proba.shape == (4,)


def test_collect_predictions_com_threshold():
    _, y_pred, _ = collect_predictions(
        _FixedLogits(), _fixed_loader(), "cpu", threshold=0.8,
    )
    # P=0.622 < 0.8 -> 0 | P=0.818 >= 0.8 -> 1
    assert list(y_pred) == [0, 0, 1, 1]


def test_metrics_calculator_aplica_threshold():
    calc = MetricsCalculator(
        _FixedLogits(), _fixed_loader(), device="cpu", threshold=0.8,
    )
    metrics, _, y_pred, _ = calc.evaluate()
    assert list(y_pred) == [0, 0, 1, 1]
    # y_true = [1,0,1,0] -> TP=1, TN=1, FP=1, FN=1
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_metrics_calculator_sem_threshold_preserva_comportamento():
    calc = MetricsCalculator(_FixedLogits(), _fixed_loader(), device="cpu")
    _, _, y_pred, _ = calc.evaluate()
    assert (y_pred == 1).all()
