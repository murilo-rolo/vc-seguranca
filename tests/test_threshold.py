"""Testes da calibração de limiar (threshold) e sua aplicação na avaliação."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.inference import collect_predictions
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.threshold import (
    DEFAULT_THRESHOLD,
    calibrate_threshold,
    calibrate_threshold_per_subset,
)


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


def test_calibrate_spec_constrained_prefers_recall():
    # spec_target alto troca recall por margem de spec; alvo baixo permite
    # o ponto de max recall com spec ≥ alvo (critério D4b).
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.95, 0.5, 0.6, 0.7, 0.8, 0.96]

    strict = calibrate_threshold(y_true, scores, spec_target=0.9)
    assert strict["criterion"] == "spec_constrained"
    assert strict["spec"] == pytest.approx(1.0)
    assert strict["recall"] == pytest.approx(0.2)
    assert 0.95 < strict["threshold"] <= 0.96

    loose = calibrate_threshold(y_true, scores, spec_target=0.7)
    assert loose["criterion"] == "spec_constrained"
    assert loose["recall"] == pytest.approx(1.0)
    assert loose["spec"] == pytest.approx(0.8)
    assert 0.4 < loose["threshold"] <= 0.5


def test_calibrate_spec_target_infeasible_fallback_maximin():
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.95, 0.5, 0.6, 0.7, 0.8, 0.96]
    result = calibrate_threshold(y_true, scores, spec_target=1.01)
    assert result["criterion"] == "maximin"
    assert min(result["recall"], result["spec"]) == pytest.approx(0.8)
    assert 0.4 < result["threshold"] <= 0.5


# ---------------------------------------------------------------------------
# calibrate_threshold_per_subset (D4c)
# ---------------------------------------------------------------------------


def _hard_subsets():
    """Dois subsets: A perfeito e B com spec baixa na faixa de recall 1.

    No t ≈ 0.71 o pooled tem spec 0.9 (mascarado pelo A), mas o B tem só
    spec 0.70 — exatamente o problema da Fase F (val limpo fácil esconde
    o val_swap no critério poolado).
    """
    y_a = [0] * 20 + [1] * 20
    s_a = [0.1] * 20 + [0.9] * 20
    y_b = [0] * 10 + [1] * 10
    s_b = [0.7] * 7 + [0.8, 0.9, 0.95] + [0.75] * 8 + [0.96, 0.97]
    return (y_a, s_a), (y_b, s_b)


def test_per_subset_forca_spec_no_subset_dificil():
    subset_a, subset_b = _hard_subsets()
    result = calibrate_threshold_per_subset(
        [subset_a, subset_b], spec_target=0.73,
    )
    assert result["criterion"] == "per_subset_spec_constrained"
    # Ambos os subsets cumprem a constraint no limiar escolhido.
    for subset in result["per_subset"]:
        assert subset["spec"] >= 0.73
    # O subset difícil dita o recall mínimo (sobe o t p/ proteger o spec);
    # t fica em (0.8, 0.9] — acima de 0.9 o recall do A (pos=0.9) cai a 0.
    assert result["min_recall"] == pytest.approx(0.2)
    assert 0.80 < result["threshold"] <= 0.90

    # O pooled NO MESMO DADOS escolhe t mais baixo e deixa o difícil
    # abaixo do alvo — é o bug que o per-subset corrige.
    y_pool = list(subset_a[0]) + list(subset_b[0])
    s_pool = list(subset_a[1]) + list(subset_b[1])
    pooled = calibrate_threshold(y_pool, s_pool, spec_target=0.73)
    assert pooled["threshold"] < 0.80
    y_b = np.asarray(subset_b[0])
    spec_b = float(((np.asarray(subset_b[1]) < pooled["threshold"]) & (y_b == 0)).sum()) / 10
    assert spec_b < 0.73  # pooled mascara; per-subset não


def test_per_subset_fallback_maximin_quando_inviavel():
    subset_a, subset_b = _hard_subsets()
    result = calibrate_threshold_per_subset(
        [subset_a, subset_b], spec_target=1.5,
    )
    assert result["criterion"] == "per_subset_maximin"
    # Melhor ponto balanceado: t ∈ (0.7, 0.75] → min spec 0.7/recall 1.
    assert 0.70 < result["threshold"] <= 0.75
    assert min(result["min_recall"], result["min_spec"]) == pytest.approx(0.7)


def test_per_subset_single_class_exclui_metrica():
    subset_a, subset_b = _hard_subsets()
    # Subset C sem negativos: spec None, não participa da constraint.
    y_c = [1, 1, 1, 1]
    s_c = [0.6, 0.7, 0.8, 0.9]
    result = calibrate_threshold_per_subset(
        [subset_a, subset_b, (y_c, s_c)], spec_target=0.73,
    )
    assert result["per_subset"][2]["spec"] is None
    assert result["per_subset"][2]["recall"] is not None
    for subset in result["per_subset"][:2]:
        assert subset["spec"] >= 0.73


def test_per_subset_rejeita_todos_single_class():
    with pytest.raises(ValueError, match="single-class"):
        calibrate_threshold_per_subset([([1, 1], [0.2, 0.9])])


def test_per_subset_rejeita_subset_vazio():
    with pytest.raises(ValueError, match="vazio"):
        calibrate_threshold_per_subset([([], []), ([0, 1], [0.2, 0.9])])


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
