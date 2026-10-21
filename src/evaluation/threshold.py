"""
Calibração do limiar de decisão (classificação binária).

Motivação (impact study cross-label): com AUC alto mas ponto de operação
enviesado, o limiar 0.5 (argmax) não satisfaz simultaneamente recall e
especificidade nas células do teste. O sweep escolhe o limiar que maximiza
min(recall, spec) — critério alinhado às metas de ambas as células.
"""

from typing import Dict, Optional, Union

import numpy as np

DEFAULT_THRESHOLD = 0.5

ArrayLike = Union[np.ndarray, list]


def _positive_scores(y_proba: ArrayLike) -> np.ndarray:
    scores = np.asarray(y_proba, dtype=float)
    if scores.ndim == 2:
        if scores.shape[1] < 2:
            raise ValueError(f"y_proba (N, {scores.shape[1]}) sem classe positiva")
        scores = scores[:, 1]
    return scores


def calibrate_threshold(
    y_true: ArrayLike,
    y_proba: ArrayLike,
    grid_size: int = 1001,
) -> Dict[str, Optional[float]]:
    """
    Varre limiares em [0, 1] sobre a probabilidade da classe positiva.

    Escolhe t que maximiza min(recall, spec); desempate por Youden J e,
    em último caso, por acurácia. y_true single-class => fallback 0.5
    (recall/spec indefinidos).

    Returns:
        {"threshold", "recall", "spec", "accuracy"}
    """
    y = np.asarray(y_true).astype(int)
    scores = _positive_scores(y_proba)
    if y.size != scores.size:
        raise ValueError(f"tamanhos divergentes: y_true={y.size}, scores={scores.size}")

    is_pos = y == 1
    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "threshold": DEFAULT_THRESHOLD,
            "recall": None,
            "spec": None,
            "accuracy": None,
        }

    best_key: Optional[tuple] = None
    best: Dict[str, Optional[float]] = {}

    for t in np.linspace(0.0, 1.0, grid_size):
        pred = scores >= t
        tp = int((pred & is_pos).sum())
        tn = int(((~pred) & (~is_pos)).sum())
        recall = tp / n_pos
        spec = tn / n_neg
        acc = (tp + tn) / y.size
        key = (min(recall, spec), recall + spec - 1.0, acc)
        if best_key is None or key > best_key:
            best_key = key
            best = {
                "threshold": float(t),
                "recall": recall,
                "spec": spec,
                "accuracy": acc,
            }

    return best
