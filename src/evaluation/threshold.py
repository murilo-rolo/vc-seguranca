"""
Calibração do limiar de decisão (classificação binária).

Motivação (impact study cross-label): com AUC alto mas ponto de operação
enviesado, o limiar 0.5 (argmax) não satisfaz simultaneamente recall e
especificidade nas células do teste.

Critério (Fase F/D4b): maximizar recall sujeito a spec ≥ spec_target no
conjunto de calibração (val [+ val_swap]). O alvo de spec com margem
(0.73 > meta 0.70) compensa o shift val→teste observado na Fase E, onde o
critério maximin puro escolheu spec 0.77/recall 0.68 no teste embora a
curva tenha pontos com recall 0.76/spec 0.75. Fallback: maximin
(min(recall, spec)) quando nenhum limiar atende a constraint.
"""

from typing import Dict, Optional, Union

import numpy as np

DEFAULT_THRESHOLD = 0.5
DEFAULT_SPEC_TARGET = 0.73

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
    spec_target: float = DEFAULT_SPEC_TARGET,
) -> Dict[str, Optional[float]]:
    """
    Varre limiares em [0, 1] sobre a probabilidade da classe positiva.

    Escolhe t que maximiza recall sujeito a spec ≥ spec_target; desempate
    por spec e acurácia. Se nenhum t satisfaz a constraint, faz fallback
    para max min(recall, spec) (desempate por Youden J e acurácia).
    y_true single-class => fallback 0.5 (recall/spec indefinidos).

    Returns:
        {"threshold", "recall", "spec", "accuracy", "criterion", "spec_target"}
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
            "criterion": None,
            "spec_target": spec_target,
        }

    constrained_key: Optional[tuple] = None
    constrained: Dict[str, Optional[float]] = {}
    maximin_key: Optional[tuple] = None
    maximin: Dict[str, Optional[float]] = {}

    for t in np.linspace(0.0, 1.0, grid_size):
        pred = scores >= t
        tp = int((pred & is_pos).sum())
        tn = int(((~pred) & (~is_pos)).sum())
        recall = tp / n_pos
        spec = tn / n_neg
        acc = (tp + tn) / y.size

        mkey = (min(recall, spec), recall + spec - 1.0, acc)
        if maximin_key is None or mkey > maximin_key:
            maximin_key = mkey
            maximin = {
                "threshold": float(t),
                "recall": recall,
                "spec": spec,
                "accuracy": acc,
            }

        if spec >= spec_target:
            ckey = (recall, spec, acc)
            if constrained_key is None or ckey > constrained_key:
                constrained_key = ckey
                constrained = {
                    "threshold": float(t),
                    "recall": recall,
                    "spec": spec,
                    "accuracy": acc,
                }

    if constrained:
        constrained["criterion"] = "spec_constrained"
    else:
        maximin["criterion"] = "maximin"
        constrained = maximin
    constrained["spec_target"] = spec_target
    return constrained
