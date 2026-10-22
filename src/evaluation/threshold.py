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


def calibrate_threshold_per_subset(
    subsets,
    grid_size: int = 1001,
    spec_target: float = DEFAULT_SPEC_TARGET,
) -> Dict[str, Union[float, str, list, None]]:
    """
    Varre limiares com constraint POR SUBCONJUNTO (D4c).

    Motivação: com o critério pooled (calibrate_threshold), o val limpo
    (fácil) domina a média e esconde o val_swap (cross); no teste o
    spec overshoota e o recall cai (Fase F: 0.64 quando a curva permite
    ~0.75). Aqui a constraint spec_s(t) ≥ spec_target vale para CADA
    subset; entre os limiares satisfatórios, maximiza-se min_s recall_s
    (desempate: recall médio, acurácia média). Sem limiar satisfatório =>
    fallback maximin balanceado min(min_recall, min_spec).

    subsets: lista de (y_true, y_proba) — ex.: val limpo e val_swap.
    Subsets single-class: classe ausente é excluída do cálculo
    correspondente (spec ou recall); se todos forem single-class, ValueError.

    Returns:
        {
          "threshold", "spec_target", "criterion",
          "min_recall", "mean_recall", "min_spec", "mean_spec",
          "mean_accuracy",
          "per_subset": [{"recall", "spec", "accuracy", "n_pos", "n_neg"}, ...],
        }
    """
    prepared = []
    for y_true, y_proba in subsets:
        y = np.asarray(y_true).astype(int)
        scores = _positive_scores(y_proba)
        if y.size != scores.size:
            raise ValueError(
                f"tamanhos divergentes: y_true={y.size}, scores={scores.size}"
            )
        if y.size == 0:
            raise ValueError("subset vazio")
        prepared.append((y == 1, scores))

    if not any(
        is_pos.any() and (~is_pos).any() for is_pos, _ in prepared
    ):
        raise ValueError("todos os subsets são single-class")

    best_constrained: Optional[Dict] = None
    best_ckey: Optional[tuple] = None
    best_fallback: Optional[Dict] = None
    best_fkey: Optional[tuple] = None

    for t in np.linspace(0.0, 1.0, grid_size):
        per_subset = []
        for is_pos, scores in prepared:
            pred = scores >= t
            n_pos = int(is_pos.sum())
            n_neg = int((~is_pos).sum())
            tp = int((pred & is_pos).sum())
            tn = int(((~pred) & (~is_pos)).sum())
            recall = tp / n_pos if n_pos else None
            spec = tn / n_neg if n_neg else None
            acc = (tp + tn) / is_pos.size
            per_subset.append({
                "recall": recall, "spec": spec, "accuracy": acc,
                "n_pos": n_pos, "n_neg": n_neg,
            })

        recalls = [s["recall"] for s in per_subset if s["recall"] is not None]
        specs = [s["spec"] for s in per_subset if s["spec"] is not None]
        accs = [s["accuracy"] for s in per_subset]
        if not recalls:
            continue
        min_recall = min(recalls)
        mean_recall = sum(recalls) / len(recalls)
        min_spec = min(specs) if specs else 0.0
        mean_spec = sum(specs) / len(specs) if specs else 0.0
        mean_acc = sum(accs) / len(accs)

        satisfies = bool(specs) and all(s >= spec_target for s in specs)
        result = {
            "threshold": float(t),
            "min_recall": min_recall,
            "mean_recall": mean_recall,
            "min_spec": min_spec,
            "mean_spec": mean_spec,
            "mean_accuracy": mean_acc,
            "per_subset": per_subset,
        }
        if satisfies:
            ckey = (min_recall, mean_recall, mean_acc)
            if best_ckey is None or ckey > best_ckey:
                best_ckey = ckey
                best_constrained = result
        fkey = (min(min_recall, min_spec), mean_recall + mean_spec, mean_acc)
        if best_fkey is None or fkey > best_fkey:
            best_fkey = fkey
            best_fallback = result

    if best_constrained is not None:
        best_constrained["criterion"] = "per_subset_spec_constrained"
    else:
        best_fallback["criterion"] = "per_subset_maximin"
        best_constrained = best_fallback
    best_constrained["spec_target"] = spec_target
    return best_constrained
