"""
Cálculo de métricas de avaliação para modelos de detecção de violência.

Inclui: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC, AUC-PR
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .charts import plot_confusion_matrix
from .inference import collect_predictions

BINARY_CLASS_NAMES = ["Non-Violent", "Violent"]


def _core_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Precision/recall/f1 por classe + macro/weighted + matriz de confusão."""
    labels = list(range(len(class_names)))

    precision = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    def as_dict(values, average):
        return {
            **{name: float(v) for name, v in zip(class_names, values)},
            "macro": float(average(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
            "weighted": float(average(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": as_dict(precision, precision_score),
        "recall": as_dict(recall, recall_score),
        "f1_score": as_dict(f1, f1_score),
        "confusion_matrix_array": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: List[str] = BINARY_CLASS_NAMES,
) -> Dict:
    """
    Calcula métricas binárias completas.

    Args:
        y_true: Labels verdadeiros (0 ou 1)
        y_pred: Predições (0 ou 1)
        y_proba: Probabilidades da classe positiva (opcional, para AUC)
        class_names: Nomes das classes

    Returns:
        Dicionário com todas as métricas
    """
    metrics = _core_metrics(y_true, y_pred, list(class_names))

    cm = np.asarray(metrics["confusion_matrix_array"])
    tn, fp, fn, tp = cm.ravel()
    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    metrics["confusion_matrix"] = {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    if y_proba is not None:
        _add_auc(metrics, y_true, y_proba)

    return metrics


def calculate_multiclass_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Calcula métricas multiclasse (ex: 2 classes do EmotionNet).

    Args:
        y_true: Labels verdadeiros (índices de 0 a K-1)
        y_pred: Predições (índices de 0 a K-1)
        y_proba: Matriz de probabilidades (N, K) (opcional, para AUC OvR)
        class_names: Nomes das classes (K)

    Returns:
        Dicionário com métricas (accuracy, precision/recall/f1 por classe,
        macro/weighted, confusion matrix). Mantém as mesmas chaves padrão.
    """
    n_classes = len(class_names) if class_names else int(max(y_true.max(), y_pred.max()) + 1)
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(n_classes)]

    metrics = _core_metrics(y_true, y_pred, list(class_names))
    metrics["num_classes"] = n_classes

    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == n_classes:
        try:
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError:
            pass
        try:
            metrics["auc_pr"] = float(
                average_precision_score(y_true, y_proba, average="macro")
            )
        except ValueError:
            pass

    return metrics


def _add_auc(metrics: Dict, y_true: np.ndarray, y_proba: np.ndarray) -> None:
    try:
        auc_roc = roc_auc_score(y_true, y_proba)
        if np.isfinite(auc_roc):
            metrics["auc_roc"] = float(auc_roc)
    except ValueError:
        pass
    try:
        auc_pr = average_precision_score(y_true, y_proba)
        if np.isfinite(auc_pr):
            metrics["auc_pr"] = float(auc_pr)
    except ValueError:
        pass


class MetricsCalculator:
    """
    Classe para calcular e salvar métricas de avaliação.
    """

    def __init__(
        self,
        model,
        dataloader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        class_names: List[str] = BINARY_CLASS_NAMES,
        num_classes: Optional[int] = None,
        prepare_inputs: Optional[Callable] = None,
    ):
        """
        Args:
            model: Modelo PyTorch para avaliação
            dataloader: DataLoader com dados de teste
            device: Device para inferência
            class_names: Nomes das classes
            num_classes: Número de classes. Se None, inferido de class_names
                (>2 classes = multiclass).
            prepare_inputs: hook aplicado aos inputs antes do forward
                (ex.: ensure_channels_first para CNN3D).
        """
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device(device)
        self.class_names = list(class_names)
        self.num_classes = num_classes if num_classes is not None else len(self.class_names)
        self.prepare_inputs = prepare_inputs
        self.model.eval()

    def evaluate(self) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
        """
        Avalia o modelo.

        Returns:
            (metrics, y_true, y_pred, y_proba)
        """
        y_true, y_pred, y_proba = collect_predictions(
            self.model,
            self.dataloader,
            self.device,
            prepare_inputs=self.prepare_inputs,
            keep_full_probs=self.num_classes > 2,
        )

        if self.num_classes > 2:
            metrics = calculate_multiclass_metrics(y_true, y_pred, y_proba, self.class_names)
        else:
            metrics = calculate_metrics(y_true, y_pred, y_proba, self.class_names)

        return metrics, y_true, y_pred, y_proba

    def save_results(
        self,
        output_dir: str,
        experiment_name: str,
        metrics: Dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> None:
        """Salva métricas (JSON em metrics/) e gráficos (CM, ROC, PR em charts/)."""
        import json

        output_path = Path(output_dir)
        metrics_dir = output_path / "metrics"
        charts_dir = output_path / "charts"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        charts_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = metrics_dir / f"{experiment_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        plot_confusion_matrix(
            confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes))),
            self.class_names,
            charts_dir / f"{experiment_name}_confusion_matrix.png",
            title=f"{experiment_name} Confusion Matrix",
        )

        if y_proba is not None and self.num_classes <= 2:
            self._plot_roc_curve(
                y_true, y_proba, experiment_name, charts_dir / f"{experiment_name}_roc_curve.png"
            )
            self._plot_pr_curve(
                y_true, y_proba, experiment_name, charts_dir / f"{experiment_name}_pr_curve.png"
            )

    def _plot_roc_curve(self, y_true, y_proba, experiment_name, output_path):
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{experiment_name} ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def _plot_pr_curve(self, y_true, y_proba, experiment_name, output_path):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        auc = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR Curve (AP = {auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0, 1)
        plt.title(f"{experiment_name} Precision-Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
