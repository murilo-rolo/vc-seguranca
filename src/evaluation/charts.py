"""
Geração padronizada de gráficos a partir de dicionários de métricas (F5).

Função principal: `generate_report(metrics_by_model, out_dir)`.

Produz, em `results/charts/`:
- `accuracy_f1_bar.png`: barras de accuracy e F1 (macro) por modelo (uma figura).
- `confusion_<model>.png`: heatmap da confusion matrix por modelo (quando disponível).
- `roc_pr_curves.png`: curvas ROC e Precision-Recall em eixos compartilhados
  (quando curvas são fornecidas).

Estrutura de `metrics_by_model`: `{nome_do_modelo: dict_de_metricas}` onde cada
dict segue o formato de `MetricsCalculator` (ex: `accuracy`, `f1_score`,
`confusion_matrix_array`, `auc_roc`, `auc_pr`).

Estrutura opcional de `roc_pr`: `{nome_do_modelo: curva}` ou
`{nome_do_modelo: [curva, ...]}` (multiclasse OvR) onde cada curva é:
`{"fpr": [...], "tpr": [...], "precision": [...], "recall": [...],
  "auc": float|None, "ap": float|None, "label": str}`.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bar_value(metrics: Dict, key: str, which: str = "macro"):
    """Extrai um valor escalar de um dict de métricas (ou de um dict por classe)."""
    value = metrics.get(key)
    if value is None:
        return None
    if isinstance(value, dict):
        if which in value:
            return value[which]
        return value.get("weighted")
    return value


def _save(fig, out_dir: Path, name: str) -> Path:
    """Salva a figura e devolve o caminho do arquivo criado."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_accuracy_f1_bar(metrics_by_model: Dict[str, Dict], out_dir: Path) -> Path:
    """Barras de accuracy e F1 (macro) para todos os modelos em uma figura."""
    models = list(metrics_by_model.keys())
    accs = [_bar_value(m, "accuracy") for m in metrics_by_model.values()]
    f1s = [_bar_value(m, "f1_score") for m in metrics_by_model.values()]

    fig, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(models)), 5.0))
    x = np.arange(len(models))
    width = 0.38

    acc_bars = ax.bar(x - width / 2, [a if a is not None else 0.0 for a in accs],
                      width, label="Accuracy", color="#4C72B0")
    f1_bars = ax.bar(x + width / 2, [f if f is not None else 0.0 for f in f1s],
                     width, label="F1 (macro)", color="#DD8452")

    for bars, vals in ((acc_bars, accs), (f1_bars, f1s)):
        for bar, v in zip(bars, vals):
            if v is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Métricas por modelo (sub-modelos avaliados independentemente)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, out_dir, "accuracy_f1_bar.png")


def _plot_confusion_matrices(
    metrics_by_model: Dict[str, Dict],
    out_dir: Path,
    class_names_by_model: Optional[Dict[str, List[str]]] = None
) -> List[Path]:
    """Heatmap da confusion matrix por modelo (quando o array está presente)."""
    written: List[Path] = []
    for model_name, metrics in metrics_by_model.items():
        cm = metrics.get("confusion_matrix_array")
        if cm is None:
            continue
        cm = np.asarray(cm)
        n = cm.shape[0]
        labels = (class_names_by_model or {}).get(model_name) or [
            f"C{i}" for i in range(n)
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {model_name}")
        written.append(_save(fig, out_dir, f"confusion_{model_name}.png"))
    return written


def _plot_roc_pr(
    roc_pr: Dict[str, Dict],
    out_dir: Path
) -> Path:
    """Curvas ROC e PR em eixos compartilhados para todos os modelos fornecidos."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for model_name, curves in roc_pr.items():
        if isinstance(curves, dict):
            curves = [curves]
        for curve in curves:
            label = f"{model_name} · {curve.get('label', '')}".rstrip(" ·")

            fpr = curve.get("fpr", [])
            tpr = curve.get("tpr", [])
            if fpr and tpr:
                auc = curve.get("auc")
                ax_roc.plot(fpr, tpr, label=f"{label}" + (f" (AUC={auc:.3f})" if auc is not None else ""))

            precision = curve.get("precision", [])
            recall = curve.get("recall", [])
            if precision and recall:
                ap = curve.get("ap")
                ax_pr.plot(recall, precision, label=f"{label}" + (f" (AP={ap:.3f})" if ap is not None else ""))

    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC")
    ax_roc.legend(fontsize=7, loc="lower right")
    ax_roc.grid(True, alpha=0.3)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_ylim(0, 1.05)
    ax_pr.set_title("Precision-Recall")
    ax_pr.legend(fontsize=7, loc="lower left")
    ax_pr.grid(True, alpha=0.3)

    return _save(fig, out_dir, "roc_pr_curves.png")


def generate_report(
    metrics_by_model: Dict[str, Dict],
    out_dir: str,
    class_names_by_model: Optional[Dict[str, List[str]]] = None,
    roc_pr: Optional[Dict[str, Dict]] = None
) -> List[str]:
    """
    Gera o relatório gráfico padrão a partir de dicts de métricas.

    Args:
        metrics_by_model: {nome_do_modelo: dict_de_metricas}
        out_dir: Diretório de saída (ex: "results/charts")
        class_names_by_model: {nome_do_modelo: [nomes das classes]} (opcional)
        roc_pr: {nome_do_modelo: curva | [curvas]} (opcional)

    Returns:
        Lista com os caminhos dos arquivos PNG gerados.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if metrics_by_model:
        written.append(_plot_accuracy_f1_bar(metrics_by_model, out))
        written.extend(_plot_confusion_matrices(metrics_by_model, out, class_names_by_model))

    if roc_pr:
        written.append(_plot_roc_pr(roc_pr, out))

    return [str(w) for w in written]
