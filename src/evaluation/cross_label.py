"""
Avaliação cross-label (face × vídeo) sobre o split de teste do dataset pareado.

O split `dataset/paired/test/` contém 4 células de mesmo tamanho por construção:

    violent + violent face          (congruente)
    violent + non_violent face      (incongruente)
    non_violent + non_violent face  (congruente)
    non_violent + violent face      (incongruente)

Aqui ficam os cenários, a avaliação por célula e a geração de relatório/gráficos
antes espalhados em run_cross_label_evaluation.py e run_impact_study.py.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import paths as p
from src.datasets.paired_dataset import PairedSurveillanceDataset, paired_root_exists
from src.preprocessing.build_paired_dataset import (
    CONGRUENT_CELLS,
    INCONGRUENT_CELLS,
    TEST_CELLS,
)

from .metrics import MetricsCalculator
from .model_loading import build_roc_pr

SCENARIOS = [
    ("baseline (congruente)", list(CONGRUENT_CELLS)),
    ("cross-label (incongruente)", list(INCONGRUENT_CELLS)),
    ("violent + violent face", ["violent_violent_face"]),
    ("violent + non_violent face", ["violent_non_violent_face"]),
    ("non_violent + non_violent face", ["non_violent_non_violent_face"]),
    ("non_violent + violent face", ["non_violent_violent_face"]),
]

# comparações justas: agregado vs agregado e cada célula incongruente
# vs a célula congruente correspondente (mesmo rótulo de vídeo)
DELTA_PAIRS = [
    ("cross-label (incongruente)", "baseline (congruente)"),
    ("violent + non_violent face", "violent + violent face"),
    ("non_violent + violent face", "non_violent + non_violent face"),
]


def cm_values(metrics: dict) -> dict:
    """Extrai tn/fp/fn/tp do metrics['confusion_matrix'] (chaves longas ou curtas)."""
    cm = metrics.get("confusion_matrix", {})
    return {
        "tn": cm.get("tn", cm.get("true_negative", 0)),
        "fp": cm.get("fp", cm.get("false_positive", 0)),
        "fn": cm.get("fn", cm.get("false_negative", 0)),
        "tp": cm.get("tp", cm.get("true_positive", 0)),
    }


def evaluate_cells(
    model,
    cells,
    device: str,
    batch_size: int = 8,
    paired_root: Optional[str] = None,
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, int]:
    """Avalia o modelo no split 'test' do dataset pareado, filtrado por células."""
    root = Path(paired_root) if paired_root else p.PAIRED_ROOT
    if not paired_root_exists(root):
        raise FileNotFoundError(
            f"Dataset pareado não encontrado em {root}. "
            "Gere com: python -m src.preprocessing.build_paired_dataset"
        )

    dataset = PairedSurveillanceDataset(
        split="test",
        paired_root=str(root),
        cells=list(cells),
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="keypoints",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    calculator = MetricsCalculator(
        model, loader, device=device,
        class_names=["Non-Violent", "Violent"], num_classes=2,
    )
    metrics, y_true, y_pred, y_proba = calculator.evaluate()
    return metrics, y_true, y_pred, y_proba, len(dataset)


def evaluate_scenario(
    model,
    cells,
    device: str,
    batch_size: int,
    paired_root: Optional[str] = None,
) -> dict:
    metrics, y_true, _, y_proba, n = evaluate_cells(model, cells, device, batch_size, paired_root)
    return {
        "metrics": metrics,
        "n_samples": n,
        "cells": list(cells),
        "roc_pr": build_roc_pr("multimodal", y_true, y_proba, 2, ["Non-Violent", "Violent"]),
    }


def run_scenarios(
    model,
    device: str,
    batch_size: int,
    paired_root: Optional[str] = None,
) -> dict:
    """Avalia todos os cenários (4 células + agregados) e retorna resultados."""
    all_results = {}
    for scenario_name, cells in SCENARIOS:
        print(f"Avaliando: {scenario_name}...")
        data = evaluate_scenario(model, cells, device, batch_size, paired_root)
        all_results[scenario_name] = data
        m = data["metrics"]
        print(f"  ✓ n={data['n_samples']}, accuracy={m['accuracy']:.4f}, "
              f"f1(macro)={m['f1_score']['macro']:.4f}")
    return all_results


def generate_comparison_table(results: dict) -> str:
    """Gera tabela comparativa em texto."""
    header = f"{'Cenário':<35} {'N':>5} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10} {'AUC-ROC':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for scenario, data in results.items():
        m = data["metrics"]
        lines.append(
            f"{scenario:<35} "
            f"{data['n_samples']:>5} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_score']['macro']:>10.4f} "
            f"{m['precision']['macro']:>10.4f} "
            f"{m['recall']['macro']:>10.4f} "
            f"{m.get('auc_roc', 0.0):>10.4f}"
        )

    lines.append(sep)
    lines.append("")
    lines.append("Deltas (incongruente - congruente):")
    for cand, ref in DELTA_PAIRS:
        if cand not in results or ref not in results:
            continue
        m = results[cand]["metrics"]
        r = results[ref]["metrics"]
        delta_f1 = m["f1_score"]["macro"] - r["f1_score"]["macro"]
        delta_acc = m["accuracy"] - r["accuracy"]
        lines.append(
            f"  {cand:<31} vs {ref:<24} "
            f"ΔF1={'+' if delta_f1 >= 0 else ''}{delta_f1:.4f}  "
            f"ΔAcc={'+' if delta_acc >= 0 else ''}{delta_acc:.4f}"
        )

    return "\n".join(lines)


def generate_confusion_matrix_summary(results: dict) -> str:
    """Gera resumo das matrizes de confusão."""
    lines = ["Matrizes de Confusão:", ""]
    header = f"{'Cenário':<35} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}"
    sep = "-" * len(header)
    lines += [sep, header, sep]

    for scenario, data in results.items():
        tn, fp, fn, tp = cm_values(data["metrics"]).values()
        lines.append(f"{scenario:<35} {tn:>6} {fp:>6} {fn:>6} {tp:>6}")

    lines.append(sep)
    return "\n".join(lines)


def generate_charts(output_dir: Path, report: dict) -> List[Path]:
    """Gera gráficos comparativos a partir do relatório (barras, CMs, ΔF1, ROC/PR)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib não instalado. Pulando geração de gráficos.")
        return []

    scenarios = list(report["scenarios"].keys())
    metrics_keys = ["accuracy", "f1_score", "precision", "recall"]
    metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    metric_subkey = {"f1_score": "macro", "precision": "macro", "recall": "macro"}
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    written: List[Path] = []

    # ── Gráfico 1: Barras comparativas de métricas ────────────────────────
    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(18, 5))
    for ax, key, label in zip(axes, metrics_keys, metric_labels):
        values = []
        for s in scenarios:
            m = report["scenarios"][s]["metrics"]
            v = m.get(key, 0.0)
            if isinstance(v, dict):
                v = v.get(metric_subkey.get(key, "macro"), 0.0)
            values.append(float(v))

        bars = ax.bar(range(len(scenarios)), values, color=colors[:len(scenarios)])
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(
            [s.replace(" + ", "\n").replace(" (congruente)", "\n(congruente)")
             .replace(" (incongruente)", "\n(incongruente)") for s in scenarios],
            fontsize=8,
        )
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Impact Study: Face × Video (células do split de teste)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    chart_path = output_dir / "comparison_metrics.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(chart_path)
    print(f"  ✓ {chart_path}")

    # ── Gráfico 2: Matriz de confusão lado a lado ─────────────────────────
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        tn, fp, fn, tp = cm_values(report["scenarios"][scenario]["metrics"]).values()
        matrix = [[tn, fp], [fn, tp]]

        ax.imshow(matrix, cmap="Blues", vmin=0)
        ax.set_title(scenario.replace(" + ", "\n"), fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Violent", "Violent"])
        ax.set_yticklabels(["Non-Violent", "Violent"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i][j]), ha="center", va="center",
                        color="white" if matrix[i][j] > max(tn, tp) / 2 else "black",
                        fontsize=12, fontweight="bold")

    fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(cm_path)
    print(f"  ✓ {cm_path}")

    # ── Gráfico 3: Delta de F1 (incongruente vs congruente) ───────────────
    pairs = [
        (cand, ref) for cand, ref in DELTA_PAIRS
        if cand in report["scenarios"] and ref in report["scenarios"]
    ]
    if pairs:
        deltas, labels = [], []
        for cand, ref in pairs:
            f1_c = report["scenarios"][cand]["metrics"]["f1_score"]["macro"]
            f1_r = report["scenarios"][ref]["metrics"]["f1_score"]["macro"]
            deltas.append(float(f1_c) - float(f1_r))
            labels.append(f"{cand}\nvs {ref}")

        fig, ax = plt.subplots(figsize=(9, 4))
        colors_delta = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
        bars = ax.bar(range(len(deltas)), deltas, color=colors_delta)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Δ F1-Score (incongruente - congruente)")
        ax.set_title("F1-Score Impact (cross-label vs congruent)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:+.4f}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

        plt.tight_layout()
        delta_path = output_dir / "f1_delta.png"
        fig.savefig(delta_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(delta_path)
        print(f"  ✓ {delta_path}")

    # ── Gráfico 4: Curvas ROC e Precision-Recall por cenário ──────────────
    roc_curves = {
        s: report["scenarios"][s]["roc_pr"]
        for s in scenarios
        if report["scenarios"][s].get("roc_pr")
    }
    if roc_curves:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

        for scenario, curve in roc_curves.items():
            fpr, tpr = curve.get("fpr", []), curve.get("tpr", [])
            if fpr and tpr:
                auc = curve.get("auc")
                ax_roc.plot(fpr, tpr,
                            label=f"{scenario} (AUC={auc:.3f})" if auc is not None else scenario)

            precision, recall = curve.get("precision", []), curve.get("recall", [])
            if precision and recall:
                ap = curve.get("ap")
                ax_pr.plot(recall, precision,
                           label=f"{scenario} (AP={ap:.3f})" if ap is not None else scenario)

        ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC — Impact Study (multimodal)")
        ax_roc.legend(fontsize=7, loc="lower right")
        ax_roc.grid(True, alpha=0.3)

        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_ylim(0, 1.05)
        ax_pr.set_title("Precision-Recall — Impact Study (multimodal)")
        ax_pr.legend(fontsize=7, loc="lower left")
        ax_pr.grid(True, alpha=0.3)

        plt.tight_layout()
        roc_path = output_dir / "roc_pr_curves.png"
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(roc_path)
        print(f"  ✓ {roc_path}")

    return written


def build_report(all_results: dict, config: dict) -> dict:
    """Monta o relatório JSON a partir dos resultados por cenário."""
    comparison_table = generate_comparison_table(all_results)
    cm_summary = generate_confusion_matrix_summary(all_results)
    return {
        "scenarios": {
            name: {
                "metrics": data["metrics"],
                "n_samples": data["n_samples"],
                "cells": data["cells"],
                "roc_pr": data.get("roc_pr"),
            }
            for name, data in all_results.items()
        },
        "comparison_text": comparison_table,
        "confusion_matrix_text": cm_summary,
        "config": config,
    }


__all__ = [
    "SCENARIOS",
    "DELTA_PAIRS",
    "TEST_CELLS",
    "cm_values",
    "evaluate_cells",
    "evaluate_scenario",
    "run_scenarios",
    "generate_comparison_table",
    "generate_confusion_matrix_summary",
    "generate_charts",
    "build_report",
]
