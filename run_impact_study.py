"""
Impact Study: Orquestrador completo (validação + avaliação + gráficos).

Executa o pipeline completo do impact study sobre o split 'test' do dataset
pareado (dataset/paired/test/), que contém 4 células de mesmo tamanho:
violent+violent face, violent+non_violent face,
non_violent+non_violent face, non_violent+violent face.

  1. Valida os symlinks do dataset pareado
  2. Avalia o modelo por célula e nos agregados (congruente/incongruente)
  3. Gera gráficos comparativos

Uso:
    python run_impact_study.py
    python run_impact_study.py --model_path models/multimodal/weights/best_model.pth
    python run_impact_study.py --batch_size 16 --charts
    python run_impact_study.py --skip_validate
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src import paths as p
from src.preprocessing.build_paired_dataset import validate


def generate_charts(output_dir: Path, report: dict):
    """Gera gráficos comparativos a partir do relatório."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠️  matplotlib não instalado. Pulando geração de gráficos.")
        return

    from run_cross_label_evaluation import DELTA_PAIRS, _cm_values

    scenarios = list(report["scenarios"].keys())
    metrics_keys = ["accuracy", "f1_score", "precision", "recall"]
    metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    metric_subkey = {"f1_score": "macro", "precision": "macro", "recall": "macro"}
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]

    # ── Gráfico 1: Barras comparativas de métricas ────────────────────────
    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(18, 5))

    for ax, key, label in zip(axes, metrics_keys, metric_labels):
        values = []
        for s in scenarios:
            m = report["scenarios"][s]["metrics"]
            if key in m:
                v = m[key]
                if isinstance(v, dict):
                    v = v.get(metric_subkey.get(key, "macro"), 0.0)
                values.append(float(v))
            else:
                values.append(0.0)

        bars = ax.bar(range(len(scenarios)), values, color=colors[:len(scenarios)])
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(
            [s.replace(" + ", "\n").replace(" (congruente)", "\n(congruente)")
             .replace(" (incongruente)", "\n(incongruente)")
             for s in scenarios],
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
    print(f"  ✓ {chart_path}")

    # ── Gráfico 2: Matriz de confusão lado a lado ─────────────────────────
    fig, axes = plt.subplots(1, len(scenarios), figsize=(5 * len(scenarios), 4))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        tn, fp, fn, tp = _cm_values(
            report["scenarios"][scenario]["metrics"]
        ).values()
        matrix = [[tn, fp], [fn, tp]]

        im = ax.imshow(matrix, cmap="Blues", vmin=0)
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
            y_pos = bar.get_height() if val >= 0 else bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{val:+.4f}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=10)

        plt.tight_layout()
        delta_path = output_dir / "f1_delta.png"
        fig.savefig(delta_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {delta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Impact Study completo: validação + avaliação + gráficos"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="models/multimodal/weights/best_model.pth",
        help="Caminho do checkpoint multimodal",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Tamanho do batch",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para inferência",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Diretório de saída (padrão: results/cross_label_impact)",
    )
    parser.add_argument(
        "--charts", action="store_true",
        help="Gerar gráficos comparativos",
    )
    parser.add_argument(
        "--skip_validate", action="store_true",
        help="Pular validação dos symlinks do dataset pareado",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else p.CROSS_LABEL_RESULTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Impact Study: Cross-Label Emotion × Video")
    print("=" * 60)
    print(f"Modelo: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print(f"Charts: {args.charts}")
    print()

    # ── 1. Validar dataset pareado ─────────────────────────────────────────
    if args.skip_validate:
        print("━━━ Etapa 1: Validação pulada (--skip_validate) ━━━\n")
    else:
        print("━━━ Etapa 1: Validando symlinks do dataset pareado ━━━")
        n_ok, errors = validate()
        print(f"  ✓ {n_ok} symlinks resolvendo")
        for err in errors:
            print(f"  ✗ {err}")
        if errors:
            print("\n✗ Dataset com links quebrados. Rebuild: "
                  "python -m src.preprocessing.build_paired_dataset --force")
            sys.exit(1)
        print()

    # ── 2. Avaliar ────────────────────────────────────────────────────────
    print("━━━ Etapa 2: Avaliando modelo ━━━")
    from run_cross_label_evaluation import (
        load_multimodal_model,
        run_scenarios,
        generate_comparison_table,
        generate_confusion_matrix_summary,
    )

    print("Carregando modelo...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    all_results = run_scenarios(model, args.device, args.batch_size)

    # ── 3. Relatório ──────────────────────────────────────────────────────
    print("\n━━━ Etapa 3: Relatório ━━━")

    comparison_table = generate_comparison_table(all_results)
    print(comparison_table)

    cm_summary = generate_confusion_matrix_summary(all_results)
    print("\n" + cm_summary)

    report = {
        "scenarios": {
            name: {
                "metrics": data["metrics"],
                "n_samples": data["n_samples"],
                "cells": data["cells"],
            }
            for name, data in all_results.items()
        },
        "comparison_text": comparison_table,
        "confusion_matrix_text": cm_summary,
        "config": {
            "model_path": args.model_path,
            "device": args.device,
        },
    }

    report_path = output_dir / "cross_label_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Relatório salvo em: {report_path}")

    # ── 4. Gráficos ───────────────────────────────────────────────────────
    if args.charts:
        print("\n━━━ Etapa 4: Gerando gráficos ━━━")
        generate_charts(output_dir, report)

    print("\n" + "=" * 60)
    print("Impact study concluído!")
    print(f"Resultados em: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
