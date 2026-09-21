"""
Impact Study: Orquestrador completo (geração + avaliação + gráficos).

Executa o pipeline completo do impact study:
  1. Gera CSVs cross-label (baseline, violent_face, non_violent_face)
  2. Avalia o modelo multimodal em cada cenário
  3. Gera gráficos comparativos

Uso:
    python run_impact_study.py
    python run_impact_study.py --model_path models/multimodal/weights/best_model.pth
    python run_impact_study.py --seed 123 --batch_size 16 --charts
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src import paths as p
from src.preprocessing.build_dataset_index import (
    build_cross_label_index,
    build_index,
    save_csv,
)


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

    scenarios = list(report["scenarios"].keys())
    metrics_keys = ["accuracy", "f1_score", "precision", "recall"]
    metric_labels = ["Accuracy", "F1-Score", "Precision", "Recall"]
    metric_subkey = {"f1_score": "macro", "precision": "macro", "recall": "macro"}

    # ── Gráfico 1: Barras comparativas de métricas ────────────────────────
    fig, axes = plt.subplots(1, len(metrics_keys), figsize=(18, 5))
    colors = ["#2ecc71", "#e74c3c", "#3498db"]

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
             for s in scenarios],
            fontsize=8,
        )
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Impact Study: Cross-Label Emotion × Video", fontsize=14, fontweight="bold")
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
        cm = report["scenarios"][scenario]["metrics"].get("confusion_matrix", {})
        tn = cm.get("tn", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tp = cm.get("tp", 0)
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

    # ── Gráfico 3: Delta de F1 (waterfall) ───────────────────────────────
    if "baseline (congruente)" in report["scenarios"]:
        baseline_f1 = report["scenarios"]["baseline (congruente)"]["metrics"]["f1_score"]["macro"]
        cross_scenarios = [s for s in scenarios if s != "baseline (congruente)"]
        deltas = []
        labels = []
        for s in cross_scenarios:
            f1 = report["scenarios"][s]["metrics"]["f1_score"]["macro"]
            deltas.append(float(f1) - float(baseline_f1))
            labels.append(s.replace(" + ", "\n"))

        fig, ax = plt.subplots(figsize=(8, 4))
        colors_delta = ["#e74c3c" if d < 0 else "#2ecc71" for d in deltas]
        bars = ax.bar(range(len(deltas)), deltas, color=colors_delta)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Δ F1-Score vs Baseline")
        ax.set_title("F1-Score Impact (cross-label vs congruent)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, deltas):
            y_pos = bar.get_height() if val >= 0 else bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"{val:+.4f}", ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=10)

        plt.tight_layout()
        delta_path = output_dir / "f1_delta.png"
        fig.savefig(delta_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {delta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Impact Study completo: geração + avaliação + gráficos"
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
        "--seed", type=int, default=42,
        help="Seed para geração dos CSVs cross-label",
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
        "--skip_generation", action="store_true",
        help="Pular geração de CSVs (usar CSVs existentes no output_dir)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else p.CROSS_LABEL_RESULTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Impact Study: Cross-Label Emotion × Video")
    print("=" * 60)
    print(f"Modelo: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print(f"Output: {output_dir}")
    print(f"Charts: {args.charts}")
    print()

    # ── 1. Gerar CSVs ─────────────────────────────────────────────────────
    if not args.skip_generation:
        print("━━━ Etapa 1: Gerando CSVs cross-label ━━━")

        print("  [1/3] Baseline (congruente)...")
        baseline_rows = build_index(split="all", seed=args.seed)
        baseline_csv = output_dir / "pipeline_baseline.csv"
        save_csv(baseline_rows, str(baseline_csv))

        print("  [2/3] violent_face + non_violent_video...")
        vf_rows = build_cross_label_index(
            scenario="violent_face_non_violent_video",
            split="all", seed=args.seed,
        )
        vf_csv = output_dir / "pipeline_violent_face.csv"
        save_csv(vf_rows, str(vf_csv))

        print("  [3/3] non_violent_face + violent_video...")
        nvf_rows = build_cross_label_index(
            scenario="non_violent_face_violent_video",
            split="all", seed=args.seed,
        )
        nvf_csv = output_dir / "pipeline_non_violent_face.csv"
        save_csv(nvf_rows, str(nvf_csv))
        print()
    else:
        print("━━━ Etapa 1: Usando CSVs existentes ━━━")
        baseline_csv = output_dir / "pipeline_baseline.csv"
        vf_csv = output_dir / "pipeline_violent_face.csv"
        nvf_csv = output_dir / "pipeline_non_violent_face.csv"

        for csv_path in [baseline_csv, vf_csv, nvf_csv]:
            if not csv_path.exists():
                print(f"✗ CSV não encontrado: {csv_path}")
                print("  Remova --skip_generation para gerar automaticamente.")
                sys.exit(1)
        print("  ✓ CSVs encontrados\n")

    # ── 2. Avaliar ────────────────────────────────────────────────────────
    print("━━━ Etapa 2: Avaliando modelo ━━━")
    # Reutiliza o script de avaliação
    eval_args = [
        "--model_path", args.model_path,
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--device", args.device,
        "--output_dir", str(output_dir),
    ]
    # Simula a chamada importando e chamando diretamente
    from run_cross_label_evaluation import (
        load_multimodal_model,
        evaluate_on_csv,
        generate_comparison_table,
        generate_confusion_matrix_summary,
    )

    print("Carregando modelo...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    scenarios = [
        ("baseline (congruente)", baseline_csv),
        ("violent_face + non_violent_video", vf_csv),
        ("non_violent_face + violent_video", nvf_csv),
    ]

    all_results = {}
    for scenario_name, csv_path in scenarios:
        print(f"Avaliando: {scenario_name}...")
        metrics, y_true, y_pred, y_proba, n_samples = evaluate_on_csv(
            model, str(csv_path), args.device, args.batch_size,
        )
        all_results[scenario_name] = {
            "metrics": metrics,
            "n_samples": n_samples,
        }
        print(f"  ✓ accuracy={metrics['accuracy']:.4f}, "
              f"f1(macro)={metrics['f1_score']['macro']:.4f}")

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
                "csv_path": str(scenarios[i][1]),
            }
            for i, (name, data) in enumerate(all_results.items())
        },
        "comparison_text": comparison_table,
        "confusion_matrix_text": cm_summary,
        "config": {
            "model_path": args.model_path,
            "seed": args.seed,
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
