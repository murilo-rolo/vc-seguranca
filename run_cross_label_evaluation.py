"""
Impact Study: Avaliação cross-label (face × vídeo).

O split de teste do dataset pareado (dataset/paired/test/) já contém as
4 células de mesmo tamanho por construção (ver src.evaluation.cross_label).

Uso:
    python run_cross_label_evaluation.py
    python run_cross_label_evaluation.py --model_path models/multimodal/weights/best_model.pth
    python run_cross_label_evaluation.py --batch_size 16
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from src import paths as p
from src.datasets.paired_dataset import paired_root_exists
from src.evaluation.cross_label import build_report, run_scenarios
from src.evaluation.model_loading import (
    load_feature_stats_for_checkpoint,
    load_multimodal_model,
)
from src.preprocessing.build_paired_dataset import TEST_CELLS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Impact Study: avaliação cross-label por células do dataset pareado"
    )
    parser.add_argument("--model_path", type=str,
                        default="models/multimodal/weights/best_model.pth",
                        help="Caminho do checkpoint multimodal")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device para inferência")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Diretório de saída (padrão: results/cross_label_impact)")
    return parser


def main():
    args = build_parser().parse_args()

    if not paired_root_exists():
        print("✗ dataset/paired não encontrado.")
        print("  Gere com: python -m src.preprocessing.build_paired_dataset")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else p.CROSS_LABEL_RESULTS_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Impact Study: Cross-Label Emotion × Video")
    print("=" * 60)
    print(f"Modelo: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print()

    print("Carregando modelo multimodal...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    print("Avaliando cenários (split test do dataset pareado)...")
    feature_stats = load_feature_stats_for_checkpoint(args.model_path)
    all_results = run_scenarios(
        model, args.device, args.batch_size, feature_stats=feature_stats,
    )

    report = build_report(all_results, config={
        "model_path": args.model_path,
        "device": args.device,
        "test_cells": list(TEST_CELLS),
        "normalize_features": feature_stats is not None,
    })

    print("\n" + "=" * 60 + "\nResultados\n" + "=" * 60)
    print(report["comparison_text"])
    print("\n" + report["confusion_matrix_text"])

    report_path = output_dir / "cross_label_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n✓ Relatório salvo em: {report_path}")

    for scenario_name, data in all_results.items():
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
        (output_dir / f"metrics_{safe_name}.json").write_text(
            json.dumps(data["metrics"], indent=2)
        )
    print(f"✓ Métricas salvas em: {output_dir}/metrics_*.json")

    print("\n" + "=" * 60 + "\nImpact study concluído!\n" + "=" * 60)


if __name__ == "__main__":
    main()
