"""
Impact Study: Orquestrador completo (validação + avaliação + gráficos).

Executa o pipeline completo do impact study sobre o split 'test' do dataset
pareado (dataset/paired/test/), que contém 4 células de mesmo tamanho
(ver src.evaluation.cross_label).

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
from src.evaluation.cross_label import build_report, generate_charts, run_scenarios
from src.evaluation.model_loading import load_multimodal_model
from src.preprocessing.build_paired_dataset import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Impact Study completo: validação + avaliação + gráficos"
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
    parser.add_argument("--charts", action="store_true",
                        help="Gerar gráficos comparativos")
    parser.add_argument("--skip_validate", action="store_true",
                        help="Pular validação dos symlinks do dataset pareado")
    return parser


def _validate_dataset() -> None:
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


def main():
    args = build_parser().parse_args()

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

    if args.skip_validate:
        print("━━━ Etapa 1: Validação pulada (--skip_validate) ━━━\n")
    else:
        _validate_dataset()

    print("━━━ Etapa 2: Avaliando modelo ━━━")
    print("Carregando modelo...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    all_results = run_scenarios(model, args.device, args.batch_size)
    report = build_report(all_results, config={
        "model_path": args.model_path,
        "device": args.device,
    })

    print("\n━━━ Etapa 3: Relatório ━━━")
    print(report["comparison_text"])
    print("\n" + report["confusion_matrix_text"])

    report_path = output_dir / "cross_label_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n✓ Relatório salvo em: {report_path}")

    if args.charts:
        print("\n━━━ Etapa 4: Gerando gráficos ━━━")
        generate_charts(output_dir, report)

    print("\n" + "=" * 60)
    print("Impact study concluído!")
    print(f"Resultados em: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
