"""
Diagnóstico de ablação de modalidades (Fase C).

Avalia sub-modelos standalone (uma modalidade) e máscaras de entrada no
modelo multimodal sobre as células do dataset pareado (test).

Uso:
    python run_modality_ablation.py
    python run_modality_ablation.py --model_path models/multimodal/weights/best_model.pth
    python run_modality_ablation.py --batch_size 16 --skip_validate
"""

import argparse
import json
from pathlib import Path

import torch

from src import paths as p
from src.evaluation.cross_label import SCENARIOS, cm_values, evaluate_scenario
from src.evaluation.modality_ablation import build_ablation_models
from src.preprocessing.build_paired_dataset import validate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ablação de modalidades: sub-modelos standalone + máscaras"
    )
    parser.add_argument("--model_path", type=str,
                        default="models/multimodal/weights/best_model.pth",
                        help="Checkpoint multimodal (branches e máscaras)")
    parser.add_argument("--video_model_path", type=str, default=None,
                        help="Checkpoint CNN3D (padrão: models/cnn3d/weights/best_model.pth)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Padrão: results/modality_ablation")
    parser.add_argument("--skip_validate", action="store_true")
    return parser


def _validate_dataset() -> None:
    print("━━━ Validando symlinks do dataset pareado ━━━")
    n_ok, errors = validate()
    print(f"  ✓ {n_ok} symlinks resolvendo")
    for err in errors:
        print(f"  ✗ {err}")
    if errors:
        raise SystemExit("Dataset com links quebrados. "
                         "Rebuild: python -m src.preprocessing.build_paired_dataset --force")
    print()


def _scenario_row(data: dict) -> dict:
    m = data["metrics"]
    tn, fp, fn, tp = cm_values(m).values()
    n = data["n_samples"]
    # recall/specificity condicionais (células single-class são só uma linha da CM)
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    return {
        "n": n,
        "accuracy": m["accuracy"],
        "auc_roc": m.get("auc_roc"),
        "f1_macro": m["f1_score"]["macro"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall_cond": recall,
        "specificity_cond": specificity,
    }


def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else p.PROJECT_ROOT / "results" / "modality_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Ablação de Modalidades — diagnóstico do atalho facial")
    print("=" * 60)
    print(f"Modelo multimodal: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}\n")

    if not args.skip_validate:
        _validate_dataset()

    print("━━━ Montando modelos de ablação ━━━")
    models = build_ablation_models(
        args.model_path, args.device, video_model_path=args.video_model_path,
    )
    for name in models:
        print(f"  ✓ {name}")
    print()

    report = {"model_path": args.model_path, "models": {}}
    for name, model in models.items():
        print(f"━━━ Modelo: {name} ━━━")
        model_results = {}
        for scenario_name, cells in SCENARIOS:
            data = evaluate_scenario(model, cells, args.device, args.batch_size)
            row = _scenario_row(data)
            model_results[scenario_name] = row
            auc = f"{row['auc_roc']:.4f}" if row["auc_roc"] is not None else "n/a"
            print(f"  {scenario_name:<35} n={row['n']:>4}  acc={row['accuracy']:.4f}  AUC={auc}")
        report["models"][name] = model_results
        print()

    # Tabela resumo: accuracy por modelo × cenário
    scenario_names = [name for name, _ in SCENARIOS]
    header = f"{'Modelo':<32}" + "".join(f"{s[:18]:>20}" for s in scenario_names)
    print("━━━ Resumo: Accuracy por modelo × cenário ━━━")
    print(header)
    print("-" * len(header))
    for name, results in report["models"].items():
        cells_str = "".join(f"{results[s]['accuracy']:>20.4f}" for s in scenario_names)
        print(f"{name:<32}{cells_str}")
    print()

    # Foco: dependência da fusão (máscaras) nas células incongruentes
    print("━━━ Dependência da fusão (células incongruentes) ━━━")
    focus = [
        "violent + non_violent face",
        "non_violent + violent face",
    ]
    print(f"{'Modelo':<32}" + "".join(f"{s[:24]:>26}" for s in focus))
    print("-" * (32 + 26 * len(focus)))
    for name, results in report["models"].items():
        cells_str = ""
        for s in focus:
            r = results[s]
            cond = r["recall_cond"] if "violent +" in s and "non_violent face" in s and s.startswith("violent") else r["specificity_cond"]
            # para violent+non_violent face: todos positivos -> recall; para non_violent+violent face: todos negativos -> specificity
            if s == "violent + non_violent face":
                val = r["recall_cond"]
            else:
                val = r["specificity_cond"]
            cells_str += f"{(f'{val:.4f}' if val is not None else 'n/a'):>26}"
        print(f"{name:<32}{cells_str}")
    print()

    report_path = output_dir / "modality_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"✓ Relatório salvo em: {report_path}")


if __name__ == "__main__":
    main()
