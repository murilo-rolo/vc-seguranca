"""
Script principal para executar todos os experimentos de avaliação.

Uso:
    # Avaliar multimodal
    python run_evaluation.py --model multimodal --model_path models/multimodal/weights/best_model.pth

    # Avaliar CNN 3D
    python run_evaluation.py --model cnn3d --model_path models/cnn3d/weights/best_model.pth

    # Avaliar sub-modelo de vídeo (CNN 3D) independentemente
    python run_evaluation.py --model video --model_path models/cnn3d/weights/best_model.pth --metrics

    # Avaliar sub-modelo de pose (branch do multimodal) independentemente
    python run_evaluation.py --model pose --model_path models/multimodal/weights/best_model.pth --metrics

    # Avaliar sub-modelo de emoção (EmotionNet, classificação binária) independentemente
    python run_evaluation.py --model emotion --model_path models/emotion_cnn/weights/best_model.pth --metrics

    # Avaliar TODOS os sub-modelos (video/pose/emotion) + gerar gráficos
    python run_evaluation.py --model multimodal --model_path models/multimodal/weights/best_model.pth --all --charts

    # Estudo de impacto no modelo fusionado
    python run_evaluation.py --model multimodal --model_path models/multimodal/weights/best_model.pth --impact_study --noise_std 0.05

    # Executar todos os experimentos
    python run_evaluation.py --model multimodal --model_path models/multimodal/weights/best_model.pth --all
"""

import argparse
import json

import torch

from src import paths as p
from src.evaluation.ablation_study import ImpactStudy
from src.evaluation.charts import generate_report
from src.evaluation.dataloaders import get_full_model_test_loader
from src.evaluation.limitations_analysis import LimitationsAnalyzer
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.model_loading import build_roc_pr, load_model
from src.evaluation.per_model import run_per_model_evaluation
from src.evaluation.performance_eval import PerformanceEvaluator
from src.evaluation.robustness_eval import DEFAULT_DISTORTION_CONFIGS, RobustnessEvaluator
from src.models.cnn3d_risk import ensure_channels_first

FULL_MODELS = ("cnn3d", "multimodal")
SUB_MODELS = ("video", "pose", "emotion")
BINARY_CLASS_NAMES = ["Non-Violent", "Violent"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executar experimentos de avaliação")

    parser.add_argument(
        "--model", type=str,
        choices=["cnn3d", "multimodal", "video", "pose", "emotion"],
        required=True,
        help="Tipo de modelo (video/pose/emotion = avaliação de sub-modelo)",
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Caminho para checkpoint do modelo")
    parser.add_argument("--batch_size", type=int, default=8, help="Tamanho do batch")

    parser.add_argument("--video_model_path", type=str,
                        default="models/cnn3d/weights/best_model.pth",
                        help="Checkpoint do sub-modelo de vídeo")
    parser.add_argument("--pose_model_path", type=str,
                        default="models/multimodal/weights/best_model.pth",
                        help="Checkpoint do sub-modelo de pose")
    parser.add_argument("--emotion_model_path", type=str,
                        default="models/emotion_cnn/weights/best_model.pth",
                        help="Checkpoint do sub-modelo de emoção")

    parser.add_argument("--all", action="store_true",
                        help="Executar todos os experimentos + avaliar sub-modelos video/pose/emotion")
    parser.add_argument("--metrics", action="store_true", help="Calcular métricas básicas")
    parser.add_argument("--robustness", action="store_true", help="Testar robustez")
    parser.add_argument("--performance", action="store_true", help="Avaliar performance")
    parser.add_argument("--limitations", action="store_true", help="Analisar limitações")

    parser.add_argument("--charts", action="store_true",
                        help="Gerar gráficos padronizados: agregados em results/charts/ "
                             "e CMs de sub-modelos em results/<modelo>/charts/")
    parser.add_argument("--impact_study", action="store_true",
                        help="Executar estudo de impacto das modalidades no modelo fusionado")
    parser.add_argument("--noise_std", type=float, default=0.05,
                        help="Desvio padrão do ruído gaussiano no estudo de impacto (0 = sanity check)")

    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Nome do experimento (padrão: tipo do modelo)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device para inferência")
    return parser


def _resolve_experiments(args) -> tuple:
    """Decide quais experimentos rodar (--all liga todos; default: métricas)."""
    if args.all:
        return True, True, True, True
    flags = (args.metrics, args.robustness, args.performance, args.limitations)
    if not any(flags):
        flags = (True, False, False, False)
    return flags


def _output_root(model: str, experiment_name: str):
    """Raiz results/<experiment_name|modelo>/; prefixo dos arquivos gerados."""
    return p.results_for_model(model, experiment_name), experiment_name


def _run_full_model_experiments(args, run_metrics, run_robustness, run_performance, run_limitations):
    """Executa os experimentos dos modelos completos; retorna coletores de gráficos."""
    print("Loading model...")
    model = load_model(args.model, args.model_path, args.device,
                       video_model_path=args.video_model_path)
    print("✓ Model loaded")

    print("Loading dataset...")
    test_loader = get_full_model_test_loader(args.model, args.batch_size)
    print(f"✓ Dataset loaded ({len(test_loader)} batches)")

    results = {}
    metrics_by_model, class_names_by_model, roc_pr_data = {}, {}, {}
    summary_root, exp_label = _output_root(args.model, args.experiment_name)
    prepare = ensure_channels_first if args.model == "cnn3d" else None

    if run_metrics:
        print("\n" + "=" * 60 + "\n1. Calculating Basic Metrics\n" + "=" * 60)
        calculator = MetricsCalculator(model, test_loader, device=args.device,
                                       prepare_inputs=prepare)
        metrics, y_true, y_pred, y_proba = calculator.evaluate()
        results["metrics"] = metrics

        output_path = summary_root / "metrics"
        output_path.mkdir(parents=True, exist_ok=True)
        calculator.save_results(str(output_path), exp_label, metrics, y_true, y_pred, y_proba)
        print("✓ Metrics saved")

        metrics_by_model[args.model] = metrics
        class_names_by_model[args.model] = BINARY_CLASS_NAMES
        roc_pr_data[args.model] = build_roc_pr(args.model, y_true, y_proba, 2, BINARY_CLASS_NAMES)

    if run_robustness:
        print("\n" + "=" * 60 + "\n2. Testing Robustness\n" + "=" * 60)
        evaluator = RobustnessEvaluator(model, test_loader, device=args.device)
        results["robustness"] = evaluator.evaluate_all_distortions(
            DEFAULT_DISTORTION_CONFIGS,
            str(summary_root),
            "robustness",
        )
        print("✓ Robustness tests completed")

    if run_performance:
        print("\n" + "=" * 60 + "\n3. Evaluating Performance\n" + "=" * 60)
        evaluator = PerformanceEvaluator(model, test_loader, device=args.device)
        results["performance"] = evaluator.evaluate_all(
            str(summary_root),
            "performance",
        )
        print("✓ Performance evaluation completed")

    if run_limitations:
        print("\n" + "=" * 60 + "\n4. Analyzing Limitations\n" + "=" * 60)
        analyzer = LimitationsAnalyzer(model, test_loader, device=args.device)
        results["limitations"] = analyzer.generate_error_report(
            str(summary_root),
            "limitations",
        )
        print("✓ Limitations analysis completed")

    return model, test_loader, results, metrics_by_model, class_names_by_model, roc_pr_data, summary_root


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.experiment_name is None:
        args.experiment_name = args.model

    is_full_model = args.model in FULL_MODELS
    run_metrics, run_robustness, run_performance, run_limitations = _resolve_experiments(args)

    print("=" * 60)
    print("Advanced Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Dir: {p.results_for_model(args.model, args.experiment_name)}")
    print(f"Experiment Name: {args.experiment_name}")
    print(f"Device: {args.device}")
    print()
    print("Experiments to run:")
    print(f"  - Metrics: {run_metrics}")
    print(f"  - Robustness: {run_robustness}")
    print(f"  - Performance: {run_performance}")
    print(f"  - Limitations: {run_limitations}")
    print(f"  - Per-model (video/pose/emotion): {args.all or not is_full_model}")
    print(f"  - Impact study: {args.impact_study or (args.all and args.model == 'multimodal')}")
    print(f"  - Charts: {args.charts}")
    print("=" * 60)
    print()

    metrics_by_model, class_names_by_model, roc_pr_data = {}, {}, {}
    results, model, test_loader, summary_root = {}, None, None, None

    if is_full_model:
        (model, test_loader, results, metrics_by_model,
         class_names_by_model, roc_pr_data, summary_root) = _run_full_model_experiments(
            args, run_metrics, run_robustness, run_performance, run_limitations
        )
    else:
        print(f"Modelo '{args.model}' é um sub-modelo; executando avaliação por sub-modelo.")

    if args.all or not is_full_model:
        sub_targets = list(SUB_MODELS) if args.all else [args.model]
        sub_metrics, sub_class_names, sub_roc_pr = run_per_model_evaluation(
            device=args.device,
            targets=sub_targets,
            model_path=args.model_path,
            batch_size=args.batch_size,
            video_model_path=args.video_model_path,
            pose_model_path=args.pose_model_path,
            emotion_model_path=args.emotion_model_path,
        )
        metrics_by_model.update(sub_metrics)
        class_names_by_model.update(sub_class_names)
        roc_pr_data.update(sub_roc_pr)

    if args.impact_study or (args.all and args.model == "multimodal"):
        if args.model != "multimodal":
            print("\n⚠️  Estudo de impacto requer o modelo multimodal fusionado "
                  "(--model multimodal). Pulando...")
        else:
            print("\n" + "=" * 60 + "\n5. Impact Study (input noise perturbation)\n" + "=" * 60)
            study = ImpactStudy(test_loader, device=args.device)
            results["impact_study"] = study.run_impact_study(
                model,
                dataloader=test_loader,
                noise_std=args.noise_std,
                output_dir=str(p.results_for_model(args.model, args.experiment_name)),
                experiment_name="impact_study",
            )

    if args.charts and metrics_by_model:
        # Heatmaps por modelo: sub-modelos não passam pelo save_results; o modelo
        # completo avaliado já gravou seu CM em <raiz>/charts/ (MetricsCalculator).
        confusion_dirs = {
            name: p.results_for_model(name) / "charts"
            for name in metrics_by_model
            if not (is_full_model and name == args.model)
        }
        written = generate_report(
            metrics_by_model,
            str(p.GLOBAL_CHARTS_ROOT),
            class_names_by_model,
            roc_pr_data,
            confusion_out_dirs=confusion_dirs,
        )
        print("\n" + "=" * 60)
        print("Gráficos gerados:")
        for w in written:
            print(f"  ✓ {w}")

    if is_full_model:
        summary_path = summary_root / "evaluation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 60)
    print("All experiments completed!")
    if is_full_model:
        print(f"Results saved to: {summary_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()
