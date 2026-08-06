"""
Exemplos de uso do módulo de avaliação experimental.
"""

import torch
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.robustness_eval import RobustnessEvaluator, DEFAULT_DISTORTION_CONFIGS
from src.evaluation.performance_eval import PerformanceEvaluator
from src.evaluation.limitations_analysis import LimitationsAnalyzer
from src.models.resnet_lstm import create_model as create_video_model
from src.datasets.surveillance_dataset import get_dataloaders
from src import paths as p


def example_basic_metrics():
    """Exemplo: Calcular métricas básicas."""
    print("=" * 60)
    print("Exemplo 1: Métricas Básicas")
    print("=" * 60)
    
    # Carregar modelo
    model = create_video_model(
        num_frames=16,
        hidden_size=256,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Carregar dataset
    _, _, test_loader = get_dataloaders(
        processed_data_root=str(p.PROCESSED_ROOT),
        batch_size=8,
        num_frames=16
    )
    
    # Calcular métricas
    calculator = MetricsCalculator(model, test_loader)
    metrics, y_true, y_pred, y_proba = calculator.evaluate()
    
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision']['macro']:.4f}")
    print(f"Recall (macro): {metrics['recall']['macro']:.4f}")
    print(f"F1-Score (macro): {metrics['f1_score']['macro']:.4f}")
    
    # Salvar resultados
    calculator.save_results(
        str(p.EXPERIMENTS_ROOT / "example"),
        "baseline",
        metrics,
        y_true,
        y_pred,
        y_proba
    )


def example_robustness():
    """Exemplo: Testar robustez."""
    print("\n" + "=" * 60)
    print("Exemplo 2: Teste de Robustez")
    print("=" * 60)
    
    # Carregar modelo e dataset
    model = create_video_model(num_frames=16, device="cuda")
    _, _, test_loader = get_dataloaders(batch_size=8)
    
    # Avaliar robustez
    robustness_eval = RobustnessEvaluator(model, test_loader)
    
    # Testar apenas algumas distorções
    test_configs = {
        "gaussian_blur": [0.0, 0.3, 0.6, 1.0],
        "gaussian_noise": [0.0, 0.3, 0.6, 1.0],
        "darkening": [0.0, 0.3, 0.6]
    }
    
    results = robustness_eval.evaluate_all_distortions(
        test_configs,
        str(p.EXPERIMENTS_ROOT / "example"),
        "robustness_test"
    )
    
    print("✓ Robustness tests completed")
    print(f"Results saved to: {p.EXPERIMENTS_ROOT / 'example' / 'robustness_test' / ''}")


def example_performance():
    """Exemplo: Avaliar performance."""
    print("\n" + "=" * 60)
    print("Exemplo 3: Avaliação de Performance")
    print("=" * 60)
    
    # Carregar modelo e dataset
    model = create_video_model(num_frames=16, device="cuda")
    _, _, test_loader = get_dataloaders(batch_size=8)
    
    # Avaliar performance
    perf_eval = PerformanceEvaluator(model, test_loader)
    
    # Medir FPS
    fps_results = perf_eval.measure_fps(num_iterations=50)
    print(f"Average FPS: {fps_results['fps_mean']:.2f}")
    
    # Medir latência
    latency_results = perf_eval.measure_latency(num_iterations=50)
    print(f"Average latency: {latency_results['total']['mean']*1000:.2f} ms")
    
    # Medir recursos
    resource_results = perf_eval.measure_resource_usage(duration=5.0)
    print(f"Average CPU usage: {resource_results['cpu']['mean']:.1f}%")
    print(f"Average memory: {resource_results['memory']['mean']:.2f} GB")


def example_limitations():
    """Exemplo: Analisar limitações."""
    print("\n" + "=" * 60)
    print("Exemplo 4: Análise de Limitações")
    print("=" * 60)
    
    # Carregar modelo e dataset
    model = create_video_model(num_frames=16, device="cuda")
    _, _, test_loader = get_dataloaders(batch_size=8)
    
    # Analisar limitações
    limitations_analyzer = LimitationsAnalyzer(model, test_loader)
    
    # Analisar erros
    analysis = limitations_analyzer.analyze_errors(
        str(p.EXPERIMENTS_ROOT / "example"),
        "limitations_test",
        save_examples=True
    )
    
    print(f"False Positives: {analysis['false_positives']['count']}")
    print(f"False Negatives: {analysis['false_negatives']['count']}")
    print(f"Borderline Cases: {analysis['borderline_cases']['count']}")
    
    # Gerar relatório
    report = limitations_analyzer.generate_error_report(
        str(p.EXPERIMENTS_ROOT / "example"),
        "error_report"
    )
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Exemplos de Uso - Módulo de Avaliação")
    print("=" * 60)
    print("\nCertifique-se de ter:")
    print("  1. Modelos treinados")
    print("  2. Dataset processado")
    print("  3. GPU disponível (recomendado)")
    print()
    
    try:
        # Descomente o exemplo que deseja executar:
        
        # example_basic_metrics()
        # example_robustness()
        # example_performance()
        # example_limitations()
        
        print("\n⚠ Descomente um dos exemplos acima para executar")
        print("\nOu use o script principal:")
        print(f"  python run_evaluation.py --model baseline --model_path {p.RESNET_LSTM_WEIGHTS / 'best_model.pth'} --all")
        
    except Exception as e:
        print(f"\nErro: {e}")
        import traceback
        traceback.print_exc()

