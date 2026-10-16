"""
Módulo de avaliação experimental para modelos de detecção de violência.
"""

from .ablation_study import ImpactStudy
from .charts import generate_report, plot_confusion_matrix
from .inference import collect_predictions, forward_batch, unpack_batch
from .limitations_analysis import LimitationsAnalyzer
from .metrics import (
    MetricsCalculator,
    calculate_metrics,
    calculate_multiclass_metrics,
)
from .performance_eval import PerformanceEvaluator
from .robustness_eval import DEFAULT_DISTORTION_CONFIGS, RobustnessEvaluator
from .utils import (
    apply_distortions,
    create_experiment_dir,
    load_results,
    save_results,
)

__all__ = [
    # métricas
    "calculate_metrics",
    "calculate_multiclass_metrics",
    "MetricsCalculator",
    # avaliações
    "RobustnessEvaluator",
    "DEFAULT_DISTORTION_CONFIGS",
    "PerformanceEvaluator",
    "LimitationsAnalyzer",
    "ImpactStudy",
    # inferência
    "collect_predictions",
    "unpack_batch",
    "forward_batch",
    # gráficos / utilidades
    "generate_report",
    "plot_confusion_matrix",
    "save_results",
    "load_results",
    "create_experiment_dir",
    "apply_distortions",
]
