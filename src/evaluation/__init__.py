"""
Módulo de avaliação experimental para modelos de detecção de violência.
"""

from .metrics import calculate_metrics, calculate_multiclass_metrics, MetricsCalculator
from .robustness_eval import RobustnessEvaluator, DEFAULT_DISTORTION_CONFIGS
from .performance_eval import PerformanceEvaluator
from .limitations_analysis import LimitationsAnalyzer
from .ablation_study import AblationStudy
from .charts import generate_report
from .utils import (
    save_results,
    load_results,
    create_experiment_dir,
    apply_distortions
)

__all__ = [
    'calculate_metrics',
    'calculate_multiclass_metrics',
    'MetricsCalculator',
    'RobustnessEvaluator',
    'DEFAULT_DISTORTION_CONFIGS',
    'apply_distortions',
    'PerformanceEvaluator',
    'LimitationsAnalyzer',
    'AblationStudy',
    'generate_report',
    'save_results',
    'load_results',
    'create_experiment_dir'
]
