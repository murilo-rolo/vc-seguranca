"""
Avaliação de sub-modelos (video / pose / emotion) independentemente (F5, EVAL-T1).

Extraído de run_evaluation.py: cada sub-modelo é avaliado com seu dataloader
e as métricas são salvas em `results/{video,pose,emotion}/metrics.json`.
Sob `--all`, os três são avaliados e um relatório agregado é gravado em
`results/reports/`.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src import paths as p
from src.models.cnn3d_risk import ensure_channels_first

from .dataloaders import get_per_model_test_loader
from .metrics import MetricsCalculator
from .model_loading import build_roc_pr, load_model, resolve_sub_model_path

SUB_MODEL_TYPES = ("video", "pose", "emotion")


def run_per_model_evaluation(
    device: str,
    targets: Optional[List[str]] = None,
    model_path: str = "",
    batch_size: int = 8,
    video_model_path: Optional[str] = None,
    pose_model_path: Optional[str] = None,
    emotion_model_path: Optional[str] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Avalia sub-modelos independentemente.

    Returns:
        (metrics_by_model, class_names_by_model, roc_pr_data)
    """
    targets = [t for t in (targets or []) if t in SUB_MODEL_TYPES]
    metrics_by_model: Dict = {}
    class_names_by_model: Dict = {}
    roc_pr_data: Dict = {}

    if not targets:
        return metrics_by_model, class_names_by_model, roc_pr_data

    print("\n" + "=" * 60)
    print("Avaliação por sub-modelo (video / pose / emotion)")
    print("=" * 60)

    for model_type in targets:
        try:
            sub_path = resolve_sub_model_path(
                model_type,
                model_path,
                video_model_path=video_model_path,
                pose_model_path=pose_model_path,
                emotion_model_path=emotion_model_path,
            )
            print(f"\n--- Sub-modelo: {model_type} ---")
            print(f"Checkpoint: {sub_path}")

            if not Path(sub_path).exists():
                raise FileNotFoundError(f"Checkpoint não encontrado: {sub_path}")

            model = load_model(model_type, sub_path, device, video_model_path=video_model_path)
            test_loader, class_names, num_classes = get_per_model_test_loader(
                model_type, batch_size
            )

            calculator = MetricsCalculator(
                model,
                test_loader,
                device=device,
                class_names=class_names,
                num_classes=num_classes,
                prepare_inputs=ensure_channels_first if model_type == "video" else None,
            )
            metrics, y_true, y_pred, y_proba = calculator.evaluate()

            out_dir = p.RESULTS_ROOT / model_type
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(
                f"✓ {model_type}: accuracy={metrics['accuracy']:.4f}, "
                f"f1(macro)={metrics['f1_score']['macro']:.4f}"
            )
            print(f"  Salvo em: {out_dir / 'metrics.json'}")

            metrics_by_model[model_type] = metrics
            class_names_by_model[model_type] = class_names
            roc_pr_data[model_type] = build_roc_pr(
                model_type, y_true, y_proba, num_classes, class_names
            )
        except Exception as e:
            print(f"⚠️  Erro ao avaliar sub-modelo '{model_type}': {e}")
            print("    Continuando com os demais modelos...")

    if metrics_by_model:
        report = {
            "models": {
                mtype: {
                    "metrics": metrics,
                    "class_names": class_names_by_model.get(mtype),
                }
                for mtype, metrics in metrics_by_model.items()
            },
            "summary": {
                mtype: {
                    "accuracy": metrics.get("accuracy"),
                    "f1_macro": metrics.get("f1_score", {}).get("macro"),
                }
                for mtype, metrics in metrics_by_model.items()
            },
        }
        report_dir = p.RESULTS_ROOT / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "per_model_evaluation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(f"\n✓ Relatório agregado salvo em: {report_dir / 'per_model_evaluation.json'}")

    return metrics_by_model, class_names_by_model, roc_pr_data
