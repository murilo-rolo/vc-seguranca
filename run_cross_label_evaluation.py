"""
Impact Study: Avaliação cross-label (face × vídeo).

O split de teste do dataset pareado (dataset/paired/test/) já contém as
4 células de mesmo tamanho por construção:

    violent + violent face              (congruente)
    violent + non_violent face          (incongruente)
    non_violent + non_violent face      (congruente)
    non_violent + violent face          (incongruente)

Este script avalia o modelo em cada célula (e nos agregados congruente /
incongruente) sem gerar CSVs cross-label.

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
from torch.utils.data import DataLoader

from src import paths as p
from src.datasets.paired_dataset import (
    PairedSurveillanceDataset,
    paired_root_exists,
)
from src.evaluation.metrics import MetricsCalculator
from src.models.multimodal_risk import create_multimodal_model
from src.models.cnn3d_risk import create_cnn3d_model
from src.preprocessing.build_paired_dataset import (
    CONGRUENT_CELLS,
    INCONGRUENT_CELLS,
    TEST_CELLS,
)

# cenário -> células do split de teste
SCENARIOS = [
    ("baseline (congruente)", list(CONGRUENT_CELLS)),
    ("cross-label (incongruente)", list(INCONGRUENT_CELLS)),
    ("violent + violent face", ["violent_violent_face"]),
    ("violent + non_violent face", ["violent_non_violent_face"]),
    ("non_violent + non_violent face", ["non_violent_non_violent_face"]),
    ("non_violent + violent face", ["non_violent_violent_face"]),
]

# comparações justas: agregado vs agregado e cada célula incongruente
# vs a célula congruente correspondente (mesmo rótulo de vídeo)
DELTA_PAIRS = [
    ("cross-label (incongruente)", "baseline (congruente)"),
    ("violent + non_violent face", "violent + violent face"),
    ("non_violent + violent face", "non_violent + non_violent face"),
]


def _cm_values(metrics: dict) -> dict:
    """Extrai tn/fp/fn/tp do metrics['confusion_matrix'] (chaves longas ou curtas)."""
    cm = metrics.get("confusion_matrix", {})
    return {
        "tn": cm.get("tn", cm.get("true_negative", 0)),
        "fp": cm.get("fp", cm.get("false_positive", 0)),
        "fn": cm.get("fn", cm.get("false_negative", 0)),
        "tp": cm.get("tp", cm.get("true_positive", 0)),
    }


def load_multimodal_model(model_path: str, device: str):
    """Carrega o modelo multimodal + backbone de vídeo do checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    fusion_method = checkpoint.get("fusion_method", "cross_attention")
    use_temporal = checkpoint.get("use_temporal_modeling", True)
    video_backbone_ckpt = checkpoint.get("video_backbone", "cnn3d")
    if video_backbone_ckpt != "cnn3d":
        raise ValueError(
            f"Backbone de vídeo '{video_backbone_ckpt}' não é mais suportado; "
            "o projeto usa apenas 'cnn3d'."
        )
    video_feature_dim = checkpoint.get("video_feature_dim") or 512

    model = create_multimodal_model(
        video_feature_dim=video_feature_dim,
        pose_feature_dim=51,
        emotion_feature_dim=128,
        num_frames=16,
        fusion_method=fusion_method,
        use_temporal_modeling=use_temporal,
        device=device,
    )
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    # Backbone de vídeo
    vckpt_path = str(p.CNN3D_WEIGHTS / "best_model.pth")
    vckpt = torch.load(vckpt_path, map_location=device)
    vmodel_name = vckpt.get("model_name") or vckpt.get("backbone") or "r2plus1d_18"
    video_model = create_cnn3d_model(
        model_name=vmodel_name,
        num_classes=2,
        checkpoint_path=vckpt_path,
        device=device,
    )
    video_model.eval()

    from run_evaluation import _MultimodalEvalWrapper
    wrapper = _MultimodalEvalWrapper(model, video_model)
    wrapper.eval()
    return wrapper


def evaluate_cells(
    model,
    cells,
    device: str,
    batch_size: int = 8,
    paired_root: str = None,
):
    """Avalia o modelo no split 'test' do dataset pareado, filtrado por células."""
    root = Path(paired_root) if paired_root else p.PAIRED_ROOT
    if not paired_root_exists(root):
        raise FileNotFoundError(
            f"Dataset pareado não encontrado em {root}. "
            "Gere com: python -m src.preprocessing.build_paired_dataset"
        )

    dataset = PairedSurveillanceDataset(
        split="test",
        paired_root=str(root),
        cells=list(cells),
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="keypoints",
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    calculator = MetricsCalculator(
        model, loader, device=device,
        class_names=["Non-Violent", "Violent"], num_classes=2,
    )
    metrics, y_true, y_pred, y_proba = calculator.evaluate()
    return metrics, y_true, y_pred, y_proba, len(dataset)


def evaluate_scenario(model, cells, device, batch_size, paired_root=None):
    metrics, y_true, y_pred, y_proba, n = evaluate_cells(
        model, cells, device, batch_size, paired_root,
    )
    return {
        "metrics": metrics,
        "n_samples": n,
        "cells": list(cells),
    }


def generate_comparison_table(results: dict) -> str:
    """Gera tabela comparativa em texto."""
    header = f"{'Cenário':<35} {'N':>5} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10} {'AUC-ROC':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for scenario, data in results.items():
        m = data["metrics"]
        line = (
            f"{scenario:<35} "
            f"{data['n_samples']:>5} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_score']['macro']:>10.4f} "
            f"{m['precision']['macro']:>10.4f} "
            f"{m['recall']['macro']:>10.4f} "
            f"{m.get('auc_roc', 0.0):>10.4f}"
        )
        lines.append(line)

    lines.append(sep)

    # Deltas justos: agregados e células vs sua contraparte congruente
    lines.append("")
    lines.append("Deltas (incongruente - congruente):")
    for cand, ref in DELTA_PAIRS:
        if cand not in results or ref not in results:
            continue
        m = results[cand]["metrics"]
        r = results[ref]["metrics"]
        delta_f1 = m["f1_score"]["macro"] - r["f1_score"]["macro"]
        delta_acc = m["accuracy"] - r["accuracy"]
        sign_f1 = "+" if delta_f1 >= 0 else ""
        sign_acc = "+" if delta_acc >= 0 else ""
        lines.append(
            f"  {cand:<31} vs {ref:<24} "
            f"ΔF1={sign_f1}{delta_f1:.4f}  ΔAcc={sign_acc}{delta_acc:.4f}"
        )

    return "\n".join(lines)


def generate_confusion_matrix_summary(results: dict) -> str:
    """Gera resumo das matrizes de confusão."""
    lines = ["Matrizes de Confusão:", ""]
    header = f"{'Cenário':<35} {'TN':>6} {'FP':>6} {'FN':>6} {'TP':>6}"
    sep = "-" * len(header)
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for scenario, data in results.items():
        tn, fp, fn, tp = _cm_values(data["metrics"]).values()
        lines.append(f"{scenario:<35} {tn:>6} {fp:>6} {fn:>6} {tp:>6}")

    lines.append(sep)
    return "\n".join(lines)


def run_scenarios(model, device: str, batch_size: int, paired_root: str = None) -> dict:
    """Avalia todos os cenários (4 células + agregados) e retorna resultados."""
    all_results = {}
    for scenario_name, cells in SCENARIOS:
        print(f"Avaliando: {scenario_name}...")
        data = evaluate_scenario(model, cells, device, batch_size, paired_root)
        all_results[scenario_name] = data
        m = data["metrics"]
        print(f"  ✓ n={data['n_samples']}, accuracy={m['accuracy']:.4f}, "
              f"f1(macro)={m['f1_score']['macro']:.4f}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Impact Study: avaliação cross-label por células do dataset pareado"
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
    args = parser.parse_args()

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

    # ── 1. Carregar modelo ─────────────────────────────────────────────────
    print("Carregando modelo multimodal...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    # ── 2. Avaliar cenários (células do split de teste) ────────────────────
    print("Avaliando cenários (split test do dataset pareado)...")
    all_results = run_scenarios(model, args.device, args.batch_size)

    # ── 3. Relatório ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Resultados")
    print("=" * 60)

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
            "test_cells": list(TEST_CELLS),
        },
    }

    report_path = output_dir / "cross_label_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Relatório salvo em: {report_path}")

    for scenario_name, data in all_results.items():
        safe_name = scenario_name.replace(" ", "_").replace("(", "").replace(")", "")
        metrics_path = output_dir / f"metrics_{safe_name}.json"
        with open(metrics_path, "w") as f:
            json.dump(data["metrics"], f, indent=2)

    print(f"✓ Métricas salvas em: {output_dir}/metrics_*.json")
    print("\n" + "=" * 60)
    print("Impact study concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
