"""
Impact Study: Avaliação cross-label (emoção × vídeo).

Testa o impacto da emoção facial no modelo multimodal trocando os labels:
  - violent_face: vídeos violent com faces non_violent
  - non_violent_face: vídeos non_violent com faces violent

Compara com o baseline (labels congruentes).

Uso:
    python run_cross_label_evaluation.py
    python run_cross_label_evaluation.py --model_path models/multimodal/weights/best_model.pth
    python run_cross_label_evaluation.py --seed 123 --batch_size 16
"""

import argparse
import json
import csv
import sys
from pathlib import Path

import torch

from src import paths as p
from src.evaluation.metrics import MetricsCalculator
from src.models.multimodal_risk import create_multimodal_model
from src.models.cnn3d_risk import create_cnn3d_model
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.preprocessing.build_dataset_index import (
    build_cross_label_index,
    build_index,
    save_csv,
)


def load_multimodal_model(model_path: str, device: str):
    """Carrega o modelo multimodal + backbone de vídeo do checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)

    fusion_method = checkpoint.get("fusion_method", "cross_attention")
    use_temporal = checkpoint.get("use_temporal_modeling", True)
    video_backbone_ckpt = checkpoint.get("video_backbone", "cnn3d")
    video_feature_dim = checkpoint.get("video_feature_dim")
    if video_feature_dim is None:
        video_feature_dim = 512 if video_backbone_ckpt == "cnn3d" else 256

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
    if video_backbone_ckpt == "cnn3d":
        vckpt_path = str(p.CNN3D_WEIGHTS / "best_model.pth")
        vckpt = torch.load(vckpt_path, map_location=device)
        vmodel_name = vckpt.get("model_name") or vckpt.get("backbone") or "r2plus1d_18"
        video_model = create_cnn3d_model(
            model_name=vmodel_name,
            num_classes=2,
            checkpoint_path=vckpt_path,
            device=device,
        )
    else:
        from src.models.resnet_lstm import create_model as create_video_model
        vckpt_path = str(p.RESNET_LSTM_WEIGHTS / "best_model.pth")
        video_model = create_video_model(
            num_frames=16, hidden_size=256, num_layers=2,
            dropout=0.3, num_classes=2, pretrained=True, device=device,
        )
        vckpt = torch.load(vckpt_path, map_location=device)
        video_model.load_state_dict(vckpt.get("model_state_dict", vckpt))
    video_model.eval()

    from run_evaluation import _MultimodalEvalWrapper
    wrapper = _MultimodalEvalWrapper(model, video_model, video_backbone_ckpt)
    wrapper.eval()
    return wrapper


def evaluate_on_csv(model, csv_path: str, device: str, batch_size: int = 8):
    """Avalia o modelo em um CSV específico e retorna métricas."""
    _, _, test_loader = get_multimodal_dataloaders(
        video_data_root=str(p.PROCESSED_ROOT),
        pose_data_root=str(p.POSE_ROOT),
        emotion_data_root=str(p.EMOTION_ROOT),
        batch_size=batch_size,
        num_frames=16,
        window_size=16,
        video_mode="frames",
        pose_mode="keypoints",
        index_csv=str(csv_path),
    )

    calculator = MetricsCalculator(
        model, test_loader, device=device,
        class_names=["Non-Violent", "Violent"], num_classes=2,
    )
    metrics, y_true, y_pred, y_proba = calculator.evaluate()
    return metrics, y_true, y_pred, y_proba, len(test_loader.dataset)


def generate_comparison_table(results: dict) -> str:
    """Gera tabela comparativa em texto."""
    header = f"{'Cenário':<35} {'Accuracy':>10} {'F1 Macro':>10} {'Precision':>10} {'Recall':>10} {'AUC-ROC':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for scenario, data in results.items():
        m = data["metrics"]
        line = (
            f"{scenario:<35} "
            f"{m['accuracy']:>10.4f} "
            f"{m['f1_score']['macro']:>10.4f} "
            f"{m['precision']['macro']:>10.4f} "
            f"{m['recall']['macro']:>10.4f} "
            f"{m.get('auc_roc', 0.0):>10.4f}"
        )
        lines.append(line)

    lines.append(sep)

    # Delta entre baseline e cross-label
    if "baseline (congruente)" in results:
        baseline_f1 = results["baseline (congruente)"]["metrics"]["f1_score"]["macro"]
        baseline_acc = results["baseline (congruente)"]["metrics"]["accuracy"]
        lines.append("")
        lines.append("Deltas vs baseline:")
        for scenario, data in results.items():
            if scenario == "baseline (congruente)":
                continue
            m = data["metrics"]
            delta_f1 = m["f1_score"]["macro"] - baseline_f1
            delta_acc = m["accuracy"] - baseline_acc
            sign_f1 = "+" if delta_f1 >= 0 else ""
            sign_acc = "+" if delta_acc >= 0 else ""
            lines.append(
                f"  {scenario:<33} "
                f"ΔF1={sign_f1}{delta_f1:.4f}  "
                f"ΔAcc={sign_acc}{delta_acc:.4f}"
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
        m = data["metrics"]
        cm = m.get("confusion_matrix", {})
        tn = cm.get("tn", 0)
        fp = cm.get("fp", 0)
        fn = cm.get("fn", 0)
        tp = cm.get("tp", 0)
        lines.append(f"{scenario:<35} {tn:>6} {fp:>6} {fn:>6} {tp:>6}")

    lines.append(sep)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Impact Study: Avaliação cross-label (emoção × vídeo)"
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
    print()

    # ── 1. Gerar CSVs cross-label ──────────────────────────────────────────
    print("Gerando CSVs cross-label...")

    # Baseline (congruente)
    print("  [1/3] Baseline (congruente)...")
    baseline_rows = build_index(split="all", seed=args.seed)
    baseline_csv = output_dir / "pipeline_baseline.csv"
    save_csv(baseline_rows, str(baseline_csv))

    # violent_face + non_violent_video
    print("  [2/3] violent_face + non_violent_video...")
    vf_rows = build_cross_label_index(
        scenario="violent_face_non_violent_video",
        split="all", seed=args.seed,
    )
    vf_csv = output_dir / "pipeline_violent_face.csv"
    save_csv(vf_rows, str(vf_csv))

    # non_violent_face + violent_video
    print("  [3/3] non_violent_face + violent_video...")
    nvf_rows = build_cross_label_index(
        scenario="non_violent_face_violent_video",
        split="all", seed=args.seed,
    )
    nvf_csv = output_dir / "pipeline_non_violent_face.csv"
    save_csv(nvf_rows, str(nvf_csv))

    print()

    # ── 2. Carregar modelo ─────────────────────────────────────────────────
    print("Carregando modelo multimodal...")
    model = load_multimodal_model(args.model_path, args.device)
    print("✓ Modelo carregado\n")

    # ── 3. Avaliar em cada cenário ─────────────────────────────────────────
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
              f"f1(macro)={metrics['f1_score']['macro']:.4f}, "
              f"samples={n_samples}")

    # ── 4. Gerar relatório ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Resultados")
    print("=" * 60)

    comparison_table = generate_comparison_table(all_results)
    print(comparison_table)

    cm_summary = generate_confusion_matrix_summary(all_results)
    print("\n" + cm_summary)

    # Salvar resultados
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

    # Salvar métricas individuais por cenário
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
