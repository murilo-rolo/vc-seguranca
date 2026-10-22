"""
Recalibra o limiar de decisão do melhor checkpoint SEM retrain (D4c).

O critério do treino (spec ≥ alvo em val+val_swap POOLADOS) sofre o shift
val→teste: o val limpo domina a média, o limiar sobe e o recall do teste
cai (Fase F: 0.64 embora a curva permita ~0.75). Aqui a constraint vale
POR SUBCONJUNTO (val e val_swap separadamente), com alvo que compensa o
shift medido (~+0.04 de spec do val para o teste => default spec_target
0.75 ≈ meta 0.70 + margem).

Carrega o checkpoint, coleta probabilidades em val e val_swap
(emotion-swap forçado) com o mesmo pipeline do treino, varre o limiar e
grava 'decision_threshold' no checkpoint. A avaliação (run_impact_study
etc.) aplica o limiar automaticamente.

Uso:
    python recalibrate_threshold.py --dry-run            # só imprime o sweep
    python recalibrate_threshold.py                      # grava no checkpoint
    python recalibrate_threshold.py --spec-target 0.76   # alvo maior (mais spec)
    python recalibrate_threshold.py --threshold 0.55     # limiar manual
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import paths as p
from src.datasets.paired_dataset import (
    PairedSurveillanceDataset,
    get_paired_dataloaders,
    paired_root_exists,
)
from src.evaluation.inference import collect_predictions
from src.evaluation.model_loading import (
    load_feature_stats_for_checkpoint,
    load_model,
)
from src.evaluation.threshold import (
    DEFAULT_SPEC_TARGET,
    calibrate_threshold_per_subset,
)

SUBSET_NAMES = ("val", "val_swap")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recalibra decision_threshold por subset (sem retrain)"
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default=str(p.MULTIMODAL_WEIGHTS / "best_model.pth"),
        help="Checkpoint multimodal (padrão: best_model.pth)",
    )
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--spec-target", type=float, default=DEFAULT_SPEC_TARGET,
        help=f"Spec mínimo POR SUBCONJUNTO (padrão: {DEFAULT_SPEC_TARGET}; "
             "0.75 compensa o shift val→teste medido ~+0.04)",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Grava este limiar direto (ignora o sweep) — útil p/ iteração manual",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Apenas imprime o sweep; não grava no checkpoint",
    )
    return parser


def build_calibration_loaders(cfg: dict, feature_stats):
    """Val + val_swap com o mesmo pipeline do treino (pose_mode=flatten)."""
    num_frames = cfg.get("num_frames", 16)
    window_size = cfg.get("window_size", 16)
    batch_size = cfg.get("batch_size", 16)
    num_workers = cfg.get("num_workers", 2)

    _, val_loader, _ = get_paired_dataloaders(
        batch_size=batch_size,
        num_frames=num_frames,
        window_size=window_size,
        video_mode="frames",
        pose_mode="flatten",
        num_workers=num_workers,
        feature_stats=feature_stats,
    )
    val_swap_dataset = PairedSurveillanceDataset(
        split="val",
        num_frames=num_frames,
        window_size=window_size,
        video_mode="frames",
        pose_mode="flatten",
        feature_stats=feature_stats,
        force_emotion_swap=True,
    )
    val_swap_loader = DataLoader(
        val_swap_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"✓ Val: {len(val_loader.dataset)} amostras | "
          f"Val-swapped: {len(val_swap_dataset)} amostras")
    return [val_loader, val_swap_loader]


def collect_subset_probs(model, loaders, device):
    subsets = []
    for loader in loaders:
        y_true, _, y_proba = collect_predictions(
            model, loader, device, keep_full_probs=False,
        )
        subsets.append((y_true, y_proba))
    return subsets


def print_result(result: dict, subsets, names=SUBSET_NAMES) -> None:
    print("━━━ Recalibração por subset (D4c) ━━━")
    print(f"  critério={result['criterion']} "
          f"(spec_target={result['spec_target']:.2f})")
    t = result["threshold"]
    print(f"  t={t:.4f} | min_recall={result['min_recall']:.4f} | "
          f"min_spec={result['min_spec']:.4f} | "
          f"mean_acc={result['mean_accuracy']:.4f}")
    for name, subset in zip(names, result["per_subset"]):
        y = np.asarray(subsets[names.index(name)][0]).astype(int)
        rec = (f"{subset['recall']:.4f}" if subset["recall"] is not None else "  n/a")
        spec = (f"{subset['spec']:.4f}" if subset["spec"] is not None else "  n/a")
        print(f"    {name:<10} n={y.size:<4d} recall={rec} spec={spec} "
              f"acc={subset['accuracy']:.4f}")


def print_sweep(subsets, spec_target: float, names=SUBSET_NAMES) -> None:
    """Impreme alguns pontos do sweep p/ inspeção (dry-run)."""
    print("\n  pontos com min_spec mais próximo do alvo:")
    rows = []
    for t in np.linspace(0.0, 1.0, 201):
        recalls, specs = [], []
        for y_true, scores in subsets:
            y = np.asarray(y_true).astype(int)
            pred = scores >= t
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            if n_pos:
                recalls.append(float((pred & (y == 1)).sum()) / n_pos)
            if n_neg:
                specs.append(float(((~pred) & (y == 0)).sum()) / n_neg)
        if not recalls or not specs:
            continue
        rows.append((t, min(recalls), min(specs)))
    rows.sort(key=lambda r: (abs(r[2] - spec_target), -r[1]))
    for t, mr, ms in rows[:8]:
        flag = "✓" if ms >= spec_target else " "
        print(f"    {flag} t={t:.3f}  min_recall={mr:.3f}  min_spec={ms:.3f}")


def main():
    args = build_parser().parse_args()
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
    if not paired_root_exists():
        raise FileNotFoundError(
            f"Dataset pareado não encontrado em {p.PAIRED_ROOT}. "
            "Gere com: python -m src.preprocessing.build_paired_dataset"
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("hyperparameters", {})
    cfg.setdefault("batch_size", args.batch_size)
    cfg.setdefault("num_workers", args.num_workers)
    old_threshold = checkpoint.get("decision_threshold")
    print(f"Checkpoint: {ckpt_path}")
    print(f"decision_threshold atual: "
          f"{old_threshold if old_threshold is not None else '(ausente → 0.5)'}")

    feature_stats = load_feature_stats_for_checkpoint(str(ckpt_path))
    print(f"Feature stats: "
          f"{'carregadas' if feature_stats else 'AUSENTES (normalize_features?)'}")

    model = load_model("multimodal", str(ckpt_path), args.device)
    loaders = build_calibration_loaders(cfg, feature_stats)
    subsets = collect_subset_probs(model, loaders, args.device)

    if args.threshold is not None:
        t = float(args.threshold)
        print(f"\n━━━ Limiar manual t={t:.4f} ━━━")
        for name, (y_true, scores) in zip(SUBSET_NAMES, subsets):
            y = np.asarray(y_true).astype(int)
            pred = scores >= t
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            rec = float((pred & (y == 1)).sum()) / n_pos if n_pos else None
            spec = float(((~pred) & (y == 0)).sum()) / n_neg if n_neg else None
            print(f"  {name:<10} recall={rec:.4f} spec={spec:.4f}" if rec is not None
                  else f"  {name:<10} single-class")
        new_threshold = t
    else:
        result = calibrate_threshold_per_subset(
            subsets, spec_target=args.spec_target,
        )
        print_result(result, subsets)
        if args.dry_run:
            print_sweep(subsets, args.spec_target)
        new_threshold = result["threshold"]

    if args.dry_run:
        print(f"\n(dry-run) novo limiar seria {new_threshold:.4f} — nada gravado")
        return

    if old_threshold is not None and abs(old_threshold - new_threshold) < 1e-9:
        print(f"\nLimiar inalterado ({new_threshold:.4f}) — checkpoint não regravado")
        return

    checkpoint["decision_threshold"] = new_threshold
    torch.save(checkpoint, ckpt_path)
    print(f"\n✓ decision_threshold={new_threshold:.4f} gravado em {ckpt_path}")
    print("Rode a avaliação p/ conferir: python run_impact_study.py --charts --skip_validate")


if __name__ == "__main__":
    main()
