"""
Script de treinamento para modelo multimodal de detecção de risco.

Hiperparâmetros: configs/treino.toml (seção [multimodal]).

Mitigação do atalho facial (cross-label):
- emotion_swap_prob: troca a face pela de label oposto (só train)
- emotion_dropout_prob/pose_dropout_prob: modal dropout (tokens learnable)
- aux_video_weight: loss auxiliar sobre o token de vídeo pré-atenção
- normalize_features: normaliza emotion/pose com stats do train
- monitor: val com emotion-swap forçado; best checkpoint e early stopping
  usam (val_acc + val_acc_swap)/2, selecionando invariância à face
- calibração: limiar de decisão (recall máximo sujeito a spec ≥ 0.73 em
  val+val_swap; fallback maximin) gravado no checkpoint como
  'decision_threshold' e aplicado na avaliação
- finetune_backbone (Fase F): fine-tune do CNN3D junto com a fusão
  (backbone_lr 10× menor); pesos do backbone salvos no best_model.pth
  como 'video_model_state_dict' e recarregados na avaliação

Uso:
    python train_multimodal.py
    python train_multimodal.py --epochs 10 --device cpu
"""

import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src import paths as p
from src.datasets.feature_stats import (
    compute_feature_stats,
    load_feature_stats,
    save_feature_stats,
)
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.evaluation.inference import collect_predictions
from src.evaluation.threshold import calibrate_threshold
from src.models.cnn3d_risk import create_cnn3d_model, ensure_channels_first
from src.models.multimodal_risk import create_multimodal_model
from src.training.config import add_config_args, resolve_config
from src.training.loop import train_loop


class MultimodalWithVideoBackbone(nn.Module):
    """Encapsula extração de features de vídeo (CNN 3D) + forward multimodal.

    finetune_backbone=False: extração sem grad (backbone congelado, Fase E).
    True (Fase F): grad flui para o CNN3D, fine-tuned com a fusão.
    head_only (Fase B): fase 1 do fine-tune em 2 estágios — backbone
    congelado (requires_grad=False, BN em eval) enquanto a fusão aprende;
    unfreeze_backbone() destrava p/ a fase 2.
    """

    def __init__(self, multimodal_model, video_model,
                 finetune_backbone: bool = True, head_only: bool = False):
        super().__init__()
        self.multimodal = multimodal_model
        self.video_model = video_model
        self.finetune_backbone = finetune_backbone
        self.head_only = bool(head_only and finetune_backbone)
        if self.head_only:
            self._set_backbone_requires_grad(False)

    def _set_backbone_requires_grad(self, requires_grad: bool) -> None:
        for param in self.video_model.parameters():
            param.requires_grad = requires_grad and self.finetune_backbone

    def unfreeze_backbone(self) -> None:
        """Destrava o backbone (fase 2 do fine-tune em 2 estágios)."""
        self.head_only = False
        self._set_backbone_requires_grad(True)

    def train(self, mode: bool = True):
        # run_epoch chama model.train(True) a cada época; na fase head-only
        # o vídeo deve permanecer em eval (BN congelado) mesmo assim.
        super().train(mode)
        if self.head_only:
            self.video_model.eval()
        return self

    def forward(self, video, pose, emotion, return_aux: bool = False):
        with torch.set_grad_enabled(self.finetune_backbone and not self.head_only):
            if video.dim() == 5:
                video = self.video_model.get_features(ensure_channels_first(video))
        return self.multimodal(video, pose, emotion, return_aux=return_aux)


class AuxTrainWrapper(nn.Module):
    """
    Wrapper de treino com loss auxiliar só-vídeo.

    Devolve apenas os logits fused (compatível com run_epoch); guarda
    aux_logits para o AuxCrossEntropyLoss somar λ·CE(video_aux).
    Durante validação (eval), aux_logits fica None -> loss = CE puro.
    """

    def __init__(self, model: MultimodalWithVideoBackbone, aux_weight: float):
        super().__init__()
        self.model = model
        self.aux_weight = aux_weight
        self.aux_logits = None

    def forward(self, video, pose, emotion):
        self.aux_logits = None
        if self.training and self.aux_weight > 0:
            logits, aux = self.model(video, pose, emotion, return_aux=True)
            self.aux_logits = aux
            return logits
        return self.model(video, pose, emotion)


class AuxCrossEntropyLoss(nn.Module):
    """CE(fused) + λ·CE(video_aux). Aux ausente (validação) -> CE puro."""

    def __init__(self, aux_weight: float, aux_source: AuxTrainWrapper,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.aux_weight = aux_weight
        self.aux_source = aux_source
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, outputs, labels):
        loss = self.ce(outputs, labels)
        aux = self.aux_source.aux_logits
        if aux is not None:
            loss = loss + self.aux_weight * self.ce(aux, labels)
        return loss


def _multimodal_hook(batch):
    return (batch[0], batch[1], batch[2]), batch[3]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_feature_stats(cfg) -> dict | None:
    """Carrega (ou calcula e salva) stats do train para normalização."""
    if not cfg.get("normalize_features", False):
        return None
    stats_path = p.MULTIMODAL_FEATURE_STATS
    stats = load_feature_stats(stats_path)
    if stats is None:
        print(f"Calculando feature stats do train -> {stats_path}")
        from src.datasets.paired_dataset import PairedSurveillanceDataset
        probe = PairedSurveillanceDataset(
            split="train",
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
        )
        stats = compute_feature_stats(probe)
        save_feature_stats(stats, stats_path)
    print(f"✓ Feature stats: {sorted(stats.keys())}")
    return stats


def _infer_pose_feature_dim(dataset) -> int:
    """Dimensão flatten da pose (joints*3); fallback = default do dataset (zeros)."""
    import numpy as np
    from src.datasets.paired_dataset import DEFAULT_POSE_JOINTS
    for sample in dataset.samples:
        path = sample["pair_dir"] / "pose.npy"
        if path.exists():
            return int(np.load(path, mmap_mode="r").shape[1]) * 3
    # Sem pose.npy o dataset usa zeros (DEFAULT_POSE_JOINTS, 3).
    return DEFAULT_POSE_JOINTS * 3


def _calibrate_decision_threshold(
    multimodal_model: torch.nn.Module,
    train_model: torch.nn.Module,
    loaders: list,
    device,
    output_dir,
) -> float:
    """
    Calibra o limiar de decisão no fim do treino e grava no best_model.pth.

    Sweep sobre (val [+ val_swap]) combinados: val_swap simula a condição
    cross-label (face trocada), ancorando o limiar nas células do teste.
    Critério (D4b): recall máximo sujeito a spec ≥ 0.73 (margem sobre a
    meta 0.70 p/ o shift val→teste); fallback maximin.
    """
    best_path = output_dir / "best_model.pth"
    best_ckpt = torch.load(best_path, map_location="cpu")
    multimodal_model.load_state_dict(best_ckpt["model_state_dict"])
    # Com fine-tune (Fase F), o backbone salvo no melhor checkpoint difere
    # dos pesos finais em memória — recarrega p/ calibrar com o modelo real.
    video_model = getattr(train_model, "video_model", None)
    if video_model is None:
        video_model = getattr(getattr(train_model, "model", None), "video_model", None)
    if video_model is not None and "video_model_state_dict" in best_ckpt:
        video_model.load_state_dict(best_ckpt["video_model_state_dict"])
        print("✓ Backbone CNN3D do melhor checkpoint recarregado p/ calibração")

    ys, ps = [], []
    for loader in loaders:
        y_true, _, y_proba = collect_predictions(
            train_model, loader, device, keep_full_probs=False,
        )
        ys.append(y_true)
        ps.append(y_proba)
    y_true = np.concatenate(ys)
    y_proba = np.concatenate(ps)

    result = calibrate_threshold(y_true, y_proba)
    t = result["threshold"]

    print("━━━ Calibração do limiar de decisão ━━━")
    print(f"  critério={result['criterion']} (spec_target={result['spec_target']:.2f})")
    print(f"  n={len(y_true)} | t={t:.4f}", end="")
    if result["recall"] is not None:
        print(f" | recall={result['recall']:.4f} | spec={result['spec']:.4f} "
              f"| acc={result['accuracy']:.4f}")
    else:
        print(" (val single-class — fallback 0.5)")

    best_ckpt["decision_threshold"] = t
    torch.save(best_ckpt, best_path)
    print(f"✓ decision_threshold={t:.4f} gravado em {best_path}")
    return t


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Treinar modelo multimodal de detecção de risco (configs/treino.toml)"
    )
    add_config_args(parser)
    return parser


def main():
    args = build_parser().parse_args()
    cfg = resolve_config("multimodal", args)

    _set_seed(cfg.get("seed", 42))

    output_dir = p.MULTIMODAL_WEIGHTS
    experiments_dir = p.MULTIMODAL_EXPERIMENTS

    aux_video_weight = cfg.get("aux_video_weight", 0.0)
    emotion_dropout_prob = cfg.get("emotion_dropout_prob", 0.0)
    pose_dropout_prob = cfg.get("pose_dropout_prob", 0.0)
    emotion_swap_prob = cfg.get("emotion_swap_prob", 0.0)
    finetune_backbone = cfg.get("finetune_backbone", True)
    backbone_lr = cfg.get("backbone_lr", 1e-5)
    weight_decay = cfg.get("weight_decay", 0.0)
    backbone_weight_decay = cfg.get("backbone_weight_decay", weight_decay)
    head_only_epochs = cfg.get("head_only_epochs", 0) if finetune_backbone else 0

    print("=" * 60)
    print("Treinamento do Modelo Multimodal")
    print("=" * 60)
    print(f"Fusion method: cross_attention")
    print(f"Temporal modeling: {cfg['use_temporal_modeling']}")
    print(f"Device: {cfg['device']} | Épocas: {cfg['epochs']} | "
          f"Batch: {cfg['batch_size']} | LR: {cfg['learning_rate']}")
    print(f"Mitigação de atalho: swap={emotion_swap_prob} "
          f"drop(emotion)={emotion_dropout_prob} drop(pose)={pose_dropout_prob} "
          f"aux_video={aux_video_weight} "
          f"norm={cfg.get('normalize_features', False)}")
    if not finetune_backbone:
        print("Backbone CNN3D: congelado (comportamento Fase E)")
    elif head_only_epochs > 0:
        print(f"Backbone CNN3D: FINE-TUNE 2 ESTÁGIOS "
              f"(head-only {head_only_epochs} ép → lr={backbone_lr:.0e}, "
              f"wd={backbone_weight_decay:.0e})")
    else:
        print(f"Backbone CNN3D: FINE-TUNE direto (lr={backbone_lr:.0e})")
    print("=" * 60)

    print("Carregando modelo de vídeo...")
    video_model = create_cnn3d_model(
        num_classes=2,
        checkpoint_path=cfg["video_model_path"],
        num_frames=cfg["num_frames"],
        device=cfg["device"],
    )
    video_model.eval()
    print(f"Modelo de vídeo CNN 3D carregado de: {cfg['video_model_path']}")

    feature_stats = None
    pose_feature_dim = 51
    monitor_loader = None
    print("Criando DataLoaders...")
    from src.datasets.paired_dataset import (
        PairedSurveillanceDataset,
        get_paired_dataloaders,
        paired_root_exists,
    )
    if paired_root_exists():
        print(f"✓ Dataset pareado: {p.PAIRED_ROOT}")
        feature_stats = _resolve_feature_stats(cfg)
        train_loader, val_loader, _ = get_paired_dataloaders(
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
            feature_stats=feature_stats,
            emotion_swap_prob=emotion_swap_prob,
        )
        pose_feature_dim = _infer_pose_feature_dim(train_loader.dataset)
        print(f"✓ Pose feature dim inferida: {pose_feature_dim}")

        # Monitor: val com emotion-swap forçado (invariância à face).
        # best checkpoint + early stopping passam a usar
        # monitor_score = (val_acc + val_acc_swap)/2 no train_loop.
        if emotion_swap_prob > 0:
            from torch.utils.data import DataLoader

            val_swap_dataset = PairedSurveillanceDataset(
                split="val",
                num_frames=cfg["num_frames"],
                window_size=cfg["window_size"],
                video_mode="frames",
                pose_mode="flatten",
                feature_stats=feature_stats,
                force_emotion_swap=True,
            )
            monitor_loader = DataLoader(
                val_swap_dataset,
                batch_size=cfg["batch_size"],
                shuffle=False,
                num_workers=cfg["num_workers"],
                pin_memory=torch.cuda.is_available(),
            )
            print(f"✓ Monitor val trocado: {len(val_swap_dataset)} amostras "
                  f"(swap prob={val_swap_dataset.emotion_swap_prob:.1f})")
    else:
        print("⚠️  dataset/paired não encontrado — usando CSV legado "
              "(gerar com: python -m src.preprocessing.build_paired_dataset)")
        train_loader, val_loader, _ = get_multimodal_dataloaders(
            video_data_root=str(p.PROCESSED_ROOT),
            pose_data_root=str(p.POSE_ROOT),
            emotion_data_root=str(p.EMOTION_ROOT),
            batch_size=cfg["batch_size"],
            num_frames=cfg["num_frames"],
            window_size=cfg["window_size"],
            video_mode="frames",
            pose_mode="flatten",
            num_workers=cfg["num_workers"],
            dataset_name="rwf2000",
            index_csv=str(p.PIPELINE_CSV_PATH) if p.PIPELINE_CSV_PATH.exists() else None,
        )

    print("Criando modelo multimodal...")
    print(f"Monitor de seleção: "
          f"{'val + val-swapped (swap forçado)' if monitor_loader is not None else 'val apenas'}")
    multimodal_model = create_multimodal_model(
        video_feature_dim=video_model.feature_dim,
        pose_feature_dim=pose_feature_dim,
        emotion_feature_dim=128,
        num_frames=cfg["window_size"],
        fusion_method="cross_attention",
        use_temporal_modeling=cfg["use_temporal_modeling"],
        emotion_dropout_prob=emotion_dropout_prob,
        pose_dropout_prob=pose_dropout_prob,
        aux_video_weight=aux_video_weight,
        device=cfg["device"],
    )
    model = MultimodalWithVideoBackbone(
        multimodal_model, video_model,
        finetune_backbone=finetune_backbone,
        head_only=head_only_epochs > 0,
    ).to(torch.device(cfg["device"]))

    if aux_video_weight > 0:
        train_model = AuxTrainWrapper(model, aux_video_weight).to(torch.device(cfg["device"]))
        criterion = AuxCrossEntropyLoss(
            aux_video_weight, train_model,
            label_smoothing=cfg.get("label_smoothing", 0.0),
        )
    else:
        train_model = model
        criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.0)
        )

    if finetune_backbone:
        optimizer = optim.Adam([
            {"params": video_model.parameters(), "lr": backbone_lr,
             "weight_decay": backbone_weight_decay},
            {"params": multimodal_model.parameters(), "lr": cfg["learning_rate"],
             "weight_decay": weight_decay},
        ])
        print(f"Otimizer: 2 grupos (backbone lr={backbone_lr:.0e} "
              f"wd={backbone_weight_decay:.0e}, "
              f"fusão lr={cfg['learning_rate']:.0e} wd={weight_decay:.0e})")
    else:
        optimizer = optim.Adam(
            multimodal_model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=weight_decay,
        )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    def _on_epoch_start(epoch: int) -> None:
        if head_only_epochs > 0 and epoch == head_only_epochs + 1:
            model.unfreeze_backbone()
            print(f"\n→ Backbone CNN3D destravado — fase 2 de fine-tune "
                  f"(época {epoch})")

    best_val_acc, _ = train_loop(
        train_model, train_loader, val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=torch.device(cfg["device"]), epochs=cfg["epochs"],
        output_dir=output_dir, experiments_dir=experiments_dir,
        model_hook=_multimodal_hook, config=cfg,
        early_stop_patience=cfg.get("early_stop_patience"),
        checkpoint_model=multimodal_model,
        monitor_loader=monitor_loader,
        on_epoch_start=_on_epoch_start,
        checkpoint_extra={
            "fusion_method": "cross_attention",
            "use_temporal_modeling": cfg["use_temporal_modeling"],
            "emotion_feature_dim": multimodal_model.emotion_feature_dim,
            "video_feature_dim": multimodal_model.video_feature_dim,
            "video_backbone": "cnn3d",
            "emotion_swap_prob": emotion_swap_prob,
            "emotion_dropout_prob": emotion_dropout_prob,
            "pose_dropout_prob": pose_dropout_prob,
            "aux_video_weight": aux_video_weight,
            "normalize_features": cfg.get("normalize_features", False),
            "finetune_backbone": finetune_backbone,
            "head_only_epochs": head_only_epochs,
            # Pesos do backbone no momento da época vencedora (referências
            # vivas serializadas no save) — avaliação recarrega isto.
            "video_model_state_dict": video_model.state_dict(),
        },
    )

    # Calibração do limiar de decisão (Fase E): sweep em val [+ val_swap]
    # com o melhor checkpoint; gravado como 'decision_threshold' e aplicado
    # automaticamente pelos scripts de avaliação.
    calibration_loaders = [val_loader]
    if monitor_loader is not None:
        calibration_loaders.append(monitor_loader)
    _calibrate_decision_threshold(
        multimodal_model, train_model, calibration_loaders,
        torch.device(cfg["device"]), output_dir,
    )

    print("=" * 60)
    print("Treinamento concluído!")
    if monitor_loader is not None:
        print(f"Melhor monitor_score (val + val-swapped)/2: {best_val_acc:.2f}%")
    else:
        print(f"Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"Modelo salvo em: {output_dir / 'best_model.pth'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
