"""
Script principal para executar todos os experimentos de avaliação.

Uso:
    # Avaliar baseline
    python run_evaluation.py --model baseline --model_path results/models/best_model.pth
    
    # Avaliar multimodal
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth
    
    # Avaliar CNN 3D
    python run_evaluation.py --model cnn3d --model_path models/cnn3d/weights/best_model.pth
    
    # Avaliar sub-modelo de vídeo (ResNet-LSTM) independentemente
    python run_evaluation.py --model video --model_path models/resnet_lstm/weights/best_model.pth --metrics
    
    # Avaliar sub-modelo de pose (branch do multimodal) independentemente
    python run_evaluation.py --model pose --model_path models/multimodal/weights/best_model.pth --metrics
    
    # Avaliar sub-modelo de emoção (EmotionNet, 8 classes) independentemente
    python run_evaluation.py --model emotion --model_path models/emotion_cnn/weights/best_model.pth --metrics
    
    # Avaliar TODOS os sub-modelos (video/pose/emotion) + gerar gráficos
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth --all --charts
    
    # Estudo de impacto no modelo fusionado
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth --impact_study --noise_std 0.05
    
    # Executar todos os experimentos
    python run_evaluation.py --model multimodal --model_path results/multimodal/best_model.pth --all
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import json

from src.evaluation.metrics import MetricsCalculator
from src.evaluation.robustness_eval import RobustnessEvaluator, DEFAULT_DISTORTION_CONFIGS
from src.evaluation.performance_eval import PerformanceEvaluator
from src.evaluation.limitations_analysis import LimitationsAnalyzer
from src.evaluation.ablation_study import AblationStudy
from src.evaluation.charts import generate_report
from src.models.resnet_lstm import create_model as create_video_model
from src.models.multimodal_risk import create_multimodal_model
from src.models.cnn3d_risk import create_cnn3d_model
from src.models.emotion_cnn import create_emotion_model
from src.datasets.surveillance_dataset import get_dataloaders
from src.datasets.multimodal_dataset import get_multimodal_dataloaders
from src.datasets.video3d_dataset import get_rwf2000_3d_dataloaders
from src.pose.pose_dataset import get_pose_dataloaders
from src import paths as p


class PoseSubModel(nn.Module):
    """
    Sub-modelo de pose para avaliação standalone (F5, EVAL-T1).

    Replica o branch de pose do MultimodalRiskDetector: processador temporal
    (LSTM ou MLP) -> projeção -> camadas de fusão -> classificador binário.
    É construído a partir das dimensões inferidas do checkpoint multimodal
    e carregado com strict=False (somente as chaves pose/fusão/classificador).
    """

    def __init__(
        self,
        pose_feature_dim: int = 99,
        pose_hidden_dim: int = 64,
        fusion_dim: int = 256,
        use_temporal_modeling: bool = True,
        num_classes: int = 2,
    ):
        super().__init__()
        if use_temporal_modeling:
            self.pose_processor = nn.LSTM(
                pose_feature_dim, pose_hidden_dim, num_layers=1,
                batch_first=True, dropout=0
            )
            pose_output_dim = pose_hidden_dim
        else:
            self.pose_processor = nn.Sequential(
                nn.Linear(pose_feature_dim, pose_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
            )
            pose_output_dim = pose_hidden_dim
        self.pose_proj = nn.Linear(pose_output_dim, fusion_dim)
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pose: (B, T, num_joints, 3) ou (B, T, D_p)
        Returns:
            Logits (B, num_classes)
        """
        if pose.ndim == 4 and pose.shape[-1] == 3:
            pose = pose.view(pose.shape[0], pose.shape[1], -1)
        elif pose.ndim == 2:
            pose = pose.unsqueeze(1)

        if isinstance(self.pose_processor, nn.LSTM):
            lstm_out, _ = self.pose_processor(pose)
            last = lstm_out[:, -1, :]
        else:
            B, T, D = pose.shape
            h = self.pose_processor(pose.reshape(B * T, D))
            last = h.reshape(B, T, -1).mean(dim=1)

        proj = self.pose_proj(last)
        return self.classifier(self.fusion_layers(proj))


class _MultimodalEvalWrapper(nn.Module):
    """Envolve multimodal + backbone de vídeo para avaliação.

    O dataloader multimodal fornece frames brutos (B, T, C, H, W); o backbone
    extrai o clip token (B, D_v) antes do forward multimodal (cross-attention).
    """
    def __init__(self, multimodal, video_model, video_backbone: str = "cnn3d"):
        super().__init__()
        self.multimodal = multimodal
        self.video_model = video_model
        self.video_backbone = video_backbone

    def forward(self, video, pose, emotion):
        with torch.no_grad():
            if len(video.shape) == 5:
                # Frames em frame-last (B, T, C, H, W); CNN3D espera (B, C, T, H, W)
                if self.video_backbone == "cnn3d":
                    video = video.permute(0, 2, 1, 3, 4)
                video = self.video_model.get_features(video)  # (B, D_v) clip token
        return self.multimodal(video, pose, emotion)


def _infer_pose_submodel_dims(state_dict) -> dict:
    """Infere as dimensões do sub-modelo de pose a partir do state_dict."""
    dims = {"use_temporal_modeling": True, "pose_feature_dim": 99,
            "pose_hidden_dim": 64, "fusion_dim": 256, "num_classes": 2}

    if "pose_processor.weight_ih_l0" in state_dict:  # LSTM
        w = state_dict["pose_processor.weight_ih_l0"]
        dims["pose_feature_dim"] = w.shape[1]
        dims["pose_hidden_dim"] = w.shape[0] // 4
        dims["use_temporal_modeling"] = True
    elif "pose_processor.0.weight" in state_dict:  # MLP
        w = state_dict["pose_processor.0.weight"]
        dims["pose_feature_dim"] = w.shape[1]
        dims["pose_hidden_dim"] = w.shape[0]
        dims["use_temporal_modeling"] = False

    if "fusion_layers.0.weight" in state_dict:
        dims["fusion_dim"] = state_dict["fusion_layers.0.weight"].shape[0]
    if "classifier.weight" in state_dict:
        dims["num_classes"] = state_dict["classifier.weight"].shape[0]

    return dims


def load_model(
    model_type: str,
    model_path: str,
    device: str,
    video_backbone: str = "cnn3d",
    video_model_path: Optional[str] = None,
):
    """Carrega modelo do checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if model_type == "baseline":
        model = create_video_model(
            num_frames=16,
            hidden_size=256,
            num_layers=2,
            dropout=0.3,
            num_classes=2,
            pretrained=True,
            device=device
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    elif model_type == "multimodal":
        fusion_method = checkpoint.get('fusion_method', 'cross_attention')
        use_temporal = checkpoint.get('use_temporal_modeling', True)
        # Dimensão/backbone de vídeo lidos do checkpoint (AD-013: cnn3d default)
        video_backbone_ckpt = checkpoint.get('video_backbone', 'cnn3d')
        video_feature_dim = checkpoint.get('video_feature_dim')
        if video_feature_dim is None:
            video_feature_dim = 512 if video_backbone_ckpt == 'cnn3d' else 256

        model = create_multimodal_model(
            video_feature_dim=video_feature_dim,
            pose_feature_dim=99,
            emotion_feature_dim=128,
            num_frames=16,
            fusion_method=fusion_method,
            use_temporal_modeling=use_temporal,
            device=device
        )
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        # Backbone de vídeo para extrair o clip token dos frames do dataloader
        if video_backbone_ckpt == "cnn3d":
            vckpt_path = video_model_path or str(p.CNN3D_WEIGHTS / "best_model.pth")
            vckpt = torch.load(vckpt_path, map_location=device)
            vmodel_name = vckpt.get('model_name') or vckpt.get('backbone') or "r2plus1d_18"
            video_model = create_cnn3d_model(
                model_name=vmodel_name,
                num_classes=2,
                checkpoint_path=vckpt_path,
                device=device
            )
        else:
            vckpt_path = video_model_path or str(p.RESNET_LSTM_WEIGHTS / "best_model.pth")
            video_model = create_video_model(
                num_frames=16,
                hidden_size=256,
                num_layers=2,
                dropout=0.3,
                num_classes=2,
                pretrained=True,
                device=device
            )
            vckpt = torch.load(vckpt_path, map_location=device)
            video_model.load_state_dict(vckpt.get('model_state_dict', vckpt))
        video_model.eval()

        model = _MultimodalEvalWrapper(model, video_model, video_backbone_ckpt)
    
    elif model_type == "cnn3d":
        # Backbone selecionado pela metadata do checkpoint (model_name/backbone),
        # com fallback para o default r2plus1d_18 (pesos Kinetics-400).
        model_name = checkpoint.get('model_name') or checkpoint.get('backbone') or "r2plus1d_18"
        model = create_cnn3d_model(
            model_name=model_name,
            num_classes=2,
            checkpoint_path=model_path,
            device=device
        )
    
    elif model_type == "video":
        # Sub-modelo de vídeo: CNN3D (padrão) ou ResNet-LSTM avaliado standalone
        if video_backbone == "cnn3d":
            model_name = checkpoint.get('model_name') or checkpoint.get('backbone') or "r2plus1d_18"
            model = create_cnn3d_model(
                model_name=model_name,
                num_classes=2,
                checkpoint_path=model_path,
                device=device
            )
        else:
            model = create_video_model(
                num_frames=16,
                hidden_size=256,
                num_layers=2,
                dropout=0.3,
                num_classes=2,
                pretrained=True,
                device=device
            )
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict)
    
    elif model_type == "pose":
        # Sub-modelo de pose: branch de pose do multimodal, avaliado standalone.
        # O checkpoint é o multimodal (ou um checkpoint de pose dedicado).
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        dims = _infer_pose_submodel_dims(state_dict)
        model = PoseSubModel(
            pose_feature_dim=dims["pose_feature_dim"],
            pose_hidden_dim=dims["pose_hidden_dim"],
            fusion_dim=dims["fusion_dim"],
            use_temporal_modeling=dims["use_temporal_modeling"],
            num_classes=dims["num_classes"],
        ).to(device)
        loaded = model.load_state_dict(state_dict, strict=False)
        if loaded.missing_keys and "pose_processor.weight_ih_l0" in loaded.missing_keys:
            print(
                "⚠️  Aviso: checkpoint sem as chaves do branch de pose "
                "(pose_processor/pose_proj/fusion_layers/classifier). "
                "O sub-modelo de pose usará pesos aleatórios."
            )
    
    elif model_type == "emotion":
        # Sub-modelo de emoção: EmotionNet com a cabeça de classificação (8 classes)
        model = create_emotion_model(
            num_emotions=8,
            pretrained=True,
            checkpoint_path=model_path,
            device=device
        )
    
    else:
        raise ValueError(f"Tipo de modelo não suportado: {model_type}")
    
    model.eval()
    return model


def _resolve_sub_model_path(args, model_type: str) -> str:
    """Resolve o caminho do checkpoint para um sub-modelo (F5, EVAL-T1)."""
    explicit = getattr(args, f"{model_type}_model_path", None)
    if explicit:
        return explicit

    defaults = {
        # video: CNN 3D (padrão) ou ResNet-LSTM, conforme --video_backbone
        "video": (str(p.CNN3D_WEIGHTS / "best_model.pth")
                  if getattr(args, "video_backbone", "cnn3d") == "cnn3d"
                  else str(p.RESNET_LSTM_WEIGHTS / "best_model.pth")),
        # pose: branch de pose do checkpoint multimodal (--model_path)
        "pose": args.model_path,
        # emotion: EmotionNet treinado
        "emotion": str(p.EMOTION_CNN_WEIGHTS / "best_model.pth"),
    }
    return defaults[model_type]


def _get_per_model_test_loader(model_type: str, batch_size: int, video_backbone: str = "cnn3d"):
    """
    Retorna (test_loader, class_names, num_classes) para um sub-modelo.
    """
    if model_type == "video":
        if video_backbone == "cnn3d":
            # Clipes 3D do RWF-2000 (permutação (B,T,C,H,W)->(B,C,T,H,W) no MetricsCalculator)
            _, _, test_loader = get_rwf2000_3d_dataloaders(
                dataset_root=str(p.RWF2000_ROOT),
                batch_size=batch_size,
                num_frames=16,
                clip_size=(112, 112)
            )
        else:
            # Frames processados (B, T, C, H, W) + label binário
            _, _, test_loader = get_dataloaders(
                processed_data_root=str(p.PROCESSED_ROOT),
                batch_size=batch_size,
                num_frames=16,
                num_workers=0
            )
        return test_loader, ["Non-Violent", "Violent"], 2

    if model_type == "pose":
        # Sequências de keypoints flatten (B, T, 99) + label binário
        _, _, test_loader = get_pose_dataloaders(
            pose_data_root=str(p.POSE_ROOT),
            batch_size=batch_size,
            window_size=16,
            flatten=True,
            num_workers=0
        )
        return test_loader, ["Non-Violent", "Violent"], 2

    if model_type == "emotion":
        # Imagens de face (B, 3, 224, 224) + label das 8 emoções (balanced-affectnet)
        from train_emotion_model import AffectNetDataset, get_transforms
        from torch.utils.data import DataLoader

        data_root = p.DATASET_ROOT / "balanced-affectnet"
        if not data_root.exists():
            raise FileNotFoundError(
                f"Dataset de emoção não encontrado em {data_root}. "
                "Baixe o balanced-affectnet (F3) antes de avaliar o sub-modelo de emoção."
            )
        dataset = AffectNetDataset(
            data_root, split="test", transform=get_transforms(is_train=False)
        )
        if len(dataset) == 0:
            raise RuntimeError(
                f"Nenhuma imagem de teste em {data_root / 'test'}. "
                "Baixe o balanced-affectnet (F3) antes de avaliar o sub-modelo de emoção."
            )
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return test_loader, list(dataset.class_names), len(dataset.class_names)

    raise ValueError(f"Sub-modelo não suportado: {model_type}")


def _build_roc_pr(model_type: str, y_true, y_proba, num_classes: int, class_names):
    """
    Monta a estrutura de curvas ROC/PR para os gráficos (EVAL-T2).

    Binário -> uma curva; multiclasse -> uma curva por classe (one-vs-rest).
    """
    from sklearn.metrics import roc_curve, precision_recall_curve
    from sklearn.metrics import roc_auc_score, average_precision_score

    if num_classes > 2:
        curves = []
        for c in range(num_classes):
            y_true_bin = (y_true == c).astype(int)
            y_score = y_proba[:, c]
            fpr, tpr, _ = roc_curve(y_true_bin, y_score)
            precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
            curve = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "auc": None,
                "ap": None,
                "label": str(class_names[c]),
            }
            try:
                curve["auc"] = float(roc_auc_score(y_true_bin, y_score))
            except ValueError:
                pass
            try:
                curve["ap"] = float(average_precision_score(y_true_bin, y_score))
            except ValueError:
                pass
            curves.append(curve)
        return curves

    # Binário
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    curve = {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "auc": None,
        "ap": None,
        "label": model_type,
    }
    try:
        curve["auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        pass
    try:
        curve["ap"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pass
    return curve


def run_per_model_evaluation(args, device, targets: list = None):
    """
    Avalia sub-modelos (video/pose/emotion) independentemente (F5, EVAL-T1).

    Cada sub-modelo é avaliado com seu dataloader e as métricas são salvas em
    `results/{video,pose,emotion}/metrics.json`. Sob `--all`, os três são
    avaliados e um relatório agregado é gravado em `results/reports/`.

    Returns:
        (metrics_by_model, class_names_by_model, roc_pr_data)
    """
    if targets is None:
        targets = ["video", "pose", "emotion"] if args.all else [args.model]
    targets = [t for t in targets if t in ("video", "pose", "emotion")]

    metrics_by_model = {}
    class_names_by_model = {}
    roc_pr_data = {}

    if not targets:
        return metrics_by_model, class_names_by_model, roc_pr_data

    print("\n" + "=" * 60)
    print("Avaliação por sub-modelo (video / pose / emotion)")
    print("=" * 60)

    for model_type in targets:
        try:
            model_path = _resolve_sub_model_path(args, model_type)
            print(f"\n--- Sub-modelo: {model_type} ---")
            print(f"Checkpoint: {model_path}")

            if not Path(model_path).exists():
                raise FileNotFoundError(f"Checkpoint não encontrado: {model_path}")

            model = load_model(
                model_type, model_path, device,
                video_backbone=args.video_backbone,
                video_model_path=args.video_model_path
            )
            test_loader, class_names, num_classes = _get_per_model_test_loader(
                model_type, args.batch_size, video_backbone=args.video_backbone
            )

            calculator = MetricsCalculator(
                model,
                test_loader,
                device=device,
                class_names=class_names,
                num_classes=num_classes
            )
            metrics, y_true, y_pred, y_proba = calculator.evaluate()

            # Salvar em results/{video,pose,emotion}/metrics.json
            out_dir = p.RESULTS_ROOT / model_type
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "metrics.json").write_text(
                json.dumps(metrics, indent=2) + "\n"
            )
            print(
                f"✓ {model_type}: accuracy={metrics['accuracy']:.4f}, "
                f"f1(macro)={metrics['f1_score']['macro']:.4f}"
            )
            print(f"  Salvo em: {out_dir / 'metrics.json'}")

            metrics_by_model[model_type] = metrics
            class_names_by_model[model_type] = class_names
            roc_pr_data[model_type] = _build_roc_pr(
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


def main():
    parser = argparse.ArgumentParser(description="Executar experimentos de avaliação")
    
    # Modelo
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "cnn3d", "multimodal", "video", "pose", "emotion"],
        required=True,
        help="Tipo de modelo (video/pose/emotion = avaliação de sub-modelo)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Caminho para checkpoint do modelo"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Tamanho do batch"
    )
    
    # Caminhos opcionais por sub-modelo (override dos defaults)
    parser.add_argument(
        "--video_backbone",
        type=str,
        choices=["cnn3d", "resnet_lstm"],
        default="cnn3d",
        help="Backbone de vídeo para o sub-modelo video e para extrair features do multimodal (padrão: cnn3d)"
    )
    parser.add_argument(
        "--video_model_path",
        type=str,
        default="models/cnn3d/weights/best_model.pth",
        help="Checkpoint do sub-modelo de vídeo (padrão: models/cnn3d/weights/best_model.pth se --video_backbone cnn3d)"
    )
    parser.add_argument(
        "--pose_model_path",
        type=str,
        default="models/multimodal/weights/best_model.pth",
        help="Checkpoint do sub-modelo de pose (padrão: o checkpoint multimodal de --model_path)"
    )
    parser.add_argument(
        "--emotion_model_path",
        type=str,
        default="models/emotion_cnn/weights/best_model.pth",
        help="Checkpoint do sub-modelo de emoção (padrão: models/emotion_cnn/weights/best_model.pth)"
    )
    
    # Experimentos
    parser.add_argument(
        "--all",
        action="store_true",
        help="Executar todos os experimentos + avaliar os sub-modelos video/pose/emotion"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Calcular métricas básicas"
    )
    parser.add_argument(
        "--robustness",
        action="store_true",
        help="Testar robustez"
    )
    parser.add_argument(
        "--performance",
        action="store_true",
        help="Avaliar performance"
    )
    parser.add_argument(
        "--limitations",
        action="store_true",
        help="Analisar limitações"
    )
    
    # Gráficos (F5, EVAL-T2)
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Gerar gráficos padronizados em results/charts/"
    )
    
    # Estudo de impacto (F5, EVAL-T3)
    parser.add_argument(
        "--impact_study",
        action="store_true",
        help="Executar estudo de impacto das modalidades no modelo fusionado"
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.05,
        help="Desvio padrão do ruído gaussiano no estudo de impacto (0 = sanity check)"
    )
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Nome do experimento (padrão: tipo do modelo)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para inferência"
    )
    
    args = parser.parse_args()
    
    # Definir experimentos a executar
    is_full_model = args.model in ("baseline", "cnn3d", "multimodal")
    if args.all:
        run_metrics = True
        run_robustness = True
        run_performance = True
        run_limitations = True
    else:
        run_metrics = args.metrics
        run_robustness = args.robustness
        run_performance = args.performance
        run_limitations = args.limitations
        
        # Se nenhum especificado, executar métricas básicas
        if not any([run_metrics, run_robustness, run_performance, run_limitations]):
            run_metrics = True
    
    # Nome do experimento
    if args.experiment_name is None:
        args.experiment_name = args.model
    
    print("=" * 60)
    print("Advanced Evaluation Pipeline")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Model Path: {args.model_path}")
    print(f"Output Dir: {p.EXPERIMENTS_ROOT}")
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
    
    # Coletores para gráficos (F5, EVAL-T2)
    metrics_by_model = {}
    class_names_by_model = {}
    roc_pr_data = {}
    results = {}
    model = None
    test_loader = None
    
    if is_full_model:
        # Carregar modelo
        print("Loading model...")
        model = load_model(
            args.model, args.model_path, args.device,
            video_backbone=args.video_backbone,
            video_model_path=args.video_model_path
        )
        print("✓ Model loaded")
        
        # Carregar dataset
        print("Loading dataset...")
        if args.model == "baseline":
            _, _, test_loader = get_dataloaders(
                processed_data_root=str(p.PROCESSED_ROOT),
                batch_size=args.batch_size,
                num_frames=16
            )
        elif args.model == "multimodal":
            _, _, test_loader = get_multimodal_dataloaders(
                video_data_root=str(p.PROCESSED_ROOT),
                pose_data_root=str(p.POSE_ROOT),
                emotion_data_root=str(p.EMOTION_ROOT),
                batch_size=args.batch_size,
                num_frames=16,
                window_size=16,
                video_mode="frames",
                pose_mode="keypoints"
            )
        elif args.model == "cnn3d":
            # Clipes 3D do RWF-2000 em formato frame-last (B, T, C, H, W);
            # a permutação (B, T, C, H, W) -> (B, C, T, H, W) é feita no
            # MetricsCalculator (mesma semântica de train_cnn3d._permute_clips).
            _, _, test_loader = get_rwf2000_3d_dataloaders(
                dataset_root=str(p.RWF2000_ROOT),
                batch_size=args.batch_size,
                num_frames=16,
                clip_size=(112, 112)
            )
        print(f"✓ Dataset loaded ({len(test_loader)} batches)")
        
        # Executar experimentos
        
        # 1. Métricas básicas
        if run_metrics:
            print("\n" + "=" * 60)
            print("1. Calculating Basic Metrics")
            print("=" * 60)
            calculator = MetricsCalculator(
                model, 
                test_loader, 
                device=args.device
            )
            metrics, y_true, y_pred, y_proba = calculator.evaluate()
            results["metrics"] = metrics
            
            # Salvar resultados — CNN3D vai para results/cnn3d/, demais para experiments/
            if args.model == "cnn3d":
                output_root = p.CNN3D_ROOT
                exp_label = "cnn3d"
            else:
                output_root = p.EXPERIMENTS_ROOT / args.experiment_name
                exp_label = "baseline" if args.model == "baseline" else "multimodal"
            output_path = output_root / "metrics"
            output_path.mkdir(parents=True, exist_ok=True)
            calculator.save_results(
                str(output_path),
                exp_label,
                metrics,
                y_true,
                y_pred,
                y_proba
            )
            print("✓ Metrics saved")
            
            # Coletar para gráficos
            metrics_by_model[args.model] = metrics
            class_names_by_model[args.model] = ["Non-Violent", "Violent"]
            roc_pr_data[args.model] = _build_roc_pr(
                args.model, y_true, y_proba, 2, ["Non-Violent", "Violent"]
            )
        
        # 2. Robustez
        if run_robustness:
            print("\n" + "=" * 60)
            print("2. Testing Robustness")
            print("=" * 60)
            robustness_eval = RobustnessEvaluator(model, test_loader, device=args.device)
            robustness_results = robustness_eval.evaluate_all_distortions(
                DEFAULT_DISTORTION_CONFIGS,
                str(p.EXPERIMENTS_ROOT),
                f"{args.experiment_name}/robustness"
            )
            results["robustness"] = robustness_results
            print("✓ Robustness tests completed")
        
        # 3. Performance
        if run_performance:
            print("\n" + "=" * 60)
            print("3. Evaluating Performance")
            print("=" * 60)
            perf_eval = PerformanceEvaluator(model, test_loader, device=args.device)
            perf_results = perf_eval.evaluate_all(
                str(p.EXPERIMENTS_ROOT),
                f"{args.experiment_name}/performance"
            )
            results["performance"] = perf_results
            print("✓ Performance evaluation completed")
        
        # 4. Limitações
        if run_limitations:
            print("\n" + "=" * 60)
            print("4. Analyzing Limitations")
            print("=" * 60)
            limitations_analyzer = LimitationsAnalyzer(model, test_loader, device=args.device)
            limitations_results = limitations_analyzer.generate_error_report(
                str(p.EXPERIMENTS_ROOT),
                f"{args.experiment_name}/limitations"
            )
            results["limitations"] = limitations_results
            print("✓ Limitations analysis completed")
        
        # Salvar resumo geral
        summary_root = p.CNN3D_ROOT if args.model == "cnn3d" else p.EXPERIMENTS_ROOT / args.experiment_name
        summary_path = summary_root / "evaluation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        # Sub-modelo (video/pose/emotion): a avaliação por sub-modelo é a principal
        print(f"Modelo '{args.model}' é um sub-modelo; executando avaliação por sub-modelo.")
    
    # Avaliação por sub-modelo (F5, EVAL-T1): --all ou --model video|pose|emotion
    if args.all or not is_full_model:
        sub_targets = ["video", "pose", "emotion"] if args.all else [args.model]
        sub_metrics, sub_class_names, sub_roc_pr = run_per_model_evaluation(
            args, args.device, targets=sub_targets
        )
        metrics_by_model.update(sub_metrics)
        class_names_by_model.update(sub_class_names)
        roc_pr_data.update(sub_roc_pr)
    
    # Estudo de impacto (F5, EVAL-T3)
    if args.impact_study or (args.all and args.model == "multimodal"):
        if not is_full_model or args.model != "multimodal":
            print("\n⚠️  Estudo de impacto requer o modelo multimodal fusionado "
                  "(--model multimodal). Pulando...")
        else:
            print("\n" + "=" * 60)
            print("5. Impact Study (input noise perturbation)")
            print("=" * 60)
            impact_study = AblationStudy(test_loader, device=args.device)
            impact_results = impact_study.run_impact_study(
                model,
                dataloader=test_loader,
                noise_std=args.noise_std,
                output_dir=str(p.EXPERIMENTS_ROOT / args.experiment_name),
                experiment_name="impact_study"
            )
            results["impact_study"] = impact_results
    
    # Gráficos padronizados (F5, EVAL-T2)
    if args.charts and metrics_by_model:
        written = generate_report(
            metrics_by_model,
            str(p.RESULTS_ROOT / "charts"),
            class_names_by_model,
            roc_pr_data
        )
        print("\n" + "=" * 60)
        print("Gráficos gerados:")
        for w in written:
            print(f"  ✓ {w}")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    if is_full_model:
        print(f"Results saved to: {summary_root}")
    print("=" * 60)


if __name__ == "__main__":
    main()

