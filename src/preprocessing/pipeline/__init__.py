import argparse
import os
import torch

from src import paths as p
from src.preprocessing._common import _check_dataset_root, _print_header, EMOTION_MODEL_PATH, _process_rwf2000_pose
from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset
from src.models.emotion_cnn import create_emotion_model
from src.emotion.extract_emotion import process_dataset_for_emotion, extract_emotions_from_affectnet


def cmd_all(args):
    if not _check_dataset_root():
        return

    _print_header(
        "PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO",
        {"Etapa": "4 etapas: organize, frames, pose, emotion"}
    )

    print("\n" + "=" * 50)
    print("[1/4] Organizando vídeos do RWF-2000...")
    print("=" * 50)
    organize_rwf2000_dataset(
        dataset_root=str(p.RWF2000_ROOT),
        output_root=str(p.RAW_DATA_ROOT)
    )

    print("\n" + "=" * 50)
    print("[2/4] Extraindo e processando frames...")
    print("=" * 50)
    preprocess_dataset(
        raw_data_root=str(p.RAW_DATA_ROOT),
        processed_data_root=str(p.PROCESSED_ROOT),
        num_frames=args.num_frames,
        target_size=tuple(args.target_size),
        normalize=args.normalize,
        max_workers=args.workers
    )

    print("\n" + "=" * 50)
    print("[3/4] Extraindo keypoints de pose...")
    print("=" * 50)
    p.POSE_ROOT.mkdir(parents=True, exist_ok=True)
    _process_rwf2000_pose(
        args.num_frames,
        args.model_complexity,
        args.conf,
        args.iou
    )

    print("\n" + "=" * 50)
    print("[4/4] Extraindo vetores de emoção...")
    print("=" * 50)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    p.EMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(EMOTION_MODEL_PATH) if EMOTION_MODEL_PATH.exists() else None
    print("\nCarregando modelo de emoção...")
    try:
        model = create_emotion_model(
            num_emotions=2,
            pretrained=True,
            checkpoint_path=checkpoint_path,
            device=device
        )
        print("Modelo carregado com sucesso!")
        if checkpoint_path:
            print(f"  Checkpoint: {EMOTION_MODEL_PATH}")
        else:
            print(f"  ⚠️  Checkpoint não encontrado em: {EMOTION_MODEL_PATH}")
            print("  Usando pesos ImageNet (modelo não treinado em emoções)")
    except Exception as e:
        print(f"Erro ao carregar modelo de emoção: {e}")
        print("Pulando extração de emoções...")
        model = None

    if model is not None:
        if args.from_affectnet:
            affectnet_path = p.BALANCED_AFFECTNET_ROOT
            if affectnet_path.exists():
                print("\nProcessando Balanced-AffectNet (imagens)...")
                extract_emotions_from_affectnet(
                    affectnet_root=str(affectnet_path),
                    output_root=str(p.EMOTION_ROOT),
                    model=model,
                    batch_size=args.batch_size,
                )
            else:
                print(f"\nAviso: AffectNet não encontrado em {affectnet_path}")
        else:
            rwf2000_path = p.DATASET_ROOT / "RWF-2000"
            if rwf2000_path.exists():
                print("\nProcessando RWF-2000...")
                process_dataset_for_emotion(
                    dataset_root=str(rwf2000_path),
                    output_root=str(p.EMOTION_ROOT),
                    model=model,
                    dataset_name="rwf2000",
                    num_frames=args.num_frames,
                    face_detector_method=args.face_detector,
                    aggregation=args.aggregation,
                    face_aggregation=args.face_aggregation
                )
            else:
                print(f"\nAviso: Dataset RWF-2000 não encontrado em {rwf2000_path}")

    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO!")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.preprocessing.pipeline",
        description="Executa todas as etapas do pré-processamento em sequência.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplo:\n  python -m src.preprocessing.pipeline --num_frames 16"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Número de frames por vídeo (padrão: 16)"
    )
    parser.add_argument(
        "--target_size", type=int, nargs=2, default=[112, 112],
        metavar=("H", "W"),
        help="Tamanho (altura largura) dos frames (padrão: 112 112)"
    )
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True,
        help="Normalizar valores dos pixels para [0, 1] (padrão: --normalize)"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, int(os.cpu_count() / 4)),
        help="Número de workers paralelos para extração de frames (padrão: cpu_count/4)"
    )
    parser.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo YOLO26 (padrão: 1)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confiança mínima para detecção YOLO26 (padrão: 0.5)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU mínima para NMS YOLO26 (padrão: 0.7)"
    )
    parser.add_argument(
        "--face_detector", type=str, choices=["mtcnn", "retinaface", "haar"], default="mtcnn",
        help="Método de detecção de faces (padrão: 'mtcnn')"
    )
    parser.add_argument(
        "--aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação temporal (padrão: 'mean')"
    )
    parser.add_argument(
        "--face_aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação das faces por frame (padrão: 'mean')"
    )
    parser.add_argument(
        "--from-affectnet", action="store_true",
        help="Usar imagens do balanced-affectnet para emoções (em vez de vídeos RWF-2000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Tamanho do batch para extração de emoções (padrão: 32)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device para processamento (padrão: 'cuda' se disponível, senão 'cpu')"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_all(args)
