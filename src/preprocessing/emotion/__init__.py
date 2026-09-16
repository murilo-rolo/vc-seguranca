import argparse
import torch

from src import paths as p
from src.preprocessing._common import _check_dataset_root, _print_header, EMOTION_MODEL_PATH
from src.models.emotion_cnn import create_emotion_model
from src.emotion.extract_emotion import process_dataset_for_emotion


def cmd_emotion(args):
    if not _check_dataset_root():
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    p.EMOTION_ROOT.mkdir(parents=True, exist_ok=True)

    _print_header(
        "Pré-processamento de Emotion Recognition",
        {
            "Dataset raiz": str(p.DATASET_ROOT),
            "Saída raiz": str(p.EMOTION_ROOT),
            "Número de frames": args.num_frames if args.num_frames else "Todos",
            "Detector de faces": args.face_detector,
            "Agregação temporal": args.aggregation,
            "Agregação de faces": args.face_aggregation,
            "Device": device,
            "Modelo": str(EMOTION_MODEL_PATH),
        }
    )

    checkpoint_path = str(EMOTION_MODEL_PATH) if EMOTION_MODEL_PATH.exists() else None
    print("Carregando modelo de emoção...")
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
            print("  ⚠️  Para melhor performance, treine o modelo no AffectNet primeiro!")
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return

    rwf2000_path = p.DATASET_ROOT / "RWF-2000"
    if rwf2000_path.exists():
        print("\n" + "=" * 60)
        print("Processando RWF-2000...")
        print("=" * 60)
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
        print("  Pulando processamento...")

    print("\n" + "=" * 60)
    print("Pré-processamento de emoção concluído!")
    print("=" * 60)
    print(f"\nVetores de emoção salvos em: {p.EMOTION_ROOT}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.preprocessing.emotion",
        description="Extrai vetores de emoção de vídeos do dataset RWF-2000.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplo:\n  python -m src.preprocessing.emotion --num_frames 16"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
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
        "--device", type=str, default=None,
        help="Device para processamento (padrão: 'cuda' se disponível, senão 'cpu')"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_emotion(args)
