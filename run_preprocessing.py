"""
Script único para pré-processamento dos datasets.

Este script unifica todo o pipeline de pré-processamento em um único
ponto de entrada com subcomandos:

    organize  - Organiza os vídeos do RWF-2000 em data/raw
    frames    - Extrai e processa frames RGB (tensores .pt) em data/processed
    pose      - Extrai keypoints de pose (MediaPipe) em data/pose
    emotion   - Extrai vetores de emoção (EmotionNet) em data/emotion
    all       - Executa todas as etapas em sequência

Uso:
    # Organizar vídeos do RWF-2000
    python run_preprocessing.py organize

    # Extrair frames RGB
    python run_preprocessing.py frames --num_frames 16

    # Extrair keypoints de pose (RWF-2000)
    python run_preprocessing.py pose --num_frames 16

    # Extrair vetores de emoção (RWF-2000)
    python run_preprocessing.py emotion

    # Executar todo o pipeline
    python run_preprocessing.py all --num_frames 16
"""

import argparse
import os

from src import paths as p


EMOTION_MODEL_PATH = p.EMOTION_CNN_WEIGHTS / "best_model.pth"


def _check_dataset_root() -> bool:
    """Valida a existência do diretório raiz de datasets."""
    if not p.DATASET_ROOT.exists():
        print(f"Erro: Diretório de datasets não encontrado: {p.DATASET_ROOT}")
        print("  Certifique-se de que os datasets estão em 'dataset/RWF-2000'")
        return False
    return True


def _print_header(title: str, fields: dict):
    print("=" * 60)
    print(title)
    print("=" * 60)
    for key, value in fields.items():
        print(f"{key}: {value}")
    print("=" * 60)
    print()


def _process_rwf2000_pose(num_frames, model_complexity, conf=0.5, iou=0.7):
    from src.pose.extract_pose import process_dataset_for_pose

    rwf2000_path = p.DATASET_ROOT / "RWF-2000"
    if rwf2000_path.exists():
        print("\n" + "=" * 60)
        print("Processando RWF-2000...")
        print("=" * 60)
        process_dataset_for_pose(
            dataset_root=str(rwf2000_path),
            output_root=str(p.POSE_ROOT),
            dataset_name="rwf2000",
            num_frames=num_frames,
            model_complexity=model_complexity,
            conf=conf,
            iou=iou,
        )
    else:
        print(f"\nAviso: Dataset RWF-2000 não encontrado em {rwf2000_path}")
        print("  Pulando processamento de RWF-2000...")


def cmd_organize(args):
    """Organiza os vídeos do RWF-2000 em data/raw."""
    if not _check_dataset_root():
        return

    from src.preprocessing import organize_rwf2000_dataset

    print("=" * 50)
    print("PRÉ-PROCESSAMENTO DO DATASET RWF-2000")
    print("=" * 50)

    print("\nOrganizando vídeos...")
    num_violent, num_non_violent = organize_rwf2000_dataset(
        dataset_root=str(p.RWF2000_ROOT),
        output_root=str(p.RAW_DATA_ROOT)
    )
    print("\nOrganização concluída!")


def cmd_frames(args):
    """Extrai frames RGB dos vídeos organizados e salva como tensores .pt."""
    if not _check_dataset_root():
        return

    from src.preprocessing import preprocess_dataset

    target_size = tuple(args.target_size)

    _print_header(
        "Extração e processamento de frames",
        {
            "Entrada": str(p.RAW_DATA_ROOT),
            "Saída": str(p.PROCESSED_ROOT),
            "Número de frames": args.num_frames,
            "Tamanho alvo": f"{target_size[0]}x{target_size[1]}",
            "Normalizar": args.normalize,
            "Workers": args.workers,
        }
    )

    preprocess_dataset(
        raw_data_root=str(p.RAW_DATA_ROOT),
        processed_data_root=str(p.PROCESSED_ROOT),
        num_frames=args.num_frames,
        target_size=target_size,
        normalize=args.normalize,
        max_workers=args.workers
    )

    print("\nExtração de frames concluída!")


def cmd_pose(args):
    """Extrai keypoints de pose (YOLO26) do dataset RWF-2000."""
    if not _check_dataset_root():
        return

    # Validar limites dos parâmetros de confiança
    if not (0.0 <= args.conf <= 1.0):
        raise SystemExit("--conf deve estar entre 0.0 e 1.0")

    if not (0.0 <= args.iou <= 1.0):
        raise SystemExit("--iou deve estar entre 0.0 e 1.0")

    # Validar num_frames se fornecido
    if args.num_frames is not None and args.num_frames <= 0:
        raise SystemExit("--num_frames deve ser um número positivo ou None para processar todos os frames")

    p.POSE_ROOT.mkdir(parents=True, exist_ok=True)

    _print_header(
        "Pré-processamento de Pose Estimation",
        {
            "Dataset raiz": str(p.DATASET_ROOT),
            "Saída raiz": str(p.POSE_ROOT),
            "Dataset": "RWF-2000",
            "Número de frames": args.num_frames if args.num_frames else "Todos",
            "Confiança detecção": f"{args.conf} (range: 0.0-1.0)",
            "IoU NMS": f"{args.iou} (range: 0.0-1.0)",
            "Complexidade modelo": f"{args.model_complexity} (0=nano, 1=small, 2=medium)",
        }
    )

    # Avisos sobre valores aumentados
    if args.conf > 0.7:
        print(f"⚠️  AVISO: Confiança de detecção alta ({args.conf}) pode reduzir detecções válidas")
    if args.iou < 0.3:
        print(f"⚠️  AVISO: IoU baixo ({args.iou}) pode detectar muitas caixas duplicadas")
    if args.model_complexity == 2:
        print("ℹ️  INFO: Modelo medium (complexidade 2) será mais lento mas mais preciso")
    if args.num_frames and args.num_frames > 32:
        print(f"ℹ️  INFO: Processando {args.num_frames} frames por vídeo (pode aumentar tempo de processamento)")

    print()

    _process_rwf2000_pose(
        args.num_frames,
        args.model_complexity,
        args.conf,
        args.iou
    )

    print("\n" + "=" * 60)
    print("Pré-processamento de pose concluído!")
    print("=" * 60)
    print(f"\nKeypoints salvos em: {p.POSE_ROOT}")


def cmd_emotion(args):
    """Extrai vetores de emoção (EmotionNet) dos vídeos do RWF-2000."""
    if not _check_dataset_root():
        return

    import torch
    from src.models.emotion_cnn import create_emotion_model
    from src.emotion.extract_emotion import process_dataset_for_emotion

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

    # Carregar modelo
    checkpoint_path = str(EMOTION_MODEL_PATH) if EMOTION_MODEL_PATH.exists() else None
    print("Carregando modelo de emoção...")
    try:
        model = create_emotion_model(
            num_emotions=8,
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


def cmd_all(args):
    """Executa todas as etapas do pré-processamento em sequência."""
    if not _check_dataset_root():
        return

    from src.preprocessing import organize_rwf2000_dataset, preprocess_dataset

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETO DE PRÉ-PROCESSAMENTO")
    print("=" * 60)

    # Etapa 1: Organizar vídeos
    print("\n" + "=" * 50)
    print("[1/4] Organizando vídeos do RWF-2000...")
    print("=" * 50)
    organize_rwf2000_dataset(
        dataset_root=str(p.RWF2000_ROOT),
        output_root=str(p.RAW_DATA_ROOT)
    )

    # Etapa 2: Extrair frames
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

    # Etapa 3: Extrair pose
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

    # Etapa 4: Extrair emoções
    print("\n" + "=" * 50)
    print("[4/4] Extraindo vetores de emoção...")
    print("=" * 50)
    import torch
    from src.models.emotion_cnn import create_emotion_model
    from src.emotion.extract_emotion import process_dataset_for_emotion

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    p.EMOTION_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(EMOTION_MODEL_PATH) if EMOTION_MODEL_PATH.exists() else None
    print("\nCarregando modelo de emoção...")
    try:
        model = create_emotion_model(
            num_emotions=8,
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
        prog="run_preprocessing.py",
        description="Script único de pré-processamento dos datasets (frames, pose e emoção).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplos:\n"
               "  python run_preprocessing.py organize\n"
               "  python run_preprocessing.py frames --num_frames 16\n"
               "  python run_preprocessing.py pose --num_frames 16\n"
               "  python run_preprocessing.py emotion\n"
               "  python run_preprocessing.py all --num_frames 16\n"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Etapa de pré-processamento a executar"
    )

    # ── organize ──────────────────────────────────────────────────────────
    subparsers.add_parser(
        "organize",
        help="Organiza os vídeos do RWF-2000 em data/raw",
        description="Organiza os vídeos do dataset RWF-2000 nas pastas violent/non_violent."
    )

    # ── frames ────────────────────────────────────────────────────────────
    p_frames = subparsers.add_parser(
        "frames",
        help="Extrai frames RGB dos vídeos organizados",
        description="Extrai N frames por vídeo, redimensiona e normaliza para data/processed."
    )
    p_frames.add_argument(
        "--num_frames", type=int, default=16,
        help="Número de frames a extrair por vídeo (padrão: 16)"
    )
    p_frames.add_argument(
        "--target_size", type=int, nargs=2, default=[112, 112],
        metavar=("H", "W"),
        help="Tamanho (altura largura) dos frames (padrão: 112 112)"
    )
    p_frames.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True,
        help="Normalizar valores dos pixels para [0, 1] (padrão: --normalize)"
    )
    p_frames.add_argument(
        "--workers", type=int, default=max(1, int(os.cpu_count() / 4)),
        help="Número de workers paralelos para extração (padrão: cpu_count/4)"
    )
    p_frames.set_defaults(func=cmd_frames)

    # ── pose ──────────────────────────────────────────────────────────────
    p_pose = subparsers.add_parser(
        "pose",
        help="Extrai keypoints de pose (MediaPipe)",
        description="Extrai keypoints de pose de vídeos do dataset RWF-2000."
    )
    p_pose.add_argument(
        "--num_frames", type=int, default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    p_pose.add_argument(
        "--conf", type=float, default=0.5,
        help="Confiança mínima para detecção (padrão: 0.5, range: 0.0-1.0). "
             "Valores mais altos (0.7-0.9) reduzem falsos positivos mas podem perder detecções válidas. "
             "Recomendado: 0.5-0.7 para melhor balanceamento."
    )
    p_pose.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU mínima para NMS (padrão: 0.7, range: 0.0-1.0). "
             "Valores mais baixos detectam mais caixas, valores mais altos reduzem duplicatas."
    )
    p_pose.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo YOLO26 (padrão: 1). "
             "0=nano (mais rápido, menos preciso), "
             "1=small (balanceado), "
             "2=medium (mais lento, mais preciso)."
    )
    p_pose.set_defaults(func=cmd_pose)

    # ── emotion ───────────────────────────────────────────────────────────
    p_emotion = subparsers.add_parser(
        "emotion",
        help="Extrai vetores de emoção (EmotionNet)",
        description="Extrai vetores de emoção de vídeos do dataset RWF-2000."
    )
    p_emotion.add_argument(
        "--num_frames", type=int, default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    p_emotion.add_argument(
        "--face_detector", type=str, choices=["mtcnn", "retinaface", "haar"], default="mtcnn",
        help="Método de detecção de faces (padrão: 'mtcnn')"
    )
    p_emotion.add_argument(
        "--aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação temporal (padrão: 'mean')"
    )
    p_emotion.add_argument(
        "--face_aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação das faces por frame (padrão: 'mean'; "
             "'max' = máximo elemento a elemento sobre todas as faces; "
             "frames sem face usam o embedding neutro)"
    )
    p_emotion.add_argument(
        "--device", type=str, default=None,
        help="Device para processamento (padrão: 'cuda' se disponível, senão 'cpu')"
    )
    p_emotion.set_defaults(func=cmd_emotion)

    # ── all ───────────────────────────────────────────────────────────────
    p_all = subparsers.add_parser(
        "all",
        help="Executa todo o pipeline de pré-processamento",
        description="Executa todas as etapas em sequência: organize, frames, pose e emotion."
    )
    p_all.add_argument(
        "--num_frames", type=int, default=16,
        help="Número de frames por vídeo (padrão: 16)"
    )
    p_all.add_argument(
        "--target_size", type=int, nargs=2, default=[112, 112],
        metavar=("H", "W"),
        help="Tamanho (altura largura) dos frames (padrão: 112 112)"
    )
    p_all.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True,
        help="Normalizar valores dos pixels para [0, 1] (padrão: --normalize)"
    )
    p_all.add_argument(
        "--workers", type=int, default=max(1, int(os.cpu_count() / 4)),
        help="Número de workers paralelos para extração de frames (padrão: cpu_count/4)"
    )
    p_all.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo YOLO26 (padrão: 1)"
    )
    p_all.add_argument(
        "--conf", type=float, default=0.5,
        help="Confiança mínima para detecção YOLO26 (padrão: 0.5)"
    )
    p_all.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU mínima para NMS YOLO26 (padrão: 0.7)"
    )
    p_all.add_argument(
        "--face_detector", type=str, choices=["mtcnn", "retinaface", "haar"], default="mtcnn",
        help="Método de detecção de faces (padrão: 'mtcnn')"
    )
    p_all.add_argument(
        "--aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação temporal (padrão: 'mean')"
    )
    p_all.add_argument(
        "--face_aggregation", type=str, choices=["mean", "max"], default="mean",
        help="Método de agregação das faces por frame (padrão: 'mean'; "
             "'max' = máximo elemento a elemento sobre todas as faces)"
    )
    p_all.add_argument(
        "--device", type=str, default=None,
        help="Device para processamento (padrão: 'cuda' se disponível, senão 'cpu')"
    )
    p_all.set_defaults(func=cmd_all)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
