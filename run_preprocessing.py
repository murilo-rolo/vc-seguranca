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

    # Extrair keypoints de pose (UCF101 e/ou RWF-2000)
    python run_preprocessing.py pose --dataset both --num_frames 16
    python run_preprocessing.py pose --dataset rwf2000 --num_frames 16

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
        print("  Certifique-se de que os datasets estão em 'dataset/UCF101' e 'dataset/RWF-2000'")
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


def _process_rwf2000_pose(num_frames, min_detection_confidence,
                          min_tracking_confidence, model_complexity):
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
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity
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
        normalize=args.normalize
    )

    print("\nExtração de frames concluída!")


def cmd_pose(args):
    """Extrai keypoints de pose (MediaPipe) dos datasets UCF101 e RWF-2000."""
    if not _check_dataset_root():
        return

    # Validar limites dos parâmetros de confiança
    if not (0.0 <= args.min_detection_confidence <= 1.0):
        raise SystemExit("--min_detection_confidence deve estar entre 0.0 e 1.0")

    if not (0.0 <= args.min_tracking_confidence <= 1.0):
        raise SystemExit("--min_tracking_confidence deve estar entre 0.0 e 1.0")

    # Validar num_frames se fornecido
    if args.num_frames is not None and args.num_frames <= 0:
        raise SystemExit("--num_frames deve ser um número positivo ou None para processar todos os frames")

    from src.pose.extract_pose import process_dataset_for_pose

    p.POSE_ROOT.mkdir(parents=True, exist_ok=True)

    _print_header(
        "Pré-processamento de Pose Estimation",
        {
            "Dataset raiz": str(p.DATASET_ROOT),
            "Saída raiz": str(p.POSE_ROOT),
            "Dataset": args.dataset,
            "Número de frames": args.num_frames if args.num_frames else "Todos",
            "Confiança detecção": f"{args.min_detection_confidence} (range: 0.0-1.0)",
            "Confiança rastreamento": f"{args.min_tracking_confidence} (range: 0.0-1.0)",
            "Complexidade modelo": f"{args.model_complexity} (0=Lite, 1=Full, 2=Heavy)",
        }
    )

    # Avisos sobre valores aumentados
    if args.min_detection_confidence > 0.7:
        print(f"⚠️  AVISO: Confiança de detecção alta ({args.min_detection_confidence}) pode reduzir detecções válidas")
    if args.min_tracking_confidence > 0.7:
        print(f"⚠️  AVISO: Confiança de rastreamento alta ({args.min_tracking_confidence}) pode perder rastreamento em movimentos rápidos")
    if args.model_complexity == 2:
        print("ℹ️  INFO: Modelo Heavy (complexidade 2) será mais lento mas mais preciso")
    if args.num_frames and args.num_frames > 32:
        print(f"ℹ️  INFO: Processando {args.num_frames} frames por vídeo (pode aumentar tempo de processamento)")

    print()

    if args.dataset in ["ucf101", "both"]:
        ucf101_path = p.DATASET_ROOT / "UCF101"
        if ucf101_path.exists():
            print("\n" + "=" * 60)
            print("Processando UCF101...")
            print("=" * 60)
            process_dataset_for_pose(
                dataset_root=str(ucf101_path),
                output_root=str(p.POSE_ROOT),
                dataset_name="ucf101",
                num_frames=args.num_frames,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                model_complexity=args.model_complexity
            )
        else:
            print(f"\nAviso: Dataset UCF101 não encontrado em {ucf101_path}")
            print("  Pulando processamento de UCF101...")

    if args.dataset in ["rwf2000", "both"]:
        _process_rwf2000_pose(
            args.num_frames,
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.model_complexity
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
            "Agregação": args.aggregation,
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
            aggregation=args.aggregation
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
    from src.pose.extract_pose import process_dataset_for_pose

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
        normalize=args.normalize
    )

    # Etapa 3: Extrair pose
    print("\n" + "=" * 50)
    print("[3/4] Extraindo keypoints de pose...")
    print("=" * 50)
    p.POSE_ROOT.mkdir(parents=True, exist_ok=True)
    if args.dataset in ["ucf101", "both"]:
        ucf101_path = p.DATASET_ROOT / "UCF101"
        if ucf101_path.exists():
            print("\nProcessando UCF101...")
            process_dataset_for_pose(
                dataset_root=str(ucf101_path),
                output_root=str(p.POSE_ROOT),
                dataset_name="ucf101",
                num_frames=args.num_frames,
                min_detection_confidence=args.min_detection_confidence,
                min_tracking_confidence=args.min_tracking_confidence,
                model_complexity=args.model_complexity
            )
        else:
            print(f"\nAviso: Dataset UCF101 não encontrado em {ucf101_path}")

    if args.dataset in ["rwf2000", "both"]:
        _process_rwf2000_pose(
            args.num_frames,
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.model_complexity
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
                aggregation=args.aggregation
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
               "  python run_preprocessing.py pose --dataset both --num_frames 16\n"
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
        description="Extrai keypoints de pose de vídeos dos datasets UCF101 e RWF-2000."
    )
    p_pose.add_argument(
        "--dataset", type=str, choices=["ucf101", "rwf2000", "both"], default="both",
        help="Dataset a processar: 'ucf101', 'rwf2000' ou 'both'"
    )
    p_pose.add_argument(
        "--num_frames", type=int, default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    p_pose.add_argument(
        "--min_detection_confidence", type=float, default=0.5,
        help="Confiança mínima para detecção inicial (padrão: 0.5, range: 0.0-1.0). "
             "Valores mais altos (0.7-0.9) reduzem falsos positivos mas podem perder detecções válidas. "
             "Recomendado: 0.5-0.7 para melhor balanceamento."
    )
    p_pose.add_argument(
        "--min_tracking_confidence", type=float, default=0.5,
        help="Confiança mínima para rastreamento (padrão: 0.5, range: 0.0-1.0). "
             "Valores mais altos (0.7-0.9) melhoram estabilidade mas podem perder rastreamento em movimentos rápidos. "
             "Recomendado: 0.5-0.7 para melhor balanceamento."
    )
    p_pose.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo MediaPipe (padrão: 1). "
             "0=Lite (mais rápido, menos preciso), "
             "1=Full (balanceado), "
             "2=Heavy (mais lento, mais preciso). "
             "Recomendado: 2 para máxima precisão em detecção de ameaças."
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
        "--dataset", type=str, choices=["ucf101", "rwf2000", "both"], default="both",
        help="Datasets para extração de pose: 'ucf101', 'rwf2000' ou 'both'"
    )
    p_all.add_argument(
        "--min_detection_confidence", type=float, default=0.5,
        help="Confiança mínima para detecção de pose (padrão: 0.5)"
    )
    p_all.add_argument(
        "--min_tracking_confidence", type=float, default=0.5,
        help="Confiança mínima para rastreamento de pose (padrão: 0.5)"
    )
    p_all.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo MediaPipe (padrão: 1)"
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
