import argparse

from src import paths as p
from src.preprocessing._common import _check_dataset_root, _print_header, _process_rwf2000_pose


def cmd_pose(args):
    if not _check_dataset_root():
        return

    if not (0.0 <= args.conf <= 1.0):
        raise SystemExit("--conf deve estar entre 0.0 e 1.0")

    if not (0.0 <= args.iou <= 1.0):
        raise SystemExit("--iou deve estar entre 0.0 e 1.0")

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.preprocessing.pose",
        description="Extrai keypoints de pose de vídeos do dataset RWF-2000.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplo:\n  python -m src.preprocessing.pose --num_frames 16"
    )
    parser.add_argument(
        "--num_frames", type=int, default=None,
        help="Número de frames a processar por vídeo (None = todos os frames)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confiança mínima para detecção (padrão: 0.5, range: 0.0-1.0)"
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU mínima para NMS (padrão: 0.7, range: 0.0-1.0)"
    )
    parser.add_argument(
        "--model_complexity", type=int, choices=[0, 1, 2], default=1,
        help="Complexidade do modelo YOLO26 (padrão: 1). 0=nano, 1=small, 2=medium"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_pose(args)
