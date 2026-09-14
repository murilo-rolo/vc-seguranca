import argparse
import os

from src import paths as p
from src.preprocessing._common import _check_dataset_root, _print_header
from src.preprocessing import preprocess_dataset


def cmd_frames(args):
    if not _check_dataset_root():
        return

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src/preprocessing/frames.py",
        description="Extrai N frames por vídeo, redimensiona e normaliza para data/processed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplo:\n  python src/preprocessing/frames.py --num_frames 16"
    )
    parser.add_argument(
        "--num_frames", type=int, default=16,
        help="Número de frames a extrair por vídeo (padrão: 16)"
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
        help="Número de workers paralelos para extração (padrão: cpu_count/4)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_frames(args)


if __name__ == "__main__":
    main()
