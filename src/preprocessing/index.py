import argparse
import json

from src import paths as p
from src.preprocessing._common import _check_dataset_root
from src.preprocessing import build_index, save_csv


def cmd_index(args):
    if not _check_dataset_root():
        return

    rows = build_index(
        split=args.split,
        include_pose=not args.no_pose,
        include_emotion=not args.no_emotion,
        min_frames=args.min_frames,
    )

    if not rows:
        print("Nenhuma amostra encontrada.")
        return

    output = args.output or str(p.DATASET_ROOT / "pipeline_teste.csv")
    if args.output_format == "csv":
        save_csv(rows, output)
    else:
        from pathlib import Path
        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"✅ JSON salvo em: {output_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src/preprocessing/index.py",
        description="Gera índice unificado CSV que linka vídeos com emoções e pose.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Exemplo:\n  python src/preprocessing/index.py --split all"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test", "all"], default="all",
        help="Split para gerar o índice (padrão: all)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Caminho do CSV de saída (padrão: dataset/pipeline_teste.csv)"
    )
    parser.add_argument(
        "--no-pose", action="store_true",
        help="Não incluir coluna pose_path no CSV"
    )
    parser.add_argument(
        "--no-emotion", action="store_true",
        help="Não incluir coluna emotion_path no CSV"
    )
    parser.add_argument(
        "--min-frames", type=int, default=0,
        help="Mínimo de frames no vídeo para incluir (padrão: 0 = todos)"
    )
    parser.add_argument(
        "--output-format", type=str, choices=["csv", "json"], default="csv",
        help="Formato de saída (padrão: csv)"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_index(args)


if __name__ == "__main__":
    main()
