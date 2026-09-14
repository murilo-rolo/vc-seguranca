import argparse

from src import paths as p
from src.preprocessing._common import _check_dataset_root
from src.preprocessing import organize_rwf2000_dataset


def cmd_organize(args):
    if not _check_dataset_root():
        return

    print("=" * 50)
    print("PRÉ-PROCESSAMENTO DO DATASET RWF-2000")
    print("=" * 50)

    print("\nOrganizando vídeos...")
    num_violent, num_non_violent = organize_rwf2000_dataset(
        dataset_root=str(p.RWF2000_ROOT),
        output_root=str(p.RAW_DATA_ROOT)
    )
    print("\nOrganização concluída!")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src/preprocessing/organize.py",
        description="Organiza os vídeos do dataset RWF-2000 nas pastas violent/non_violent.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    cmd_organize(args)


if __name__ == "__main__":
    main()
