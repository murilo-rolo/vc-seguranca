"""
Script para gerar um índice unificado (CSV) que liga cada vídeo
violent/non_violent com suas emoções correspondentes, funcionando
como um único dataset.

Pode gerar o CSV automaticamente a partir das pastas do balanced-affectnet
e do RWF-2000 processado, com pareamento aleatório controlado por seed.

Uso:
    python build_dataset_index.py                          # Gera CSV + index
    python build_dataset_index.py --generate-csv --seed 42  # Gera CSV aleatório
    python build_dataset_index.py --split train             # Apenas treino
    python build_dataset_index.py --generate-csv --seed 42 --split all
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Set

from src import paths as p

LABEL_MAP = {
    "violent": 1,
    "non_violent": 0,
}


def find_class_videos(
    base_dir: Path,
    split: str,
    class_name: str,
) -> List[Path]:
    class_dir = base_dir / split / class_name
    if not class_dir.exists():
        return []
    pt_files = []
    for subdir in class_dir.iterdir():
        if subdir.is_dir():
            pt = subdir / "frame_sequence.pt"
            if pt.exists():
                pt_files.append(pt)
    return sorted(pt_files)


def get_split_dirs(split: str) -> List[str]:
    if split == "all":
        return ["train", "val"]
    if split == "test":
        return ["val"]
    return [split]


def generate_csv(
    split: str = "all",
    seed: int = 42,
    output_path: str = None,
) -> List[Dict]:
    """
    Gera pipeline_teste.csv pareando aleatoriamente faces do
    balanced-affectnet com diretórios de vídeos do RWF-2000 processado.

    Garante que faces violent sejam pareadas com vídeos violent e
    faces non_violent com vídeos non_violent.

    Args:
        split: "train", "val", "test", ou "all"
        seed: Seed para reprodutibilidade do pareamento
        output_path: Caminho do CSV de saída

    Returns:
        Lista de dicionários com os dados do CSV.
    """
    if output_path is None:
        output_path = str(p.PIPELINE_CSV_PATH)

    random.seed(seed)

    affectnet_root = p.BALANCED_AFFECTNET_ROOT
    processed_root = p.PROCESSED_ROOT

    splits = get_split_dirs(split)
    rows: List[Dict] = []

    for current_split in splits:
        for class_name in ["violent", "non_violent"]:
            # Listar imagens do balanced-affectnet
            class_affectnet_dir = affectnet_root / current_split / class_name
            if not class_affectnet_dir.exists():
                print(f"⚠️  Diretório não encontrado: {class_affectnet_dir}")
                continue

            image_files = sorted(class_affectnet_dir.glob("*.png"))
            if not image_files:
                image_files = sorted(class_affectnet_dir.glob("*.jpg")) + sorted(class_affectnet_dir.glob("*.jpeg"))

            if not image_files:
                print(f"⚠️  Nenhuma imagem encontrada em {class_affectnet_dir}")
                continue

            # Listar diretórios de vídeo do RWF-2000 processado
            class_video_dir = processed_root / current_split / class_name
            if not class_video_dir.exists():
                print(f"⚠️  Diretório de vídeos não encontrado: {class_video_dir}")
                continue

            video_dirs = sorted([d for d in class_video_dir.iterdir() if d.is_dir()])
            if not video_dirs:
                print(f"⚠️  Nenhum diretório de vídeo em {class_video_dir}")
                continue

            # Embaralhar imagens e vídeos para pareamento aleatório
            shuffled_images = list(image_files)
            shuffled_videos = list(video_dirs)
            random.shuffle(shuffled_images)
            random.shuffle(shuffled_videos)

            # Parear: cada imagem com um vídeo (ciclo se mais imagens que vídeos)
            num_pairs = min(len(shuffled_images), len(shuffled_videos))
            if len(shuffled_images) > len(shuffled_videos):
                # Repetir vídeos se necessário
                shuffled_videos = (shuffled_videos * ((len(shuffled_images) // len(shuffled_videos)) + 1))[:len(shuffled_images)]
                random.shuffle(shuffled_videos)

            for i in range(num_pairs):
                img_path = shuffled_images[i]
                video_dir = shuffled_videos[i]
                video_id = video_dir.name

                # Verificar se o vídeo tem frame_sequence.pt
                frame_file = video_dir / "frame_sequence.pt"
                if not frame_file.exists():
                    continue

                rows.append({
                    "video": str(video_dir.relative_to(p.PROJECT_ROOT)),
                    "emotion": class_name,
                    "split": current_split,
                    "class": class_name,
                    "image_path": str(img_path.relative_to(p.PROJECT_ROOT)),
                    "video_id": video_id,
                })

    # Embaralhar as linhas finais para misturar splits e classes
    random.shuffle(rows)

    if rows:
        violent_count = sum(1 for r in rows if r["emotion"] == "violent")
        non_violent_count = sum(1 for r in rows if r["emotion"] == "non_violent")
        print(f"\n✅ CSV gerado com sucesso!")
        print(f"   Total de amostras: {len(rows)}")
        print(f"   Violent: {violent_count}, Non-violent: {non_violent_count}")
        splits_found = sorted(set(r["split"] for r in rows))
        print(f"   Splits: {', '.join(splits_found)}")
        print(f"   Seed: {seed}")
    else:
        print("⚠️  Nenhuma amostra encontrada para gerar o CSV.")
        print(f"   Verifique: {affectnet_root} e {processed_root}")

    # Salvar CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "emotion", "split", "class", "image_path", "video_id"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ CSV salvo em: {output_file}")
    print(f"   Total de linhas: {len(rows)}")

    return rows


def build_index(
    split: str = "all",
    include_pose: bool = True,
    include_emotion: bool = True,
    min_frames: int = 0,
    seed: int = 42,
) -> List[Dict]:
    processed_root = p.PROCESSED_ROOT
    emotion_root = p.EMOTION_ROOT
    pose_root = p.POSE_ROOT

    rows: List[Dict] = []
    splits = get_split_dirs(split)

    emotion_rng = random.Random(seed)

    emotion_pools: Dict[str, List[Path]] = {}
    used_emotions: Dict[str, Set[str]] = {}
    if include_emotion:
        for s in splits:
            for class_name in LABEL_MAP:
                pool_dir = emotion_root / "balanced-affectnet" / s / class_name
                if pool_dir.exists():
                    emotion_pools[f"{s}/{class_name}"] = sorted(pool_dir.glob("*.npy"))
                else:
                    emotion_pools[f"{s}/{class_name}"] = []

    for current_split in splits:
        for class_name, label in LABEL_MAP.items():
            video_files = find_class_videos(processed_root, current_split, class_name)

            for video_path in video_files:
                video_id = video_path.parent.name
                class_dir_name = class_name

                emotion_file = ""
                if include_emotion:
                    pool_key = f"{current_split}/{class_name}"
                    pool = emotion_pools.get(pool_key, [])
                    used = used_emotions.get(pool_key, set())
                    available = [p for p in pool if str(p) not in used]

                    if available:
                        chosen = emotion_rng.choice(available)
                        used_emotions.setdefault(pool_key, set()).add(str(chosen))
                        emotion_file = str(chosen.relative_to(p.PROJECT_ROOT))
                    elif pool:
                        chosen = emotion_rng.choice(pool)
                        emotion_file = str(chosen.relative_to(p.PROJECT_ROOT))

                pose_file = None
                if include_pose:
                    pose_path = (
                        pose_root / "rwf2000" / current_split / class_dir_name / f"{video_id}.npy"
                    )
                    if pose_path.exists():
                        pose_file = str(pose_path)
                    elif current_split == "val" and split == "test":
                        pose_path = (
                            pose_root / "rwf2000" / "val" / class_dir_name / f"{video_id}.npy"
                        )
                        if pose_path.exists():
                            pose_file = str(pose_path)

                if min_frames > 0:
                    try:
                        import torch
                        frames = torch.load(video_path, map_location="cpu")
                        if frames.shape[0] < min_frames:
                            continue
                    except Exception:
                        continue

                rows.append({
                    "video_path": str(video_path.relative_to(p.PROJECT_ROOT)),
                    "emotion_path": emotion_file,
                    "pose_path": str(pose_path.relative_to(p.PROJECT_ROOT)) if pose_file else "",
                    "label": label,
                    "split": current_split,
                    "class": class_name,
                    "video_id": video_id,
                })

    if split == "test":
        val_rows = [r for r in rows if r["split"] == "val"]
        val_rows_sorted = sorted(val_rows, key=lambda x: x["video_id"])
        val_test_ratio = 0.5
        val_end = int(len(val_rows_sorted) * (1 - val_test_ratio))
        test_rows = [{**r, "split": "test"} for r in val_rows_sorted[val_end:]]
        rows = test_rows

    if rows:
        violent_count = sum(1 for r in rows if r["label"] == 1)
        non_violent_count = sum(1 for r in rows if r["label"] == 0)
        print(f"Índice gerado: {len(rows)} amostras")
        print(f"  Violent: {violent_count}, Non-violent: {non_violent_count}")
        splits_found = sorted(set(r["split"] for r in rows))
        print(f"  Splits: {', '.join(splits_found)}")
    else:
        print("Nenhuma amostra encontrada. Verifique os caminhos dos dados.")
        print(f"  Processado: {processed_root}")
        print(f"  Emoção: {emotion_root}")
        print(f"  Pose: {pose_root}")

    return rows


def save_csv(rows: List[Dict], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video_path", "emotion_path", "pose_path", "label", "split", "class", "video_id"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ CSV salvo em: {output_file}")
    print(f"   Total de linhas: {len(rows)}")


def save_json(rows: List[Dict], output_path: str) -> None:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n✅ JSON salvo em: {output_file}")
    print(f"   Total de linhas: {len(rows)}")


def build_cross_label_index(
    scenario: str = "violent_face_non_violent_video",
    split: str = "all",
    include_pose: bool = True,
    min_frames: int = 0,
    seed: int = 42,
) -> List[Dict]:
    """
    Gera um índice CSV com pareamento cross-label entre emoção facial e vídeo.

    Em vez de parear cada vídeo com uma emoção da mesma classe, troca:
      - violent_face_non_violent_video: vídeos violent recebem faces non_violent
      - non_violent_face_violent_video: vídeos non_violent recebem faces violent

    O label do vídeo (coluna 'label') permanece o original (dado do diretório).
    A coluna 'class' também permanece a classe real do vídeo.
    Apenas 'emotion_path' é trocado para a classe oposta.

    Args:
        scenario: "violent_face_non_violent_video" ou "non_violent_face_violent_video"
        split: "train", "val", "test", ou "all"
        include_pose: Incluir coluna pose_path
        min_frames: Mínimo de frames no vídeo para incluir
        seed: Seed para reprodutibilidade

    Returns:
        Lista de dicionários com os dados do CSV cross-label.
    """
    valid_scenarios = ("violent_face_non_violent_video", "non_violent_face_violent_video")
    if scenario not in valid_scenarios:
        raise ValueError(f"scenario deve ser um de {valid_scenarios}, recebeu: {scenario}")

    processed_root = p.PROCESSED_ROOT
    emotion_root = p.EMOTION_ROOT
    pose_root = p.POSE_ROOT

    rows: List[Dict] = []
    splits = get_split_dirs(split)

    emotion_rng = random.Random(seed)

    # Mapeamento de cenário: qual classe de emoção usar para cada classe de vídeo
    if scenario == "violent_face_non_violent_video":
        # Vídeo violent -> face non_violent; Vídeo non_violent -> face violent
        emotion_swap_map = {
            "violent": "non_violent",
            "non_violent": "violent",
        }
    else:
        # non_violent_face_violent_video: vídeo violent -> face violent (mantém);
        #                                  vídeo non_violent -> face violent
        # Neste cenário, ambos recebem face violent
        emotion_swap_map = {
            "violent": "violent",
            "non_violent": "violent",
        }

    # Construir pools de emoção para a classe ORIGEM de cada troca
    emotion_pools: Dict[str, List[Path]] = {}
    used_emotions: Dict[str, Set[str]] = {}
    for s in splits:
        for class_name in LABEL_MAP:
            pool_dir = emotion_root / "balanced-affectnet" / s / class_name
            if pool_dir.exists():
                emotion_pools[f"{s}/{class_name}"] = sorted(pool_dir.glob("*.npy"))
            else:
                emotion_pools[f"{s}/{class_name}"] = []

    for current_split in splits:
        for class_name, label in LABEL_MAP.items():
            video_files = find_class_videos(processed_root, current_split, class_name)

            for video_path in video_files:
                video_id = video_path.parent.name

                # Determinar de qual pool de emoção puxar (classe oposta/trocada)
                emotion_source_class = emotion_swap_map[class_name]
                pool_key = f"{current_split}/{emotion_source_class}"
                pool = emotion_pools.get(pool_key, [])
                used = used_emotions.get(pool_key, set())
                available = [ep for ep in pool if str(ep) not in used]

                emotion_file = ""
                if available:
                    chosen = emotion_rng.choice(available)
                    used_emotions.setdefault(pool_key, set()).add(str(chosen))
                    emotion_file = str(chosen.relative_to(p.PROJECT_ROOT))
                elif pool:
                    chosen = emotion_rng.choice(pool)
                    emotion_file = str(chosen.relative_to(p.PROJECT_ROOT))

                pose_file = None
                if include_pose:
                    pose_path = (
                        pose_root / "rwf2000" / current_split / class_name / f"{video_id}.npy"
                    )
                    if pose_path.exists():
                        pose_file = str(pose_path)
                    elif current_split == "val" and split == "test":
                        pose_path = (
                            pose_root / "rwf2000" / "val" / class_name / f"{video_id}.npy"
                        )
                        if pose_path.exists():
                            pose_file = str(pose_path)

                if min_frames > 0:
                    try:
                        import torch
                        frames = torch.load(video_path, map_location="cpu")
                        if frames.shape[0] < min_frames:
                            continue
                    except Exception:
                        continue

                rows.append({
                    "video_path": str(video_path.relative_to(p.PROJECT_ROOT)),
                    "emotion_path": emotion_file,
                    "pose_path": str(pose_path.relative_to(p.PROJECT_ROOT)) if pose_file else "",
                    "label": label,
                    "split": current_split,
                    "class": class_name,
                    "video_id": video_id,
                })

    if split == "test":
        val_rows = [r for r in rows if r["split"] == "val"]
        val_rows_sorted = sorted(val_rows, key=lambda x: x["video_id"])
        val_test_ratio = 0.5
        val_end = int(len(val_rows_sorted) * (1 - val_test_ratio))
        test_rows = [{**r, "split": "test"} for r in val_rows_sorted[val_end:]]
        rows = test_rows

    if rows:
        violent_count = sum(1 for r in rows if r["label"] == 1)
        non_violent_count = sum(1 for r in rows if r["label"] == 0)
        print(f"Cross-label index ({scenario}): {len(rows)} amostras")
        print(f"  Violent: {violent_count}, Non-violent: {non_violent_count}")
        splits_found = sorted(set(r["split"] for r in rows))
        print(f"  Splits: {', '.join(splits_found)}")
    else:
        print("Nenhuma amostra encontrada para gerar o índice cross-label.")

    return rows


def load_index(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def main():
    parser = argparse.ArgumentParser(
        prog="build_dataset_index.py",
        description="Gera índice unificado CSV ou gera CSV aleatório pareando faces com vídeos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:

  # Gerar CSV aleatório com seed (pareamento face↔vídeo)
  python build_dataset_index.py --generate-csv --seed 42

  # Gerar apenas para treino
  python build_dataset_index.py --generate-csv --seed 42 --split train

  # Gerar índice a partir da estrutura de diretórios
  python build_dataset_index.py

  # Gerar índice e salvar como JSON
  python build_dataset_index.py --output-format json

  # Dry run do CSV gerador
  python build_dataset_index.py --generate-csv --seed 42 --dry-run
        """
    )

    parser.add_argument(
        "--generate-csv", action="store_true",
        help="Gerar pipeline_teste.csv a partir do balanced-affectnet (pareamento aleatório)"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "val", "test", "all"],
        default="all", help="Split para gerar o índice (padrão: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para reprodutibilidade do pareamento aleatório (padrão: 42)"
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
        help="Formato de saída para o índice (padrão: csv)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Mostra quantas amostras seriam geradas sem salvar o arquivo"
    )

    args = parser.parse_args()

    if args.generate_csv:
        rows = generate_csv(
            split=args.split,
            seed=args.seed,
            output_path=args.output or str(p.PIPELINE_CSV_PATH),
        )
        if args.dry_run:
            print(f"\n[DRY RUN] {len(rows)} amostras seriam geradas no CSV.")
        return

    rows = build_index(
        split=args.split,
        include_pose=not args.no_pose,
        include_emotion=not args.no_emotion,
        min_frames=args.min_frames,
    )

    if args.dry_run:
        print(f"\n[DRY RUN] {len(rows)} amostras seriam geradas.")
        return

    if not rows:
        print("Nenhuma amostra encontrada para gerar o índice.")
        sys.exit(1)

    output = args.output or str(p.DATASET_ROOT / "pipeline_teste.csv")
    if args.output_format == "csv":
        save_csv(rows, output)
    else:
        save_json(rows, output)


if __name__ == "__main__":
    main()
