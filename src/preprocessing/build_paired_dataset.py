"""
Gera o dataset pareado RWF-2000 (vídeo) + AffectNet (face) em dataset/paired/.

Cada amostra é um par (vídeo, face) materializado como diretório com symlinks:

    dataset/paired/<split>/<cell>/<pair_id>/
        frames.pt    -> data/processed/<orig_split>/<v_label>/<video_id>/frame_sequence.pt
        features.pt  -> .../features.pt            (opcional, se existir)
        pose.npy     -> data/pose/rwf2000/...      (opcional)
        emotion.npy  -> data/emotion/balanced-affectnet/<s>/<f_label>/<face_id>.npy

Regras:
  - Splits seguem a partição ORIGINAL dos dados (sem frações novas):
      train  ← partição original "train" (RWF-2000 e balanced-affectnet);
      val/test ← partição original "val" (e "test", se existir) dividida
      50/50 por classe com seed — mesmo protocolo do pipeline CSV
      (val_test_split_ratio=0.5, seed=42), que resultava em ~86% de acurácia.
  - train/val contêm apenas pares CONGRUENTES (vídeo e face mesmo rótulo).
  - test contém 4 células de mesmo tamanho (reuso controlado de vídeo):
        violent_violent_face, violent_non_violent_face,
        non_violent_non_violent_face, non_violent_violent_face
    Cada vídeo de teste aparece em 2 células (face congruente e incongruente).
  - Target = rótulo do VÍDEO; rótulo da face é covariável (cell codifica ambos).

A árvore de diretórios é a fonte da verdade; manifest.json guarda apenas
configuração e contagens (auditoria).

Uso:
    python -m src.preprocessing.build_paired_dataset
    python -m src.preprocessing.build_paired_dataset --seed 42
    python -m src.preprocessing.build_paired_dataset --validate
    python -m src.preprocessing.build_paired_dataset --force --reuse_faces
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src import paths as p

LABEL_MAP = {"violent": 1, "non_violent": 0}
CLASSES = ("violent", "non_violent")

# célula -> (rótulo do vídeo, rótulo da face)
CELLS = {
    "violent_violent_face": ("violent", "violent"),
    "violent_non_violent_face": ("violent", "non_violent"),
    "non_violent_non_violent_face": ("non_violent", "non_violent"),
    "non_violent_violent_face": ("non_violent", "violent"),
}
CONGRUENT_CELLS = ("violent_violent_face", "non_violent_non_violent_face")
INCONGRUENT_CELLS = ("violent_non_violent_face", "non_violent_violent_face")
TEST_CELLS = (
    "violent_violent_face",
    "violent_non_violent_face",
    "non_violent_non_violent_face",
    "non_violent_violent_face",
)

MANIFEST_NAME = "manifest.json"
SEPARATOR = "__"


def cell_for(video_class: str, face_class: str) -> str:
    for cell, (v, f) in CELLS.items():
        if v == video_class and f == face_class:
            return cell
    raise ValueError(f"Combinação inválida: {video_class} + {face_class}")


# ── Inventário ──────────────────────────────────────────────────────────────

def inventory_videos() -> List[Dict]:
    """Varre data/processed/<orig_split>/<classe>/<video_id>/frame_sequence.pt."""
    videos = []
    base = p.PROCESSED_ROOT
    if not base.exists():
        return videos
    for orig_split in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for cls in CLASSES:
            cls_dir = base / orig_split / cls
            if not cls_dir.is_dir():
                continue
            for video_dir in sorted(cls_dir.iterdir()):
                frames = video_dir / "frame_sequence.pt"
                if not (video_dir.is_dir() and frames.exists()):
                    continue
                pose = p.POSE_ROOT / "rwf2000" / orig_split / cls / f"{video_dir.name}.npy"
                videos.append({
                    "video_id": video_dir.name,
                    "class": cls,
                    "label": LABEL_MAP[cls],
                    "orig_split": orig_split,
                    "frames_path": frames,
                    "features_path": video_dir / "features.pt",
                    "pose_path": pose if pose.exists() else None,
                })
    return videos


def inventory_faces() -> List[Dict]:
    """Varre data/emotion/balanced-affectnet/<orig_split>/<classe>/<face_id>.npy."""
    faces = []
    base = p.EMOTION_ROOT / "balanced-affectnet"
    if not base.exists():
        return faces
    for orig_split in sorted(d.name for d in base.iterdir() if d.is_dir()):
        for cls in CLASSES:
            cls_dir = base / orig_split / cls
            if not cls_dir.is_dir():
                continue
            for npy in sorted(cls_dir.glob("*.npy")):
                faces.append({
                    "face_id": npy.stem,
                    "class": cls,
                    "label": LABEL_MAP[cls],
                    "orig_split": orig_split,
                    "path": npy,
                })
    return faces


# ── Split pela partição original ────────────────────────────────────────────

VAL_TEST_RATIO = 0.5


def assign_splits(
    records: List[Dict],
    seed: int,
    key: str = "class",
) -> Dict[str, List[Dict]]:
    """Restaura a partição ORIGINAL dos dados.

    - orig_split == "train" → train (nunca é dividido);
    - orig_split != "train" (val, test, ...) → pool dividido 50/50 em
      val/test, por classe, com embaralhamento determinístico por seed.

    É o mesmo protocolo do pipeline CSV original (val_test_split_ratio=0.5),
    sem misturar a partição train com val/test.
    """
    result = {"train": [], "val": [], "test": []}
    remainder: List[Dict] = []
    for r in records:
        if r["orig_split"] == "train":
            result["train"].append(r)
        else:
            remainder.append(r)
    for cls in CLASSES:
        pool = [r for r in remainder if r[key] == cls]
        rng = random.Random(f"{seed}:orig:{key}:{cls}")
        rng.shuffle(pool)
        cut = int(len(pool) * VAL_TEST_RATIO)
        result["val"].extend(pool[:cut])
        result["test"].extend(pool[cut:])
    return result


# ── Pareamento ──────────────────────────────────────────────────────────────

def _pair_id(video: Dict, face: Dict) -> str:
    return f"{video['video_id']}{SEPARATOR}{face['face_id']}"


def _pair_train_val(
    videos: List[Dict],
    faces: List[Dict],
    seed: int,
    reuse_faces: bool,
    split: str,
) -> List[Dict]:
    """Pares congruentes: por classe, min(vídeos, faces) (ou ciclagem de faces)."""
    pairs = []
    for cls in CLASSES:
        v_pool = [v for v in videos if v["class"] == cls]
        f_pool = [f for f in faces if f["class"] == cls]
        rng = random.Random(f"{seed}:pair:{split}:{cls}")
        rng.shuffle(v_pool)
        rng.shuffle(f_pool)
        if not v_pool or not f_pool:
            continue
        if reuse_faces and len(v_pool) > len(f_pool):
            n = len(v_pool)
        else:
            n = min(len(v_pool), len(f_pool))
        for i in range(n):
            pairs.append((v_pool[i], f_pool[i % len(f_pool)]))
    return pairs


def _pair_test(
    videos: List[Dict],
    faces: List[Dict],
    seed: int,
    reuse_faces: bool,
) -> List[Dict]:
    """
    4 células de mesmo tamanho com reuso controlado de vídeo.

    Cada vídeo de teste aparece em exatamente 2 células (face congruente e
    incongruente). Faces são disjuntas entre células por padrão.
    """
    v_pools = {cls: [v for v in videos if v["class"] == cls] for cls in CLASSES}
    f_pools = {cls: [f for f in faces if f["class"] == cls] for cls in CLASSES}
    for cls in CLASSES:
        random.Random(f"{seed}:test:{cls}:v").shuffle(v_pools[cls])
        random.Random(f"{seed}:test:{cls}:f").shuffle(f_pools[cls])

    if any(not v_pools[c] for c in CLASSES) or any(not f_pools[c] for c in CLASSES):
        return []

    if reuse_faces:
        n = min(len(v_pools["violent"]), len(v_pools["non_violent"]))
    else:
        n = min(
            len(v_pools["violent"]),
            len(v_pools["non_violent"]),
            len(f_pools["violent"]) // 2,
            len(f_pools["non_violent"]) // 2,
        )
    if n <= 0:
        return []

    # offsets disjuntos de face por rótulo (cada face usada no máximo 1x)
    offset = {"violent": 0, "non_violent": 0}
    pairs = []
    for video_class in CLASSES:
        for face_class in CLASSES:
            cell = cell_for(video_class, face_class)
            for i in range(n):
                video = v_pools[video_class][i]
                f_pool = f_pools[face_class]
                face = f_pool[(offset[face_class] + i) % len(f_pool)]
                pairs.append({"video": video, "face": face, "cell": cell})
            offset[face_class] += n
    return pairs


# ── Materialização ──────────────────────────────────────────────────────────

def _symlink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(target, start=link.parent))


def _write_pair(pair_dir: Path, video: Dict, face: Dict) -> Dict[str, bool]:
    pair_dir.mkdir(parents=True, exist_ok=True)
    files = {"frames": False, "features": False, "pose": False, "emotion": False}

    _symlink(pair_dir / "frames.pt", video["frames_path"])
    files["frames"] = True

    if video.get("features_path") and video["features_path"].exists():
        _symlink(pair_dir / "features.pt", video["features_path"])
        files["features"] = True
    if video.get("pose_path"):
        _symlink(pair_dir / "pose.npy", video["pose_path"])
        files["pose"] = True
    _symlink(pair_dir / "emotion.npy", face["path"])
    files["emotion"] = True
    return files


def build(
    seed: int = 42,
    reuse_faces: bool = False,
    output_root: Optional[Path] = None,
    force: bool = False,
) -> Dict:
    output_root = Path(output_root) if output_root else p.PAIRED_ROOT

    videos = inventory_videos()
    faces = inventory_faces()
    if not videos:
        raise FileNotFoundError(
            f"Nenhum frame_sequence.pt encontrado em {p.PROCESSED_ROOT}. "
            "Rode a extração de frames antes (python run_preprocessing.py)."
        )
    if not faces:
        raise FileNotFoundError(
            f"Nenhum .npy de emoção encontrado em {p.EMOTION_ROOT / 'balanced-affectnet'}. "
            "Rode a extração de emoção do AffectNet antes "
            "(python -m src.preprocessing.emotion --from-affectnet)."
        )

    video_splits = assign_splits(videos, seed, key="class")
    face_splits = assign_splits(faces, seed, key="class")

    if output_root.exists():
        if not force:
            raise FileExistsError(
                f"{output_root} já existe. Use --force para reconstruir."
            )
        shutil.rmtree(output_root)

    counts: Dict[str, Counter] = {s: Counter() for s in ("train", "val", "test")}
    file_counts = Counter()

    for split in ("train", "val"):
        pairs = _pair_train_val(
            video_splits[split], face_splits[split], seed, reuse_faces, split
        )
        for video, face in pairs:
            cell = cell_for(video["class"], face["class"])
            pair_id = _pair_id(video, face)
            pair_dir = output_root / split / cell / pair_id
            files = _write_pair(pair_dir, video, face)
            counts[split][cell] += 1
            for k, ok in files.items():
                if ok:
                    file_counts[k] += 1

    test_pairs = _pair_test(video_splits["test"], face_splits["test"], seed, reuse_faces)
    for item in test_pairs:
        video, face, cell = item["video"], item["face"], item["cell"]
        pair_id = _pair_id(video, face)
        pair_dir = output_root / "test" / cell / pair_id
        files = _write_pair(pair_dir, video, face)
        counts["test"][cell] += 1
        for k, ok in files.items():
            if ok:
                file_counts[k] += 1

    # igualdade estrita das 4 células de teste
    test_cell_sizes = [counts["test"][c] for c in TEST_CELLS]
    if len(set(test_cell_sizes)) > 1:
        print(
            f"⚠️  Células de teste com tamanhos diferentes: "
            f"{dict(zip(TEST_CELLS, test_cell_sizes))}"
        )

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "split": "original_partition",
            "val_test_ratio": VAL_TEST_RATIO,
            "seed": seed,
            "reuse_faces": reuse_faces,
            "target": "video_label",
            "separator": SEPARATOR,
        },
        "inventory": {
            "videos": dict(Counter(
                f"{v['orig_split']}/{v['class']}" for v in videos
            )),
            "faces": dict(Counter(
                f"{f['orig_split']}/{f['class']}" for f in faces
            )),
            "pose_available": sum(1 for v in videos if v["pose_path"]),
        },
        "assigned": {
            split: {
                "videos": dict(Counter(v["class"] for v in video_splits[split])),
                "faces": dict(Counter(f["class"] for f in face_splits[split])),
            }
            for split in ("train", "val", "test")
        },
        "counts": {split: dict(c) for split, c in counts.items()},
        "files": dict(file_counts),
        "n_pairs": sum(sum(c.values()) for c in counts.values()),
    }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_NAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print("Dataset pareado gerado")
    print("=" * 60)
    print(f"Raiz: {output_root}")
    print(f"Split: partição original (train → train; val/test ← val 50/50)")
    print(f"Inventário: {len(videos)} vídeos, {len(faces)} faces")
    for split in ("train", "val", "test"):
        total_split = sum(counts[split].values())
        print(f"  {split:<6}: {total_split:>5} pares  {dict(counts[split])}")
    print(f"Manifest: {manifest_path}")
    return manifest


# ── Validação ───────────────────────────────────────────────────────────────

def validate(output_root: Optional[Path] = None) -> Tuple[int, List[str]]:
    """Verifica manifest, symlinks e pares completos. Retorna (n_ok, erros)."""
    output_root = Path(output_root) if output_root else p.PAIRED_ROOT
    errors: List[str] = []
    manifest_path = output_root / MANIFEST_NAME
    if not manifest_path.exists():
        return 0, [f"Manifest não encontrado: {manifest_path}"]

    n_ok = 0
    for link in sorted(output_root.rglob("*")):
        if not link.is_symlink():
            continue
        if link.exists():  # resolve (não quebrado)
            n_ok += 1
        else:
            errors.append(f"symlink quebrado: {link} -> {os.readlink(link)}")

    # pares completos: frames.pt + emotion.npy obrigatórios
    for split_dir in sorted(output_root.glob("*")):
        if not split_dir.is_dir():
            continue
        for cell_dir in sorted(split_dir.glob("*")):
            if not cell_dir.is_dir() or cell_dir.name not in CELLS:
                continue
            for pair_dir in sorted(cell_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                for required in ("frames.pt", "emotion.npy"):
                    target = pair_dir / required
                    if not (target.exists() and target.is_symlink()):
                        errors.append(f"par incompleto: {target} ausente")
    return n_ok, errors


def main():
    parser = argparse.ArgumentParser(
        prog="build_paired_dataset.py",
        description="Gera dataset/paired (RWF-2000 × AffectNet) com symlinks.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para splits e pareamento (padrão: 42)")
    parser.add_argument("--reuse_faces", action="store_true",
                        help="Permite ciclar faces quando há menos faces que vídeos")
    parser.add_argument("--output", type=str, default=None,
                        help="Raiz de saída (padrão: dataset/paired)")
    parser.add_argument("--force", action="store_true",
                        help="Reconstruir mesmo se o diretório de saída existir")
    parser.add_argument("--validate", action="store_true",
                        help="Apenas validar symlinks e manifest existentes")
    args = parser.parse_args()

    output_root = Path(args.output) if args.output else p.PAIRED_ROOT

    if args.validate:
        n_ok, errors = validate(output_root)
        print(f"Symlinks OK: {n_ok}")
        for err in errors:
            print(f"✗ {err}")
        sys.exit(1 if errors else 0)

    build(
        seed=args.seed,
        reuse_faces=args.reuse_faces,
        output_root=output_root,
        force=args.force,
    )
    n_ok, errors = validate(output_root)
    print(f"Validação: {n_ok} symlinks resolvendo, {len(errors)} erros")
    for err in errors:
        print(f"✗ {err}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
