"""Testes do PairedSurveillanceDataset e do gerador de pares (cross-label)."""

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.paired_dataset import PairedSurveillanceDataset
from src.preprocessing import build_paired_dataset as generator

CELL_VIOLENT = "violent_violent_face"
CELL_NON_VIOLENT = "non_violent_non_violent_face"


def _make_paired_root(
    root: Path, cells=(CELL_VIOLENT, CELL_NON_VIOLENT), split: str = "val",
) -> Path:
    """Cria um split mínimo: um par por célula com emotion distinto."""
    for cell in cells:
        pair = root / split / cell / "vid0__face0"
        pair.mkdir(parents=True)
        torch.save(torch.zeros(4, 3, 8, 8), pair / "frames.pt")
        # violent -> 1.0, non_violent -> 2.0 (assinatura p/ detectar o swap)
        value = 1.0 if cell == CELL_VIOLENT else 2.0
        np.save(pair / "emotion.npy", np.full((16, 128), value, dtype=np.float32))
    return root


def test_val_sem_swap_nao_troca(tmp_path):
    root = _make_paired_root(tmp_path / "paired")
    ds = PairedSurveillanceDataset(split="val", paired_root=str(root))
    assert ds.emotion_swap_prob == 0.0
    assert ds._emotion_bank == {0: [], 1: []}


def test_train_com_swap_exige_dois_labels(tmp_path):
    root = _make_paired_root(tmp_path / "paired", cells=(CELL_VIOLENT,), split="train")
    with pytest.raises(ValueError, match="dois labels"):
        PairedSurveillanceDataset(
            split="train", paired_root=str(root), emotion_swap_prob=0.5,
        )


def _toy_pools(per_class: int = 4):
    videos, faces = [], []
    for cls in generator.CLASSES:
        for i in range(per_class):
            videos.append({"video_id": f"v_{cls}_{i}", "class": cls})
            faces.append({"face_id": f"f_{cls}_{i}", "class": cls})
    return videos, faces


def test_pair_train_val_sem_ratio_gera_somente_congruentes():
    videos, faces = _toy_pools()
    pairs = generator._pair_train_val(videos, faces, 42, False, "train")
    assert len(pairs) == 8
    assert all(v["class"] == f["class"] for v, f in pairs)


def test_pair_train_val_com_ratio_adiciona_incongruentes():
    videos, faces = _toy_pools()
    pairs = generator._pair_train_val(
        videos, faces, 42, False, "train", cross_label_ratio=0.5,
    )
    congruentes = [p for p in pairs if p[0]["class"] == p[1]["class"]]
    incongruentes = [p for p in pairs if p[0]["class"] != p[1]["class"]]
    assert len(congruentes) == 8
    assert len(incongruentes) == 4
    ids = [f"{v['video_id']}__{f['face_id']}" for v, f in pairs]
    assert len(set(ids)) == len(ids)


def test_pair_train_val_rejeita_ratio_fora_de_zero_a_um():
    videos, faces = _toy_pools()
    with pytest.raises(ValueError, match="cross_label_ratio"):
        generator._pair_train_val(videos, faces, 42, False, "train", 1.5)


def test_pair_test_val_tem_quatro_celulas_iguais():
    videos, faces = _toy_pools()
    pairs = generator._pair_test(videos, faces, 42, False, split="val")
    counts = Counter(p["cell"] for p in pairs)
    assert set(counts) == set(generator.TEST_CELLS)
    assert set(counts.values()) == {2}


def test_build_registra_ratio_e_celulas_no_manifest(tmp_path, monkeypatch):
    videos, faces = [], []
    for cls in generator.CLASSES:
        for split_name in ("train", "val"):
            for i in range(2):
                frames = tmp_path / "frames" / split_name / cls / f"v_{cls}_{split_name}_{i}.pt"
                frames.parent.mkdir(parents=True, exist_ok=True)
                frames.write_bytes(b"x")
                videos.append({
                    "video_id": f"v_{cls}_{split_name}_{i}",
                    "class": cls,
                    "label": generator.LABEL_MAP[cls],
                    "orig_split": split_name,
                    "frames_path": frames,
                    "features_path": None,
                    "pose_path": None,
                })
            for i in range(4):
                emotion = tmp_path / "emotion" / split_name / cls / f"f_{cls}_{split_name}_{i}.npy"
                emotion.parent.mkdir(parents=True, exist_ok=True)
                emotion.write_bytes(b"x")
                faces.append({
                    "face_id": f"f_{cls}_{split_name}_{i}",
                    "class": cls,
                    "label": generator.LABEL_MAP[cls],
                    "orig_split": split_name,
                    "path": emotion,
                })

    monkeypatch.setattr(generator, "inventory_videos", lambda: videos)
    monkeypatch.setattr(generator, "inventory_faces", lambda: faces)

    manifest = generator.build(
        seed=42, output_root=tmp_path / "paired", cross_label_train_ratio=0.5,
    )
    assert manifest["config"]["cross_label_train_ratio"] == 0.5
    assert manifest["config"]["val_cells"] == list(generator.TEST_CELLS)
    for split in ("train", "val", "test"):
        assert set(manifest["counts"][split]) == set(generator.TEST_CELLS)
