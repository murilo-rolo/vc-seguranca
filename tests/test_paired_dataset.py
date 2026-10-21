"""Testes do PairedSurveillanceDataset (emotion-swap e force_emotion_swap)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.datasets.paired_dataset import PairedSurveillanceDataset

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


def test_val_sem_force_nao_troca(tmp_path):
    root = _make_paired_root(tmp_path / "paired")
    ds = PairedSurveillanceDataset(split="val", paired_root=str(root))
    assert ds.emotion_swap_prob == 0.0
    assert ds._emotion_bank == {0: [], 1: []}


def test_force_emotion_swap_ativa_troca_total_no_val(tmp_path):
    root = _make_paired_root(tmp_path / "paired")
    ds = PairedSurveillanceDataset(
        split="val", paired_root=str(root), force_emotion_swap=True,
    )
    assert ds.emotion_swap_prob == 1.0
    assert len(ds._emotion_bank[0]) == 1
    assert len(ds._emotion_bank[1]) == 1

    # Cada amostra recebe a face do label OPPOSTO (swap 100%).
    for idx in range(len(ds)):
        sample = ds.samples[idx]
        _, _, emotion, _ = ds[idx]
        own = 1.0 if sample["face_label"] == 1 else 2.0
        swapped = 2.0 if own == 1.0 else 1.0
        assert float(emotion.mean()) == pytest.approx(swapped)


def test_force_emotion_swap_desabilita_quando_label_ausente(tmp_path):
    # Só face violent no split: não há bank trocável -> desativa com warning.
    root = _make_paired_root(tmp_path / "paired", cells=(CELL_VIOLENT,))
    ds = PairedSurveillanceDataset(
        split="val", paired_root=str(root), force_emotion_swap=True,
    )
    assert ds.emotion_swap_prob == 0.0


def test_train_com_swap_exige_dois_labels(tmp_path):
    root = _make_paired_root(tmp_path / "paired", cells=(CELL_VIOLENT,), split="train")
    with pytest.raises(ValueError, match="dois labels"):
        PairedSurveillanceDataset(
            split="train", paired_root=str(root), emotion_swap_prob=0.5,
        )
