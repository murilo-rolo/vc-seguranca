"""
Estatísticas de features (emotion/pose) do split de treino.

Servem para normalizar as streams de features no dataset pareado — reduz
dominância por escala bruta entre modalidades (embeddings do EmotionNet
entram sem nenhuma norma hoje).

Cálculo: mean/std por dimensão, acumulados sobre todos os timesteps de
todas as amostras do split (sem carregar frames.pt — só os .npy).
"""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _flatten_time(x: np.ndarray) -> np.ndarray:
    """(T, D...) -> (T*D_last) view por timestep: (T, D_prod)."""
    if x.ndim == 1:
        x = x.reshape(1, -1)
    return x.reshape(x.shape[0], -1)


def compute_feature_stats(dataset, eps: float = 1e-6) -> Dict:
    """
    Calcula mean/std por dimensão de emotion e pose a partir dos pares do dataset.

    Args:
        dataset: PairedSurveillanceDataset (usa .samples com pair_dir).
        eps: pseudo-count para std (evita divisão por zero).

    Returns:
        {"emotion": {"mean": [...], "std": [...]}, "pose": {...}}
    """
    acc = {
        "emotion": {"count": 0, "sum": None, "sumsq": None, "dim": None},
        "pose": {"count": 0, "sum": None, "sumsq": None, "dim": None},
    }

    for sample in dataset.samples:
        pair_dir = sample["pair_dir"]
        for key, filename in (("emotion", "emotion.npy"), ("pose", "pose.npy")):
            path = pair_dir / filename
            if not path.exists():
                continue
            x = _flatten_time(np.load(path, mmap_mode="r").astype(np.float64))
            d = x.shape[1]
            slot = acc[key]
            if slot["dim"] is None:
                slot["dim"] = d
                slot["sum"] = np.zeros(d, dtype=np.float64)
                slot["sumsq"] = np.zeros(d, dtype=np.float64)
            elif slot["dim"] != d:
                raise ValueError(
                    f"Dimensão inconsistente de {key}: {d} != {slot['dim']} em {path}"
                )
            slot["sum"] += x.sum(axis=0)
            slot["sumsq"] += (x ** 2).sum(axis=0)
            slot["count"] += x.shape[0]

    stats: Dict = {}
    for key, slot in acc.items():
        if slot["count"] == 0:
            continue
        n = slot["count"]
        mean = slot["sum"] / n
        var = slot["sumsq"] / n - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0)) + eps
        stats[key] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "count": int(n),
        }
    return stats


def save_feature_stats(stats: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_feature_stats(path: Path) -> Optional[Dict]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def normalize_features(x: torch.Tensor, stats: Dict) -> torch.Tensor:
    """
    Aplica (x - mean) / std por dimensão.

    Args:
        x: (T, D) ou (T, J, 3) — a normalização usa as últimas dimensões
           compatíveis com len(mean).
        stats: {"mean": [...], "std": [...]}
    """
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype)
    std = torch.as_tensor(stats["std"], dtype=x.dtype)
    if x.shape[-1] != mean.numel():
        raise ValueError(
            f"Última dimensão {x.shape[-1]} != len(mean) {mean.numel()}"
        )
    return (x - mean) / std


__all__ = [
    "compute_feature_stats",
    "save_feature_stats",
    "load_feature_stats",
    "normalize_features",
]
