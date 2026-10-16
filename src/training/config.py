"""
Configuração de hiperparâmetros de treino.

Fonte única: `configs/treino.toml` (seções [emotion], [cnn3d], [multimodal]).
Precedência: DEFAULTS do código < arquivo TOML < overrides de CLI
(--device/--epochs, quando passados).

Os scripts de treino só expõem `--config`, `--device`, `--epochs` (e
`--resume` no emotion); todo o resto vive no TOML.
"""

import copy
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from src import paths as p

DEFAULT_CONFIG_PATH = p.PROJECT_ROOT / "configs" / "treino.toml"

DEFAULTS: Dict[str, Dict[str, Any]] = {
    "emotion": {
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "weight_decay": 0.05,
        "warmup_epochs": 5,
        "early_stop_patience": 8,
        "focal_gamma": 2.0,
        "grad_clip": 1.0,
        "balance_classes": False,
        "amp": True,
        "min_confidence": 0.7,
        "num_workers": 4,
        "dataset_path": "dataset/balanced-affectnet",
    },
    "cnn3d": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "num_workers": 2,
        "model_name": "r2plus1d_18",
        "freeze_backbone": False,
        "num_frames": 16,
        "clip_size": [112, 112],
        "dropout": 0.4,
    },
    "multimodal": {
        "epochs": 50,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "num_workers": 4,
        "num_frames": 16,
        "window_size": 16,
        "use_temporal_modeling": False,
        "video_model_path": str(p.CNN3D_WEIGHTS / "best_model.pth"),
    },
}


def add_config_args(parser) -> None:
    """Adiciona os argumentos comuns de CLI de todos os scripts de treino."""
    import argparse
    import torch

    parser.add_argument(
        "--config", type=str, default=None,
        help=f"Arquivo TOML com hiperparâmetros (padrão: {DEFAULT_CONFIG_PATH.name})",
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device para treinamento",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Sobrescreve epochs do arquivo de configuração",
    )


def load_config(model: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega a config de um modelo: defaults < TOML (seção do modelo).

    Raises:
        ValueError: modelo ou chaves desconhecidas.
    """
    if model not in DEFAULTS:
        raise ValueError(f"Modelo sem configuração: {model}. Opções: {list(DEFAULTS)}")

    cfg = copy.deepcopy(DEFAULTS[model])
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        with open(path, "rb") as f:
            data = tomllib.load(f)
        section = data.get(model, {})
        unknown = set(section) - set(cfg)
        if unknown:
            raise ValueError(
                f"Chaves desconhecidas na seção [{model}] de {path}: {sorted(unknown)}. "
                f"Chaves válidas: {sorted(cfg)}"
            )
        cfg.update(section)
    elif config_path is not None:
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")

    return cfg


def resolve_config(model: str, args) -> Dict[str, Any]:
    """Combina o TOML com os overrides mínimos de CLI (device/epochs)."""
    cfg = load_config(model, getattr(args, "config", None))
    cfg["device"] = args.device
    if getattr(args, "epochs", None) is not None:
        cfg["epochs"] = args.epochs
    return cfg
