"""
Helper para computar o embedding neutro de emoções (EMB-02).

O embedding neutro é a média dos embeddings de penúltima camada
(EmotionNet.extract_features, 128-d) dos frames classificados como 'neutral'
no split de validação. É usado como fallback para frames sem face detectada
(AD-004), substituindo o vetor one-hot.

Quando o modelo ou os dados não estão disponíveis, retorna um vetor
determinístico de zeros (128,) com um warning.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

NEUTRAL_EMBEDDING_DIM = 128


def compute_neutral_embedding(
    model=None,
    val_dir: Optional[str] = None,
    cache_path: Optional[str] = None,
    force: bool = False,
    device: str = "cpu",
) -> "np.ndarray":
    """
    Computa o embedding neutro (128,) a partir dos embeddings do val split.

    Estratégia:
    1. Se `cache_path` existe e `force=False`, carrega o cache.
    2. Se `model` e `val_dir` estão disponíveis, calcula a média dos
       embeddings (T, 128) dos frames cuja classe predita é 'neutral'.
    3. Caso contrário, retorna zeros determinísticos (128,) com warning.

    Args:
        model: EmotionNet em modo eval (com extract_features). None -> zeros.
        val_dir: Diretório com arquivos .npy (T, 128) do split de validação.
        cache_path: Caminho opcional do cache .npy.
        force: Se True, ignora o cache e sobrescreve.
        device: Device para mover os tensores.

    Returns:
        np.ndarray de shape (128,) com o embedding neutro.
    """
    try:
        import numpy as np
    except ImportError as e:
        logger.warning(
            "numpy indisponível (%s); usando zeros (%d,)",
            e, NEUTRAL_EMBEDDING_DIM,
        )
        return _zeros_fallback()

    if cache_path is not None and not force:
        cache = Path(cache_path)
        if cache.exists():
            try:
                arr = np.load(str(cache))
                if arr.shape == (NEUTRAL_EMBEDDING_DIM,):
                    return arr
                logger.warning(
                    "Cache de embedding neutro com shape inesperado %s; recomputando.",
                    arr.shape,
                )
            except Exception as e:
                logger.warning("Erro ao carregar cache %s: %s", cache_path, e)

    try:
        import torch
    except ImportError:
        torch = None

    neutral = None
    if model is not None and val_dir is not None and torch is not None:
        try:
            neutral = _mean_neutral_from_val(model, Path(val_dir), device, np, torch)
        except Exception as e:
            logger.warning(
                "Falha ao computar embedding neutro (%s); usando zeros (%d,).",
                e, NEUTRAL_EMBEDDING_DIM,
            )

    if neutral is None:
        logger.warning(
            "Modelo e/ou dados de val indisponíveis; usando zeros (%d,).",
            NEUTRAL_EMBEDDING_DIM,
        )
        neutral = np.zeros(NEUTRAL_EMBEDDING_DIM, dtype=np.float32)

    if cache_path is not None:
        try:
            cache = Path(cache_path)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(cache), neutral)
        except Exception as e:
            logger.warning("Erro ao salvar cache %s: %s", cache_path, e)

    return neutral


def _zeros_fallback() -> "np.ndarray":
    """Vetor determinístico de zeros (128,)."""
    try:
        import numpy as np
        return np.zeros(NEUTRAL_EMBEDDING_DIM, dtype=np.float32)
    except ImportError:
        return [0.0] * NEUTRAL_EMBEDDING_DIM


def _mean_neutral_from_val(model, val_dir: Path, device: str, np, torch):
    """
    Média dos embeddings das frames preditas como 'neutral' no val split.

    Para cada .npy (T, 128), prediz a classe via a cabeça do classificador
    (Linear(128, num_emotions)) sobre os embeddings e mantém as linhas cuja
    classe predita é o índice de 'neutral' em EmotionNet.EMOTION_CLASSES.
    """
    neutral_index = model.EMOTION_CLASSES.index("neutral")
    model.eval()

    neutral_rows = []
    for npy_file in sorted(val_dir.rglob("*.npy")):
        arr = np.load(str(npy_file))
        if arr.ndim != 2 or arr.shape[1] != NEUTRAL_EMBEDDING_DIM:
            logger.warning("Ignorando %s (shape %s)", npy_file, arr.shape)
            continue
        emb = torch.from_numpy(arr).float().to(device)
        with torch.no_grad():
            logits = model.classifier[3](emb)  # (T, num_emotions)
            preds = logits.argmax(dim=1).cpu().numpy()
        neutral_rows.append(arr[preds == neutral_index])

    if not neutral_rows:
        return None

    all_neutral = np.concatenate(neutral_rows, axis=0)
    if all_neutral.shape[0] == 0:
        return None

    return all_neutral.mean(axis=0).astype(np.float32)
