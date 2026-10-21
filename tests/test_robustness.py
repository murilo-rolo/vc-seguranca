"""Testes das distorções de robustez com dados sintéticos (sem datasets reais)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.robustness_eval import DEFAULT_DISTORTION_CONFIGS, RobustnessEvaluator
from src.evaluation.utils import apply_distortions

SIZES = [(112, 112), (224, 224)]


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("distortion_type", list(DEFAULT_DISTORTION_CONFIGS))
def test_apply_distortions_preserves_shape(distortion_type, h, w):
    # Regressão: resize_down usava 224 hardcoded e mudava o shape do frame,
    # quebrando o reshape de _apply_distortion_to_tensor para clips 112x112.
    frame = np.random.default_rng(0).integers(0, 255, size=(h, w, 3), dtype=np.uint8)
    for intensity in DEFAULT_DISTORTION_CONFIGS[distortion_type]:
        out = apply_distortions(frame, distortion_type, intensity)
        assert out.shape == frame.shape, (
            f"{distortion_type} intensity={intensity}: {frame.shape} -> {out.shape}"
        )


class _DummyVideoModel(torch.nn.Module):
    """Tolerante a frame-last (B,T,C,H,W) ou channel-first (B,C,T,H,W)."""

    def forward(self, x):
        if x.dim() == 5 and x.shape[1] != 3 and x.shape[2] == 3:
            x = x.permute(0, 2, 1, 3, 4)
        s = x.mean(dim=(1, 2, 3, 4))
        return torch.stack([-s, s], dim=1)


def _frame_last_loader(batch_size: int = 2):
    """Clipes frame-last (B, T, C, H, W) 112x112 como o RWF-2000 3D devolve."""
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    clips = torch.rand(len(labels), 4, 3, 112, 112)
    return DataLoader(TensorDataset(clips, labels), batch_size=batch_size)


@pytest.mark.parametrize("distortion_type", list(DEFAULT_DISTORTION_CONFIGS))
def test_robustness_video_frame_last_all_distortions(distortion_type):
    # Regressão do crash: reshape '[8,16,3,112,112]' invalid for input of size ...
    # (resize_down em intensity=0.0 dobrava H/W de clips 112x112).
    evaluator = RobustnessEvaluator(_DummyVideoModel().eval(), _frame_last_loader(), device="cpu")
    metrics = evaluator._evaluate_with_distortion(
        distortion_type, DEFAULT_DISTORTION_CONFIGS[distortion_type][0], apply_to_input=True
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
