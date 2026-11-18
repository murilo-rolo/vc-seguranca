"""Extração de features de vídeo congelada no treino multimodal."""

import torch
import torch.nn as nn

from train_multimodal import FrozenVideoFeatureExtractor


class _StubVideo(nn.Module):
    """Backbone mínimo com a interface get_features (B,C,T,H,W)->(B,D)."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.lin = nn.Linear(3 * 2 * 2 * 2, feature_dim)

    def get_features(self, x):
        assert x.shape[1] == 3
        return self.lin(x.reshape(x.shape[0], -1))


class _StubMultimodal(nn.Module):
    """Fusão mínima: linear sobre o clip token de vídeo."""

    def __init__(self, video_dim: int = 4):
        super().__init__()
        self.head = nn.Linear(video_dim, 2)

    def forward(self, video, pose, emotion):
        return self.head(video)


def _make_wrapper():
    video = _StubVideo()
    multimodal = _StubMultimodal()
    return video, multimodal, FrozenVideoFeatureExtractor(multimodal, video)


def test_wrapper_converte_frames_em_features():
    _, _, model = _make_wrapper()
    out = model(torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4), torch.randn(2, 4))
    assert out.shape == (2, 2)


def test_wrapper_mantem_backbone_congelado_e_eval():
    video, multimodal, model = _make_wrapper()
    model.train(True)
    assert all(not p.requires_grad for p in video.parameters())
    assert not video.training

    out = model(torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4), torch.randn(2, 4))
    out.sum().backward()
    assert video.lin.weight.grad is None
    assert multimodal.head.weight.grad is not None
