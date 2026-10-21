"""Fine-tune do backbone CNN3D no treino multimodal (Fase F)."""

import torch
import torch.nn as nn

from train_multimodal import MultimodalWithVideoBackbone


class _StubVideo(nn.Module):
    """Backbone mínimo com a interface get_features (B,T,C,H,W)->(B,D)."""

    def __init__(self, feature_dim: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.lin = nn.Linear(3 * 2 * 2 * 2, feature_dim)

    def get_features(self, x):
        return self.lin(x.reshape(x.shape[0], -1))


class _StubMultimodal(nn.Module):
    """Fusão mínima: linear sobre o clip token de vídeo."""

    def __init__(self, video_dim: int = 4):
        super().__init__()
        self.head = nn.Linear(video_dim, 2)

    def forward(self, video, pose, emotion, return_aux: bool = False):
        return self.head(video)


def _forward_and_backward(finetune_backbone: bool):
    video = _StubVideo()
    model = MultimodalWithVideoBackbone(
        _StubMultimodal(), video, finetune_backbone=finetune_backbone,
    )
    x = torch.randn(2, 2, 3, 2, 2)  # (B, T, C, H, W)
    out = model(x, torch.randn(2, 4), torch.randn(2, 4))
    out.sum().backward()
    return video


def test_finetune_true_backbone_recebe_grads():
    video = _forward_and_backward(finetune_backbone=True)
    assert video.lin.weight.grad is not None
    assert video.lin.weight.grad.abs().sum() > 0


def test_finetune_false_backbone_sem_grads():
    video = _forward_and_backward(finetune_backbone=False)
    assert video.lin.weight.grad is None


def test_finetune_true_grad_flui_tambem_pela_fusao():
    video = _StubVideo()
    mm = _StubMultimodal()
    model = MultimodalWithVideoBackbone(mm, video, finetune_backbone=True)
    out = model(torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4), torch.randn(2, 4))
    out.sum().backward()
    assert mm.head.weight.grad is not None
