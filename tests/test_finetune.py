"""Fine-tune do backbone CNN3D no treino multimodal (Fases F e B)."""

import torch
import torch.nn as nn

from src.training.config import DEFAULTS
from src.training.loop import train_loop
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


# ---------------------------------------------------------------------------
# Fase B: fine-tune em 2 estágios (head-only → unfreeze)
# ---------------------------------------------------------------------------


def _make_two_stage():
    video = _StubVideo()
    mm = _StubMultimodal()
    model = MultimodalWithVideoBackbone(
        mm, video, finetune_backbone=True, head_only=True,
    )
    return video, mm, model


def test_config_defaults_fase_b():
    mm = DEFAULTS["multimodal"]
    assert mm["finetune_backbone"] is True
    assert mm["head_only_epochs"] == 5
    assert mm["backbone_weight_decay"] == 1e-3
    assert mm["backbone_lr"] == 1e-5


def test_head_only_congela_backbone():
    video, _, model = _make_two_stage()
    assert all(not p.requires_grad for p in video.parameters())
    assert model.head_only is True
    # run_epoch chama model.train(True) a cada época — vídeo deve ficar eval.
    model.train(True)
    assert not video.training


def test_head_only_sem_grad_no_backbone():
    video, _, model = _make_two_stage()
    model.train(True)
    out = model(torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4), torch.randn(2, 4))
    out.sum().backward()
    assert video.lin.weight.grad is None


def test_unfreeze_backbone_destrava_e_permita_grad():
    video, mm, model = _make_two_stage()
    model.unfreeze_backbone()
    assert model.head_only is False
    assert all(p.requires_grad for p in video.parameters())
    model.train(True)
    assert video.training  # fase 2: BN do vídeo em train
    out = model(torch.randn(2, 2, 3, 2, 2), torch.randn(2, 4), torch.randn(2, 4))
    out.sum().backward()
    assert video.lin.weight.grad is not None
    assert video.lin.weight.grad.abs().sum() > 0
    assert mm.head.weight.grad is not None


def test_head_only_false_comportamento_fase_f_preservado():
    video = _StubVideo()
    model = MultimodalWithVideoBackbone(
        _StubMultimodal(), video, finetune_backbone=True, head_only=False,
    )
    assert model.head_only is False
    assert all(p.requires_grad for p in video.parameters())
    model.train(True)
    assert video.training


def test_train_loop_on_epoch_start_recebe_epocas(tmp_path):
    from torch.utils.data import DataLoader, TensorDataset

    seen = []
    model = nn.Linear(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(
        TensorDataset(torch.randn(4, 4), labels), batch_size=2,
    )
    train_loop(
        model, loader, loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        device=torch.device("cpu"), epochs=3,
        output_dir=tmp_path / "w", experiments_dir=tmp_path / "e",
        on_epoch_start=seen.append,
    )
    assert seen == [1, 2, 3]
