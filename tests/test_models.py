"""Testes de modelos: forward com pesos aleatórios + carga dos checkpoints reais."""

from pathlib import Path

import pytest
import torch

from src import paths as p
from src.models.cnn3d_risk import CNN3DRiskDetector, create_cnn3d_model
from src.models.emotion_cnn import EmotionNet, create_emotion_model
from src.models.multimodal_risk import MultimodalRiskDetector, create_multimodal_model

# ---------------------------------------------------------------------------
# EmotionNet (DeiT-Small)
# ---------------------------------------------------------------------------


def test_emotion_net_forward():
    model = EmotionNet(num_emotions=2, pretrained=False)
    logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 2)


def test_emotion_net_extract_features():
    model = EmotionNet(num_emotions=2, pretrained=False)
    emb = model.extract_features(torch.randn(2, 3, 224, 224))
    assert emb.shape == (2, 128)


def test_create_emotion_model_from_real_checkpoint():
    ckpt = p.EMOTION_CNN_WEIGHTS / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("checkpoint de emoção não disponível")
    model = create_emotion_model(
        num_emotions=2, pretrained=False, checkpoint_path=str(ckpt), device="cpu"
    )
    model.eval()
    logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# CNN 3D
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("frame_first", [True, False])
def test_cnn3d_forward_accepts_both_layouts(frame_first):
    model = CNN3DRiskDetector(model_name="r3d_18", pretrained=False)
    if frame_first:
        x = torch.randn(2, 3, 4, 112, 112)
    else:
        x = torch.randn(2, 4, 3, 112, 112)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_cnn3d_get_features_matches_forward_layout():
    model = CNN3DRiskDetector(model_name="r3d_18", pretrained=False).eval()
    x_tcw = torch.randn(1, 3, 4, 112, 112)
    x_twc = x_tcw.permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        f1 = model.get_features(x_tcw)
        f2 = model.get_features(x_twc)
    assert f1.shape == (1, model.feature_dim)
    assert torch.allclose(f1, f2, atol=1e-5)


def test_create_cnn3d_from_real_checkpoint():
    ckpt = p.CNN3D_WEIGHTS / "best_model.pth"
    if not ckpt.exists():
        pytest.skip("checkpoint cnn3d não disponível")
    model = create_cnn3d_model(
        model_name="r2plus1d_18", num_classes=2,
        checkpoint_path=str(ckpt), device="cpu",
    )
    model.eval()
    logits = model(torch.randn(1, 3, 4, 112, 112))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Multimodal
# ---------------------------------------------------------------------------


def _multimodal_inputs():
    return (
        torch.randn(2, 512),          # video: clip token
        torch.randn(2, 8, 51),        # pose
        torch.randn(2, 8, 128),       # emotion
    )


def test_multimodal_forward():
    model = MultimodalRiskDetector()
    video, pose, emotion = _multimodal_inputs()
    logits = model(video, pose, emotion)
    assert logits.shape == (2, 2)


def test_multimodal_forward_legacy_video_sequence():
    model = MultimodalRiskDetector()
    video = torch.randn(2, 4, 512)    # legado (B, T, D_v) -> mean-pooled
    pose = torch.randn(2, 4, 51)
    emotion = torch.randn(2, 4, 128)
    assert model(video, pose, emotion).shape == (2, 2)


def test_multimodal_forward_without_temporal():
    model = MultimodalRiskDetector(use_temporal_modeling=False)
    assert model(*_multimodal_inputs()).shape == (2, 2)


def test_create_multimodal_moves_to_device():
    model = create_multimodal_model(device="cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_cnn3d_get_features_is_pre_classifier():
    model = CNN3DRiskDetector(model_name="r3d_18", pretrained=False).eval()
    x = torch.randn(1, 3, 4, 112, 112)
    with torch.no_grad():
        feats = model.get_features(x)
        logits = model(x)
    assert feats.shape == (1, model.feature_dim)
    assert logits.shape == (1, model.num_classes)
    assert not torch.allclose(feats[:, : model.num_classes], logits)
