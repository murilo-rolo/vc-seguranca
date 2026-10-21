"""Testes do sistema de configuração de treino (TOML) e do train_loop."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.config import DEFAULTS, load_config, resolve_config
from src.training.loop import train_loop


def test_load_config_uses_project_default_file():
    # Usa configs/treino.toml do repositório (espelha os DEFAULTS).
    cfg = load_config("cnn3d")
    assert cfg["model_name"] == "r2plus1d_18"
    assert set(DEFAULTS["cnn3d"]) <= set(cfg)


def test_load_config_merges_toml(tmp_path):
    cfg_file = tmp_path / "treino.toml"
    cfg_file.write_text("[cnn3d]\nepochs = 3\nbatch_size = 4\n")
    cfg = load_config("cnn3d", str(cfg_file))
    assert cfg["epochs"] == 3
    assert cfg["batch_size"] == 4
    assert cfg["learning_rate"] == DEFAULTS["cnn3d"]["learning_rate"]  # não tocado


def test_load_config_rejects_unknown_keys(tmp_path):
    cfg_file = tmp_path / "treino.toml"
    cfg_file.write_text("[emotion]\nlerning_rate = 0.1\n")  # typo proposital
    with pytest.raises(ValueError, match="Chaves desconhecidas"):
        load_config("emotion", str(cfg_file))


def test_load_config_missing_explicit_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config("cnn3d", str(tmp_path / "nao_existe.toml"))


def test_resolve_config_cli_overrides():
    args = SimpleNamespace(config=None, device="cpu", epochs=7)
    cfg = resolve_config("multimodal", args)
    assert cfg["device"] == "cpu"
    assert cfg["epochs"] == 7
    # valores não-override vêm dos defaults
    assert cfg["learning_rate"] == DEFAULTS["multimodal"]["learning_rate"]


def test_resolve_config_keeps_epochs_when_not_passed():
    args = SimpleNamespace(config=None, device="cuda", epochs=None)
    cfg = resolve_config("cnn3d", args)
    assert cfg["epochs"] == DEFAULTS["cnn3d"]["epochs"]


# ---------------------------------------------------------------------------
# train_loop com modelo sintético
# ---------------------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)


def _tiny_loader():
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
    inputs = labels.float().view(-1, 1) * 2 - 1
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


def test_train_loop_saves_checkpoint_and_history(tmp_path):
    model = _TinyModel()
    out = tmp_path / "weights"
    exp = tmp_path / "experiments"
    cfg = {"epochs": 2, "batch_size": 4, "learning_rate": 1e-2}

    best_acc, history = train_loop(
        model, _tiny_loader(), _tiny_loader(),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        device=torch.device("cpu"), epochs=2,
        output_dir=out, experiments_dir=exp, config=cfg,
    )

    assert (out / "best_model.pth").exists()
    assert (exp / "training_history.json").exists()

    ckpt = torch.load(out / "best_model.pth", map_location="cpu")
    assert ckpt["hyperparameters"] == cfg
    assert "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt
    assert history["config"] == cfg
    assert len(history["val_acc"]) == 2
    assert best_acc == max(history["val_acc"])
    # sem monitor_loader: nenhuma chave de monitor
    assert "monitor_score" not in history and "val_acc_swap" not in history
    assert "monitor_score" not in ckpt


def _tiny_loader_flipped():
    labels = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1])
    inputs = labels.float().view(-1, 1) * 2 - 1
    return DataLoader(TensorDataset(inputs, labels), batch_size=4)


def test_train_loop_monitor_loader_selects_by_mean(tmp_path):
    model = _TinyModel()
    out = tmp_path / "weights"
    exp = tmp_path / "experiments"

    best, history = train_loop(
        model, _tiny_loader(), _tiny_loader(),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),
        device=torch.device("cpu"), epochs=2,
        output_dir=out, experiments_dir=exp,
        monitor_loader=_tiny_loader_flipped(),
    )

    assert len(history["val_acc_swap"]) == 2
    assert len(history["monitor_score"]) == 2
    for i in range(2):
        expected = (history["val_acc"][i] + history["val_acc_swap"][i]) / 2
        assert history["monitor_score"][i] == pytest.approx(expected)
    assert best == max(history["monitor_score"])

    ckpt = torch.load(out / "best_model.pth", map_location="cpu")
    assert ckpt["monitor_score"] == pytest.approx(best)
    assert "val_acc_swap" in ckpt and "val_acc" in ckpt
    assert (exp / "training_history.json").exists()


def test_train_loop_checkpoint_model_saves_submodule(tmp_path):
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _TinyModel()

        def forward(self, x):
            return self.inner(x)

    wrapper = Wrapper()
    best, _ = train_loop(
        wrapper, _tiny_loader(), _tiny_loader(),
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(wrapper.parameters(), lr=1e-2),
        device=torch.device("cpu"), epochs=1,
        output_dir=tmp_path / "w", experiments_dir=tmp_path / "e",
        checkpoint_model=wrapper.inner,
    )
    ckpt = torch.load(tmp_path / "w" / "best_model.pth", map_location="cpu")
    assert all(k.startswith("fc.") for k in ckpt["model_state_dict"])
