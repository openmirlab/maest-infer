"""Offline device, lifecycle, and TOML resolver contract tests."""

import pytest
import torch

from maest_infer.clean_api import MAESTSession, _resolve_device


class _Model:
    def __init__(self):
        self.devices = []
        self.released = False

    def to(self, device):
        self.devices.append(device)
        return self

    def eval(self):
        return self

    def cpu(self):
        self.released = True
        return self

    def __call__(self, audio, **kwargs):
        return audio


def test_session_reuses_then_rebuilds_and_close_is_terminal(monkeypatch):
    import maest_infer.loading as loading

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    built = []

    def build(*args, **kwargs):
        model = _Model()
        built.append((kwargs, model))
        return model

    monkeypatch.setattr(loading, "get_maest", build)
    session = MAESTSession(device="cuda:1")

    with pytest.raises(RuntimeError, match="call load"):
        session.infer("audio")
    assert session.load() is session
    assert session.load() is session
    assert session.infer("audio") == "audio"
    assert len(built) == 1
    assert built[0][1].devices == [torch.device("cuda:1")]

    session.release()
    assert session.status == "released"
    assert built[0][1].released
    session.load()
    assert len(built) == 2

    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="must be ready"):
        session.infer("audio")


def test_failed_load_is_visible(monkeypatch):
    import maest_infer.loading as loading

    monkeypatch.setattr(loading, "get_maest", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")))
    session = MAESTSession()
    with pytest.raises(ValueError, match="bad"):
        session.load()
    assert session.status == "failed"


def test_device_validation_auto_and_cuda_index(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _resolve_device(None) == torch.device("cpu")
    assert _resolve_device("auto") == torch.device("cpu")
    with pytest.raises(RuntimeError, match="CUDA"):
        _resolve_device("cuda")
    with pytest.raises(ValueError):
        _resolve_device("cuda:-1")
    with pytest.raises(ValueError):
        _resolve_device("metal")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="index 2"):
        _resolve_device("cuda:2")


def test_cache_info_uses_torch_hub_resolver_for_default_and_custom_paths(monkeypatch, tmp_path):
    import maest_infer.clean_api as api

    resolved = tmp_path / "torch-hub.ckpt"
    monkeypatch.setattr(api, "torch_hub_checkpoint_path", lambda url: resolved)
    session = MAESTSession()
    assert session.cache_info()["checkpoint_path"] == str(resolved)
    assert session.cache_info()["cached"] is False
    resolved.write_bytes(b"cached")
    assert session.cache_info()["cached"] is True

    custom = tmp_path / "private.ckpt"
    custom.write_bytes(b"cached")
    custom_session = MAESTSession(checkpoint=custom)
    assert custom_session.cache_info()["checkpoint_path"] == str(custom)
    assert custom_session.cache_info()["cached"] is True


def test_toml_drives_default_cfg_and_integrity_lookup():
    from maest_infer.checkpoints import checkpoint_artifact, checkpoint_for_url
    from maest_infer.configs import default_cfgs

    artifact = checkpoint_artifact("discogs_maest_5s_pw_129e")
    assert default_cfgs["discogs_maest_5s_pw_129e"]["url"] == artifact["url"]
    assert checkpoint_for_url(artifact["url"])["sha256"] == artifact["sha256"]
