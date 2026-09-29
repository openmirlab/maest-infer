"""Offline device, lifecycle, and TOML resolver contract tests."""

import hashlib

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


def test_invalid_device_fails_before_model_construction_and_cleans_up(monkeypatch):
    import maest_infer.loading as loading

    built = []

    def build(*args, **kwargs):
        built.append((args, kwargs))
        return _Model()

    monkeypatch.setattr(loading, "get_maest", build)
    session = MAESTSession(device="definitely-not-a-device")

    with pytest.raises(ValueError, match="device"):
        session.load()

    assert built == []
    assert session.status == "failed"
    assert session._model is None


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
    assert _resolve_device("cuda") == torch.device("cuda:0")
    assert _resolve_device("cuda:1") == torch.device("cuda:1")
    with pytest.raises(RuntimeError, match="index 2"):
        _resolve_device("cuda:2")


def test_resolve_device_rejects_mps_outright(monkeypatch):
    """mps is out of scope (org canon art. 4b) regardless of availability."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="mps"):
        _resolve_device("mps")


def test_resolve_device_rejects_mps_when_unavailable_too(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="mps"):
        _resolve_device("mps")


def test_resolve_device_auto_never_resolves_to_mps_even_if_available(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert _resolve_device("auto") == torch.device("cpu")
    assert _resolve_device(None) == torch.device("cpu")


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


def test_loaded_checkpoint_info_hashes_actual_url_and_custom_bytes(monkeypatch, tmp_path):
    import maest_infer.loading as loading

    checkpoint = tmp_path / "downloaded.ckpt"
    checkpoint.write_bytes(b"resolved artifact")
    digest = hashlib.sha256(b"resolved artifact").hexdigest()

    def build(*args, **kwargs):
        model = _Model()
        model._maest_loaded_checkpoint = {
            "path": str(checkpoint),
            "url": "https://example.test/model.ckpt",
            "checkpoint_id": "model.ckpt",
        }
        return model

    monkeypatch.setattr(loading, "get_maest", build)
    session = MAESTSession(
        checkpoint_url="https://example.test/model.ckpt",
        checkpoint_sha256="0" * 64,
        device="cpu",
    ).load()

    loaded_info = session.loaded_checkpoint_info()
    assert loaded_info == {
        "model": "discogs-maest-5s-pw-129e",
        "model_arch": "discogs-maest-5s-pw-129e",
        "device": "cpu",
        "checkpoint_id": "model.ckpt",
        "checkpoint_path": str(checkpoint),
        "checkpoint_url": "https://example.test/model.ckpt",
        "checkpoint_sha256": digest,
    }
    checkpoint.write_bytes(b"replaced after load")
    assert session.loaded_checkpoint_info() == loaded_info

    custom = tmp_path / "custom.ckpt"
    custom.write_bytes(b"custom artifact")
    custom_digest = hashlib.sha256(b"custom artifact").hexdigest()

    def build_custom(*args, **kwargs):
        model = _Model()
        model._maest_loaded_checkpoint = {
            "path": str(custom),
            "url": None,
            "checkpoint_id": "custom.ckpt",
        }
        return model

    monkeypatch.setattr(loading, "get_maest", build_custom)
    custom_session = MAESTSession(checkpoint=custom, device="cpu").load()
    info = custom_session.loaded_checkpoint_info()
    assert info["checkpoint_path"] == str(custom)
    assert info["checkpoint_url"] is None
    assert info["checkpoint_sha256"] == custom_digest
    assert info["checkpoint_sha256"] != session.cache_info()["checkpoint_sha256"]


def test_loaded_checkpoint_info_requires_loader_provenance():
    session = MAESTSession(model=_Model(), checkpoint="/tmp/looks-real.ckpt", device="cpu")

    with pytest.raises(RuntimeError, match="provenance"):
        session.loaded_checkpoint_info()


def test_loaded_checkpoint_info_requires_ready_session():
    with pytest.raises(RuntimeError, match="ready"):
        MAESTSession().loaded_checkpoint_info()


def test_toml_drives_default_cfg_and_integrity_lookup():
    from maest_infer.checkpoints import checkpoint_artifact, checkpoint_for_url
    from maest_infer.configs import default_cfgs

    artifact = checkpoint_artifact("discogs_maest_5s_pw_129e")
    assert default_cfgs["discogs_maest_5s_pw_129e"]["url"] == artifact["url"]
    assert checkpoint_for_url(artifact["url"])["sha256"] == artifact["sha256"]


def test_get_maest_accepts_external_default_cfg(monkeypatch):
    import maest_infer.loading as loading

    received = {}

    def fake_factory(pretrained=False, **kwargs):
        received["pretrained"] = pretrained
        received["kwargs"] = kwargs
        return _Model()

    monkeypatch.setitem(loading._FACTORY_FUNCTIONS, "discogs_maest_5s_pw_129e", fake_factory)
    model = loading.get_maest(
        "discogs-maest-5s-pw-129e",
        external_default_cfg={"url": "https://example.test/custom.ckpt", "checkpoint_sha256": "1" * 64},
    )

    assert isinstance(model, _Model)
    assert received["pretrained"] is True
    assert received["kwargs"]["external_default_cfg"] == {
        "url": "https://example.test/custom.ckpt",
        "checkpoint_sha256": "1" * 64,
    }


def test_custom_cache_metadata_does_not_claim_default_source(tmp_path):
    path = tmp_path / "custom.ckpt"
    path.write_bytes(b"custom")
    info = MAESTSession(checkpoint=path, device="cpu").cache_info()
    assert info["checkpoint_url"] is None
    assert info["checkpoint_sha256"] is None
    info = MAESTSession(checkpoint_url="https://example.test/custom.ckpt", device="cpu").cache_info()
    assert info["checkpoint_sha256"] is None
