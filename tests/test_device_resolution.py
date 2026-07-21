"""Regression tests for the automatic MAEST device sentinel.

``None`` and ``"auto"`` use the same legacy automatic selection. Explicit
validation and CUDA-index coverage live in ``test_session_contract.py``.

Reads: maest_infer.clean_api.MAESTSession, maest_infer.loading.get_maest
"""

import torch

import maest_infer.loading as loading
from maest_infer.clean_api import MAESTSession


class _FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self


def test_device_auto_resolves_like_none(monkeypatch):
    monkeypatch.setattr(loading, "get_maest", lambda *a, **kw: _FakeModel())

    expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    auto_session = MAESTSession(device="auto").load()
    none_session = MAESTSession(device=None).load()

    assert auto_session.device == expected
    assert none_session.device == expected


def test_device_explicit_literal_still_passes_through(monkeypatch):
    monkeypatch.setattr(loading, "get_maest", lambda *a, **kw: _FakeModel())

    session = MAESTSession(device="cpu").load()

    assert session.device == torch.device("cpu")
