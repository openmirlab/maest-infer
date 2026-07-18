"""Regression test for the "auto" device-string sentinel (clean_api.py:34).

`device=None` (unset) already auto-detects via `self.device or (...)`; the
literal string "auto" is truthy, so it used to bypass that fallback and go
straight into `torch.device("auto")`, which raises `RuntimeError`. This
exercises the real `MAESTSession.load()` path -- `get_maest` monkeypatched
so no real checkpoint/network is needed -- and asserts "auto" resolves
identically to `None`, while an explicit literal like "cpu" still passes
through unchanged.

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
