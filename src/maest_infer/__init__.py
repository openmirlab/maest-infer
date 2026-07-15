"""MAEST inference-only package for music audio classification.

Public entry point: `get_maest(arch=...)` builds one of 10 pretrained
variants; `MAEST` is the underlying nn.Module class. See README.md for the
model table and CLAUDE.md for the internal module layout.

Reads: maest_infer.maest (re-export shim), maest_infer.__about__ (version)
"""

from .__about__ import __version__
from .maest import get_maest, MAEST
from .clean_api import MAESTSession, get_maest_session

__all__ = ["get_maest", "MAEST", "MAESTSession", "get_maest_session", "__version__"]
