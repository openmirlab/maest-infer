"""MAEST inference-only package for music audio classification.

Public entry point: `get_maest(arch=...)` builds one of 10 pretrained
variants; `MAEST` is the underlying nn.Module class. See README.md for the
model table and CLAUDE.md for the internal module layout. `MAESTSession` adds
an explicit lifecycle facade, and `classify_genre` is an additive experimental
519-label task API.

Reads: maest_infer.maest (re-export shim), maest_infer.clean_api,
maest_infer.__about__ (version)
"""

from .__about__ import __version__
from .maest import get_maest, MAEST
from .clean_api import MAESTSession, classify_genre, get_maest_session

__all__ = [
    "get_maest",
    "MAEST",
    "MAESTSession",
    "get_maest_session",
    "classify_genre",
    "__version__",
]
