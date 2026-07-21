#!/usr/bin/env python3
"""Weights-liveness check -- HEADs every checkpoint URL in checkpoints.toml.

Mirrors bs-roformer-infer's tools/check_weights_liveness.py convention: MAEST's
10 checkpoints are all third-party (github.com/palonso/MAEST releases,
github.com/kkoutini/PaSST releases, dl.fbaipublicfiles.com) with no mirror of
our own, so a dead upstream host breaks `get_maest(..., pretrained=True)` for
users with no earlier warning. This script walks checkpoints.toml and HEADs
each URL. Run manually (`python tools/check_weights_liveness.py`) or via
`pytest -m network tests/test_weights_liveness.py` -- neither runs in the
default test suite since both need real network access.

Reads: maest_infer/config/checkpoints.toml
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Tuple

import requests

TIMEOUT_SECONDS = 15
CHECKPOINTS_TOML = Path(__file__).resolve().parent.parent / "src" / "maest_infer" / "config" / "checkpoints.toml"


def iter_registry_urls() -> Iterator[Tuple[str, str]]:
    """Yield (label, url) for every checkpoint in packaged TOML metadata."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib

    with open(CHECKPOINTS_TOML, "rb") as f:
        data = tomllib.load(f)
    for variant, entry in data.get("models", {}).items():
        for artifact in entry.get("artifacts", []):
            yield variant, artifact["url"]


def check_url(url: str, timeout: float = TIMEOUT_SECONDS) -> Tuple[bool, str]:
    """HEAD a URL, falling back to a ranged GET if the host doesn't support HEAD."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code in (405, 501):
            resp = requests.get(
                url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=timeout
            )
        ok = 200 <= resp.status_code < 400
        return ok, f"HTTP {resp.status_code}"
    except requests.RequestException as exc:
        return False, f"error: {exc}"


def main() -> int:
    failures = []
    for label, url in iter_registry_urls():
        ok, detail = check_url(url)
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}: {url} -> {detail}")
        if not ok:
            failures.append((label, url, detail))

    print()
    if failures:
        print(f"{len(failures)} URL(s) failed liveness check:")
        for label, url, detail in failures:
            print(f"  - {label}: {url} ({detail})")
        return 1

    print("All registry URLs are live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
