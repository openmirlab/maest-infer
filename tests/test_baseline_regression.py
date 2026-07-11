"""Regression test: reproduce the P0 baseline fixture bit-for-bit.

Re-runs `tools.capture_baseline`'s exact clip synthesis + model forward pass
and diffs every array against `tests/fixtures/baseline_<arch>.npz`. This is
the gate the whole ADOPT campaign (feat/adopt-constitution) built against:
P1 (timm removal), P3 (checkpoint-integrity wiring), and P4 (the maest.py
file split) each re-ran this comparison and got zero mismatches before being
committed. Tolerance is exact (rtol=atol=0) rather than "close enough" --
tools/capture_baseline.py separately verified this model is deterministic
both within-process and across separate process invocations, so anything
short of bit-identical here is a real regression, not measurement noise.

Requires the discogs-maest-5s-pw-129e checkpoint (downloaded once via
torch.hub and cached thereafter, same as any normal `get_maest(...,
pretrained=True)` call) -- not marked `network` since it doesn't probe
upstream liveness, it just needs the artifact once like any inference call.

Reads: tools.capture_baseline, tests/fixtures/baseline_discogs-maest-5s-pw-129e.npz
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.capture_baseline import ARCH, _load_model, _make_clips, _run_once

FIXTURE_PATH = Path(__file__).parent / "fixtures" / f"baseline_{ARCH}.npz"


def test_fixture_exists():
    assert FIXTURE_PATH.is_file(), (
        f"Baseline fixture missing: {FIXTURE_PATH}. Run "
        "`python tools/capture_baseline.py` to (re)generate it."
    )


def test_rerun_matches_baseline_bit_for_bit():
    expected = np.load(FIXTURE_PATH, allow_pickle=True)

    model = _load_model()
    clips = _make_clips()
    actual = _run_once(model, clips)

    keys = sorted(k for k in expected.files if not k.startswith("_"))
    assert keys, "baseline fixture has no data keys"

    mismatches = []
    for key in keys:
        exp_val, act_val = expected[key], actual[key]
        if exp_val.dtype.kind in "SU":  # label strings
            if not np.array_equal(exp_val, act_val):
                mismatches.append(f"{key}: label mismatch")
            continue
        if not np.array_equal(exp_val, act_val):
            max_abs_diff = np.max(
                np.abs(exp_val.astype(np.float64) - act_val.astype(np.float64))
            )
            mismatches.append(f"{key}: NOT bit-identical (max_abs_diff={max_abs_diff:.3e})")

    assert not mismatches, "Baseline regression detected:\n" + "\n".join(mismatches)
