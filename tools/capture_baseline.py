#!/usr/bin/env python3
"""Capture a numerical baseline for the smallest MAEST model before refactoring.

This script is the gate for the ADOPT campaign (feat/adopt-constitution): every
later phase (timm dependency changes, file-split, packaging) must reproduce the
melspectrogram, embeddings, and predict_labels output captured here bit-for-bit
(or document a floor if truly nondeterministic). Synthesizes deterministic
seeded test audio (chirp + tones at the model's expected 16kHz mono input),
runs discogs-maest-5s-pw-129e (the smallest/fastest variant, 5s input) end to
end, and saves melspectrogram + embeddings + top-20 label probabilities to
tests/fixtures/baseline_<arch>.npz. Run twice (`--verify-run`) to confirm
determinism before trusting the fixture.

Reads: maest_infer.get_maest, maest_infer.maest (MelSpectrogram via MAEST.forward)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ARCH = "discogs-maest-5s-pw-129e"
SAMPLE_RATE = 16000  # MelSpectrogram.sr in helpers/melspectrogram.py
DURATION_S = 5
SEED = 20260711  # today's date (2026-07-11), arbitrary but fixed
TOP_K = 20

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def _synth_clip(seed: int, duration_s: float = DURATION_S, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Deterministically synthesize one test clip: chirp + two tones + noise floor.

    Not real music -- just a fixed, reproducible signal exercising a range of
    frequencies so the mel-spectrogram and transformer see non-trivial input.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    t = np.arange(n) / sr

    # Linear chirp sweeping from 100 Hz to 4000 Hz over the clip.
    f0, f1 = 100.0, 4000.0
    k = (f1 - f0) / duration_s
    chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * k * t**2))

    # Two fixed tones (A4=440Hz and a higher harmonic-ish partial).
    tone1 = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    tone2 = 0.3 * np.sin(2 * np.pi * 1320.0 * t)

    # Small amount of deterministic noise via the seeded Generator.
    noise = 0.02 * rng.standard_normal(n)

    signal = chirp + tone1 + tone2 + noise
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.9
    return signal.astype(np.float32)


def _make_clips() -> list[np.ndarray]:
    """Three deterministic clips derived from fixed, distinct seeds."""
    return [_synth_clip(SEED + i) for i in range(3)]


def _run_once(model, clips: list[np.ndarray]) -> dict:
    """Run the model over every clip, capturing mel/embeddings/top-k labels."""
    out = {}
    for i, clip_np in enumerate(clips):
        waveform = torch.from_numpy(clip_np)

        if model.melspectrogram is None:
            model.init_melspectrogram()

        with torch.no_grad():
            mel = model.melspectrogram(waveform)
            logits, embeddings = model(waveform)
            activations, labels = model.predict_labels(waveform)

        top_idx = np.argsort(activations)[::-1][:TOP_K]
        top_labels = [labels[j] for j in top_idx]
        top_probs = activations[top_idx].astype(np.float64)

        out[f"clip{i}_mel"] = mel.detach().cpu().numpy()
        out[f"clip{i}_embeddings"] = embeddings.detach().cpu().numpy()
        out[f"clip{i}_logits"] = logits.detach().cpu().numpy()
        out[f"clip{i}_top_labels"] = np.array(top_labels)
        out[f"clip{i}_top_probs"] = top_probs
    return out


def _load_model():
    from maest_infer import get_maest

    torch.manual_seed(SEED)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass  # not all ops on all builds support strict determinism; documented in report

    model = get_maest(arch=ARCH, pretrained=True)
    model.eval()
    return model


def capture(out_path: Path) -> dict:
    model = _load_model()
    clips = _make_clips()
    result = _run_once(model, clips)

    meta = {
        "arch": ARCH,
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "seed": SEED,
        "top_k": TOP_K,
        "n_clips": len(clips),
        "torch_version": torch.__version__,
    }
    result["_meta"] = np.array(json.dumps(meta))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **result)
    return result


def compare(a: dict, b: dict) -> list[str]:
    """Return a list of mismatch descriptions; empty list means bit-identical."""
    mismatches = []
    keys = sorted(k for k in a if not k.startswith("_"))
    for k in keys:
        va, vb = a[k], b[k]
        if va.dtype.kind in "SU":  # string arrays (labels)
            if not np.array_equal(va, vb):
                mismatches.append(f"{k}: label mismatch")
            continue
        if not np.array_equal(va, vb):
            max_abs_diff = np.max(np.abs(va.astype(np.float64) - vb.astype(np.float64)))
            mismatches.append(f"{k}: NOT bit-identical (max_abs_diff={max_abs_diff:.3e})")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-run",
        action="store_true",
        help="Run capture twice in this process and diff, to check determinism.",
    )
    args = parser.parse_args()

    out_path = FIXTURES_DIR / f"baseline_{ARCH}.npz"

    print(f"[capture_baseline] arch={ARCH} seed={SEED} -> {out_path}")
    run1 = capture(out_path)

    if args.verify_run:
        print("[capture_baseline] running a second, independent capture pass to verify determinism...")
        tmp_path = FIXTURES_DIR / f"_verify_baseline_{ARCH}.npz"
        run2 = capture(tmp_path)
        mismatches = compare(run1, run2)
        tmp_path.unlink(missing_ok=True)
        if mismatches:
            print("[capture_baseline] DETERMINISM CHECK FAILED:")
            for m in mismatches:
                print(f"  - {m}")
            return 1
        print("[capture_baseline] determinism check PASSED: two independent runs are bit-identical.")

    print(f"[capture_baseline] saved baseline fixture: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
