# maest-infer

[![PyPI](https://img.shields.io/pypi/v/maest-infer)](https://pypi.org/project/maest-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

Inference-only package for [MAEST](https://github.com/palonso/maest) (Music Audio Efficient Spectrogram Transformer).

---

## Why this exists

[palonso/maest](https://github.com/palonso/maest) is the original,
actively-maintained research codebase for MAEST — but it's built for
research, not for dropping into an inference pipeline. It ships a full
Sacred experiment framework, TensorBoard logging, pre-training and
fine-tuning configurations, and (until this package removed the equivalent
dependency) a `timm` dependency pulled in for a single checkpoint-loading
function.

**maest-infer** is a lightweight, dependency-minimal repackaging focused
solely on inference: no training code, no Sacred, no TensorBoard, and no
`timm` (the one code path MAEST's inference actually exercises is vendored
directly — see [CLAUDE.md](CLAUDE.md)). It wraps the 10 pretrained
checkpoint variants behind a single `get_maest(arch=...)` entry point, and
has been verified bit-identical to upstream (see
[Verification](#verification) below).

For training, fine-tuning, and the full research codebase, use the
[original MAEST repository](https://github.com/palonso/maest) directly.

## Acknowledgments

This package is a repackaging of **MAEST** (Music Audio Efficient Spectrogram Transformer) created by [Pablo Alonso-Jimenez](https://github.com/palonso) and colleagues at the [Music Technology Group (MTG)](https://www.upf.edu/web/mtg), Universitat Pompeu Fabra. We are grateful to the original authors for making their research and pretrained models publicly available.

- **Original Repository**: [https://github.com/palonso/maest](https://github.com/palonso/maest)
- **Original Authors**: Pablo Alonso-Jiménez, Xavier Serra, Dmitry Bogdanov (MTG, Universitat Pompeu Fabra)
- **Hugging Face Models**: [https://huggingface.co/mtg-upf](https://huggingface.co/mtg-upf)
- **Checkpoint host**: 8 MAEST checkpoints are served from
  [GitHub Releases on palonso/MAEST](https://github.com/palonso/MAEST/releases);
  this package also downloads one [PaSST](https://github.com/kkoutini/PaSST)
  checkpoint (Khaled Koutini et al.) and one DeiT checkpoint (Meta AI
  Research, `dl.fbaipublicfiles.com`) that upstream MAEST itself depends on
  for initialization. See [NOTICE](NOTICE) for the full breakdown.

## Citation

If you use MAEST in your research, **please cite the original paper**:

```bibtex
@inproceedings{alonso2023efficient,
    title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
    author={Alonso-Jim{\'e}nez, Pablo and Serra, Xavier and Bogdanov, Dmitry},
    booktitle={Proceedings of the 24th International Society for Music Information Retrieval Conference (ISMIR)},
    year={2023},
}
```

---

## Features

- **Dependency-minimal**: no `timm`, no Essentia, no Sacred/TensorBoard —
  just `torch`, `torchaudio`, and `numpy`.
- **10 pretrained checkpoint variants**: 5s/10s/20s/30s input lengths, 400-
  or 519-label Discogs genre taxonomies, multiple init strategies (PaSST
  weights, DeiT weights, from-scratch, teacher-student).
- **Single entry point**: `get_maest(arch=...)` behind which all
  checkpoint-specific loading/adaptation logic lives.
- **Checkpoint integrity checking**: every download is verified against a
  recorded SHA-256 checksum.
- **Bit-identical to upstream**: verified against the original palonso/MAEST
  implementation — see [Verification](#verification).

## Scope

**In scope:**

- Loading any of the 10 documented pretrained checkpoint variants and
  running inference (`model(audio)` → logits + embeddings,
  `model.predict_labels(audio)` → labelled activations).
- Automatic, SHA-256-verified checkpoint download and caching.

**Out of scope, forever:**

- **Training / fine-tuning** — no Sacred configs, no training loop. Use
  [palonso/maest](https://github.com/palonso/maest) for that.
- **Essentia integration** — this package's mel front end is torchaudio
  only; see [Mel-spectrogram fidelity](#mel-spectrogram-fidelity) below for
  how that compares to upstream's Essentia-based training pipeline.

**License constraint on hosting this package (read before deploying it
behind any network service):** maest-infer is a derivative of AGPL-3.0-only
upstream code (see [License](#license) below) and the pretrained weights are
CC BY-NC-SA 4.0, non-commercial only. Both constraints follow the package
wherever it runs, including inside a hosted API/provider.

**Present in code but not fully supported:**

- The `passt_deit_bd_p16_384` architecture string in `get_maest`'s dispatch
  table raises `RuntimeError` on construction (a patch-embed reshape shape
  mismatch). This is an inherited upstream issue, reproduced identically on
  the pre-refactor code — not a regression introduced by this package, and
  not currently fixed. See [CLAUDE.md](CLAUDE.md) for detail.

---

## Install

```bash
# From PyPI
pip install maest-infer

# Or with uv
uv pip install maest-infer
```

For development:
```bash
git clone https://github.com/openmirlab/maest-infer.git
cd maest-infer
pip install -e .
```

## Quick Start

```python
import torch
from maest_infer import get_maest

# Load model (downloads pretrained weights automatically)
model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
model.eval()

# Inference with raw 16kHz audio
audio = torch.randn(16000 * 30)  # 30 seconds
logits, embeddings = model(audio)
# logits: (1, 519), embeddings: (1, 768)

# Predict with labels
activations, labels = model.predict_labels(audio)
```

### Explicit lifecycle

Use `MAESTSession` when a caller needs deterministic model loading and release:

```python
from maest_infer import MAESTSession

with MAESTSession(arch="discogs-maest-5s-pw-129e", device="cpu") as session:
    logits, embeddings = session.infer(audio)
```

Checkpoint metadata is package-owned in `config/checkpoints.toml`; callers may
override a checkpoint path, URL, or SHA-256 without changing the package.
`load()` is idempotent, `infer()` requires a ready session, `release()` allows
reload while retaining cached upstream torch-hub files, and `close()` is
terminal. Devices preserve legacy `None`/`auto` selection and accept explicit
`cpu`, `cuda`, or `cuda:N`; unavailable or invalid explicit requests raise
before model construction. `mps` is not supported -- Apple MLX/MPS backends
are permanently out of scope for this org's projects (org canon
openmirlab-dev 5e588e6, art. 4b).

`cache_info()` is read-only and never downloads weights. After `load()`,
`loaded_checkpoint_info()` reports the actual resolved checkpoint path, SHA-256
computed from those bytes, checkpoint id, model arch, and concrete device.
Custom local checkpoints report their own hash rather than any packaged default
hash.

### Experimental genre analysis

The additive task API returns ranked Discogs styles for a selected segment, a
whole track, or a time curve. It currently requires
`discogs-maest-30s-pw-129e-519l`; the legacy `MAESTSession` default remains
unchanged. Existing `get_maest`, `infer`, and `predict_labels` behavior is preserved.
This branch API has not been released or certified for hosted service use.

```python
import numpy as np
from maest_infer import MAESTSession, genre_metadata, preview_classification

# Replace with decoded mono floating-point audio and its actual sample rate.
audio = np.zeros(16000 * 60, dtype=np.float32)
metadata = genre_metadata()
preview = preview_classification(sample_count=audio.shape[0], window_seconds=30)

with MAESTSession(arch="discogs-maest-30s-pw-129e-519l", device="cpu") as session:
    summary = session.classify(audio, sample_rate=16000)
    opening = session.classify(audio, sample_rate=16000, mode="segment")
    later = session.classify(
        audio, sample_rate=16000, mode="segment", start_seconds=30,
    )
    curve = session.classify(
        audio, sample_rate=16000, mode="time_curve", hop_seconds=5,
        curve_labels=["Electronic---House"],
    )
    print(summary["rankings"])
```

The silence above demonstrates the input contract, not a meaningful genre
example. Decode real music in the caller; this API accepts only mono floating
NumPy arrays or torch tensors, with an explicit positive integer sample rate.
It resamples to 16 kHz, checks finite samples, and does not silently downmix
stereo or normalize integer PCM. File paths are not accepted.
`classify_genre(audio, sample_rate=..., device="cpu", ...)` is the one-shot
alternative: it creates and closes a fresh session each call. Reuse an explicit
session for multiple requests; it is not safe for concurrent classify/release calls.
`genre_metadata()` and `preview_classification(...)` do not construct a model,
download weights, activate CUDA, or require a waveform. Preview validates the
same mode/window/label/batch options as `classify`; pass a known duration as
16 kHz `sample_count` to get the same coverage and window-count analysis that
classification will use. If `sample_count` is omitted, `analysis` is `None`
because duration-dependent checks are unknown.

| Parameter | Contract |
|---|---|
| `mode` | `full_track` (default), `segment`, or `time_curve` |
| `window_seconds` | Integer 5–30, default 30; shorter contexts may alter predictions |
| `hop_seconds` | Integer 1–window, default window; segment requires explicit `end_seconds` |
| `start_seconds` | Segment or time curve, default 0; finite nonnegative time inside the audio |
| `end_seconds` | Segment or time curve; exclusive end, greater than start, clipped at EOF |
| `top_n` | Integer 1–50, default 10; number of ranked candidates |
| `curve_labels` | Time curve only, unique exact labels, up to 50; omit for automatic selection |
| `batch_size` | Runtime tuning for bounded inference batches; default 1, range 1–32 |

`full_track` covers the whole input and rejects range bounds. `segment` without
`end_seconds` preserves the single-window behavior. With `end_seconds`, it averages
all windows covering that range, even if it exceeds 30 seconds. `time_curve` accepts
the same bounds; an omitted end continues to EOF. Curve timestamps and analysis
bounds remain absolute in the original audio. `window_seconds` is the model context,
not a limit on range duration. For example, analyze an entire 39-second chorus:

```python
chorus = session.classify(
    audio, sample_rate=16000, mode="segment",
    start_seconds=42, end_seconds=81, window_seconds=30,
)
```

Multi-window ranges use an end-aligned final window, so the last step may be shorter
than hop. Short ranges are right-padded without reading past the requested end;
padding contributes no aggregation weight. End values beyond EOF clip to EOF;
start must precede the effective end by at least one sample.
A request is rejected if its planned window count exceeds 4096. Hosted input
limits and artifact delivery are the calling service's responsibility.

All modes return JSON-compatible `rankings`, ordered by descending `score`,
then ascending `label_id` on exact ties, with consecutive one-based `rank`.
Scores are sigmoid activations, not calibrated probabilities or a distribution
summing to one. These scores do not measure energy, loudness, mood, instrument
presence, arrangement changes, or musical section boundaries; those claims require
separate measurements or listening evidence. Each row also includes `genre` and `subgenre`; parent genre
scores are not synthesized by summing children. Aggregation uses all 519 label
scores before selecting Top-N. Each instant averages its covering windows, then
the track averages those values over real duration (`coverage_weighted_mean_v1`).
`analysis` records resolved settings, coverage, and window count.

`time_curve` additionally returns a `timeline` payload containing time-ordered
`windows` and dense `series`, with one score per window for every selected label.
Automatic selection considers the union of each window's Top-N and picks up to
Top-N labels by peak score; explicit `curve_labels` preserves the requested order.
Global rankings still
use time-weighted mean, not peaks. Curves describe window-sized contexts;
a 5-second hop does not imply precise 5-second genre-change localization.
No interpolation, smoothing, silence removal, or partial-success fallback is applied.

The existing ready-only lifecycle also applies to `classify`: load first;
release permits reload; close is terminal. Numerical and runtime probe evidence
is recorded under `docs/blueprints/`; classification accuracy is not established
by successful execution alone.

## Available Models

| Model | Input Length | Labels | Description |
|-------|--------------|--------|-------------|
| `discogs-maest-5s-pw-129e` | 5 sec | 400 | PaSST weights |
| `discogs-maest-10s-fs-129e` | 10 sec | 400 | From scratch |
| `discogs-maest-10s-pw-129e` | 10 sec | 400 | PaSST weights |
| `discogs-maest-10s-dw-75e` | 10 sec | 400 | DeiT weights |
| `discogs-maest-20s-pw-129e` | 20 sec | 400 | PaSST weights |
| `discogs-maest-30s-pw-129e` | 30 sec | 400 | PaSST weights |
| `discogs-maest-30s-pw-73e-ts` | 30 sec | 400 | Teacher-student |
| `discogs-maest-30s-pw-129e-519l` | 30 sec | 519 | Extended labels |

## Verification

- **Bit-identical to upstream MAEST**: mel/embeddings/logits max\|delta\|=0.0
  against the original palonso/MAEST implementation (verified 2026-07,
  matched torch 2.9.1).
- A captured baseline fixture
  (`tests/fixtures/baseline_discogs-maest-5s-pw-129e.npz`) gates every
  structural change to this repo — the `timm` removal, checkpoint-integrity
  wiring, and the module split all re-ran it and diffed bit-identical.

### Mel-spectrogram fidelity

This package's inference mel front end uses torchaudio — matching upstream
MAEST's own shipped inference design (Essentia was only ever used by
upstream to build the *training* dataset, never at inference). Empirically
bounded (2026-07, 4 clips including real music,
`discogs-maest-10s-pw-129e`): final-embedding cosine similarity ≥0.999 and
100% top-5 label agreement vs Essentia-derived features, though only
~74-97% of individual mel bins meet the stricter per-bin rtol/atol=1e-3
claim documented in `helpers/melspectrogram.py`'s docstring (mismatch
concentrates in the lowest mel bands and edge frames).

---

## What this project will NEVER bundle

None of the pretrained checkpoints are committed to this repository or
bundled in the PyPI package. All model checkpoints are hosted on
[GitHub Releases](https://github.com/palonso/MAEST/releases) (plus one
PaSST and one DeiT checkpoint from their respective upstream hosts — see
[Acknowledgments](#acknowledgments)) and downloaded automatically on first
use, cached in `~/.cache/torch/hub/checkpoints/`. Each download is verified
against a recorded SHA-256 checksum in packaged `config/checkpoints.toml`;
a corrupted or tampered file raises an error instead of silently loading.
This is a permanent constraint, not a temporary limitation — keeping
multi-hundred-megabyte weights out of the repo and the wheel will not
change.

---

## Development

```bash
uv sync                                             # install/update the environment
uv run --with pytest python -m pytest -q            # unit tests (network tests deselected)
uv run --with pytest python -m pytest -m network -q # + live checkpoint-URL liveness check
uv run python tools/capture_baseline.py --verify-run   # re-capture baseline + determinism check
uv run python tools/check_weights_liveness.py          # HEAD every checkpoint URL
```

See [CLAUDE.md](CLAUDE.md) for the module layout (post-refactor file split),
the file-header convention, and what was deliberately left untouched.

---

## License

This package is licensed under [AGPL-3.0-only](LICENSE), following the
original MAEST license. It is a derivative work of
[palonso/maest](https://github.com/palonso/maest) (verified
`AGPL-3.0` via `gh api repos/palonso/maest/license`, 2026-09-14) — the model
architecture, checkpoint-loading logic, and mel front end are a direct,
inference-only port of that code, so this package cannot be relicensed away
from AGPL-3.0-only. See [NOTICE](NOTICE) for the full third-party
attribution and weights-licensing breakdown.

**Weights license**: the 8 MAEST checkpoints are licensed **CC BY-NC-SA
4.0**, non-commercial and share-alike — confirmed directly on MTG's own
model page ("All the models created by the MTG are licensed under [CC BY-NC-SA
4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)",
https://essentia.upf.edu/models.html, which lists the `discogs-maest-*`
models; the PyTorch `.ckpt` files this package downloads from
[palonso/MAEST releases](https://github.com/palonso/MAEST/releases) are the
same underlying weights in a different export format — same project, same
license terms, cross-referenced but not independently re-confirmed on the
release page itself, (uncertain) only on that last link). The 2 backbone
init checkpoints (PaSST, DeiT) are **Apache-2.0**, from their own upstream
repos (`gh api repos/kkoutini/PaSST/license`, `gh api
repos/facebookresearch/deit --jq .license`, both verified 2026-09-14) — more
permissive than the package's own AGPL-3.0, no extra constraint. openmirlab
is non-commercial, so the NC weights are usable here, but any downstream
consumer of a hosted endpoint (see below) is not automatically covered.

**Hosted / network use (AGPL-3.0 §13, "Remote Network Interaction")**: if
this package is run inside a network-facing service (e.g. as a phonon HTTP
provider behind an MCP/API gateway) and users interact with it remotely,
§13 requires the operator to give those users a way to obtain the
Corresponding Source of the exact modified version being run — normally by
prominently offering it through the service itself (a source-code link plus
the running commit/version surfaced in the provider's own response or
status metadata), not merely by having the code exist on GitHub. This
package's own source is already public
(https://github.com/openmirlab/maest-infer), so the obligation is
satisfiable, but the *offer* has to be made by whatever wraps this package
as a network service — that wrapper is outside this repo and is not audited
here. Whether the §13 boundary extends to a calling process that merely
`import`s this package (rather than modifying it) is a genuinely contested
question under AGPL and is not resolved here (uncertain) — treat any
network-facing wrapper as in-scope until an org decision says otherwise.
This is not legal advice.

---

## Support

For bugs and feature requests, please open an issue on
[GitHub](https://github.com/openmirlab/maest-infer/issues).

---

## Related Projects

- [MAEST](https://github.com/palonso/maest) - Original research repository with training code
- [PaSST](https://github.com/kkoutini/PaSST) - Patchout faSt Spectrogram Transformer (base architecture)
- [Essentia](https://essentia.upf.edu/models.html#maest) - MAEST models in Essentia
