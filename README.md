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
against a recorded SHA-256 checksum (`src/maest_infer/data/checkpoints.json`);
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

This package is licensed under [AGPL-3.0-only](LICENSE), following the original MAEST license. See [NOTICE](NOTICE) for the full third-party attribution and weights-licensing breakdown.

---

## Support

For bugs and feature requests, please open an issue on
[GitHub](https://github.com/openmirlab/maest-infer/issues).

---

## Related Projects

- [MAEST](https://github.com/palonso/maest) - Original research repository with training code
- [PaSST](https://github.com/kkoutini/PaSST) - Patchout faSt Spectrogram Transformer (base architecture)
- [Essentia](https://essentia.upf.edu/models.html#maest) - MAEST models in Essentia
