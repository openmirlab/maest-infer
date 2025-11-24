# maest-infer

[![PyPI](https://img.shields.io/pypi/v/maest-infer)](https://pypi.org/project/maest-infer/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)

Inference-only package for [MAEST](https://github.com/palonso/maest) (Music Audio Efficient Spectrogram Transformer).

This is a lightweight, dependency-minimal repackaging of MAEST focused solely on inference. For training, fine-tuning, and the full research codebase, please visit the [original MAEST repository](https://github.com/palonso/maest).

---

## Installation

```bash
# From PyPI
pip install maest-infer

# Or with uv
uv pip install maest-infer
```

For development:
```bash
git clone https://github.com/palonso/maest.git
cd maest
pip install -e .
```

---

## Usage

```python
import torch
from maest_infer import get_maest

# Load model (downloads pretrained weights automatically from Hugging Face)
model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
model.eval()

# Inference with raw 16kHz audio
audio = torch.randn(16000 * 30)  # 30 seconds
logits, embeddings = model(audio)
# logits: (1, 519), embeddings: (1, 768)

# Predict with labels
activations, labels = model.predict_labels(audio)
```

---

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

All models are hosted on [Hugging Face](https://huggingface.co/mtg-upf) and downloaded automatically on first use.

---

## License

This package is licensed under [AGPL-3.0-only](LICENSE), following the original MAEST license.

---

## Credits & Acknowledgments

This package is a repackaging of **MAEST** (Music Audio Efficient Spectrogram Transformer) created by [Pablo Alonso-Jimenez](https://github.com/palonso) and colleagues at the [Music Technology Group (MTG)](https://www.upf.edu/web/mtg), Universitat Pompeu Fabra.

- **Original Repository**: [https://github.com/palonso/maest](https://github.com/palonso/maest)
- **Hugging Face Models**: [https://huggingface.co/mtg-upf](https://huggingface.co/mtg-upf)
- **Paper**: [arXiv:2309.16418](https://doi.org/10.48550/arXiv.2309.16418)

We are grateful to the original authors for making their research and pretrained models publicly available.

---

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

## Related Projects

- [MAEST](https://github.com/palonso/maest) - Original research repository with training code
- [PaSST](https://github.com/kkoutini/PaSST) - Patchout faSt Spectrogram Transformer (base architecture)
- [Essentia](https://essentia.upf.edu/models.html#maest) - MAEST models in Essentia
