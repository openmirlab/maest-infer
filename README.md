# maest-infer

Inference-only package for MAEST (Music Audio Efficient Spectrogram Transformer).

## Installation

```bash
uv pip install -e .
```

## Usage

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

- `discogs-maest-5s-pw-129e` - 5 second input, 400 labels
- `discogs-maest-10s-fs-129e` - 10 second input, 400 labels (from scratch)
- `discogs-maest-10s-pw-129e` - 10 second input, 400 labels (PaSST weights)
- `discogs-maest-10s-dw-75e` - 10 second input, 400 labels (DeiT weights)
- `discogs-maest-20s-pw-129e` - 20 second input, 400 labels
- `discogs-maest-30s-pw-129e` - 30 second input, 400 labels
- `discogs-maest-30s-pw-73e-ts` - 30 second input, 400 labels (teacher-student)
- `discogs-maest-30s-pw-129e-519l` - 30 second input, 519 labels

## License

AGPL-3.0-only

## Credits

Based on [MAEST](https://github.com/palonso/maest) by Pablo Alonso.
