"""Mel-spectrogram extraction module for MAEST inference.

Implements MAEST's exact mel front end using torchaudio (Spectrogram + MelScale
+ logC compression + z-norm), matching upstream MAEST's own inference code
(upstream also never uses Essentia at inference -- Essentia was only used to
build the training dataset). See MelSpectrogram's docstring for the measured
deviation against Essentia-derived features.

Reads: torch, torchaudio.transforms; read by maest_infer.model.MAEST
"""

import torch
from torch.nn import Module
from torchaudio.transforms import Spectrogram, MelScale


class MelSpectrogram(Module):
    """Extract mel-spectrograms as a torchaudio module.

    The torchaudio implementation is similar but not identical to Essentia's,
    which upstream MAEST used to build its training dataset (never at
    inference). Empirically verified (2026-07, 4 clips incl. real music,
    discogs-maest-10s-pw-129e): final-embedding cosine similarity >=0.999 and
    100% top-5 label agreement vs Essentia-derived features; residual
    mismatch concentrates in the lowest mel bands and edge frames. Note the
    per-bin rtol/atol=1e-3 claim below holds for only ~74-97% of bins, not
    all of them -- the embedding-level agreement is what actually matters for
    inference, and that holds comfortably. This torchaudio approximation is
    inherited from upstream MAEST's own inference code, not introduced by
    this package.
    """

    sr = 16000
    win_len = 512
    hop_len = 256
    power = 2
    n_mel = 96
    norm = "slaney"
    mel_scale_type = "slaney"
    norm_mean = 2.06755686098554
    norm_std = 1.268292820667291

    def __init__(self):
        super().__init__()

        self.spec = Spectrogram(
            n_fft=self.win_len,
            win_length=self.win_len,
            hop_length=self.hop_len,
            power=self.power,
        )

        self.mel_scale = MelScale(
            n_mels=self.n_mel,
            sample_rate=self.sr,
            n_stft=self.win_len // 2 + 1,
            norm=self.norm,
            mel_scale=self.mel_scale_type,
        )

    def znorm(self, input_values: torch.Tensor) -> torch.Tensor:
        return (input_values - (self.norm_mean)) / (self.norm_std * 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # convert to power spectrogram
        spec = self.spec(waveform)

        # convert to mel-scale
        mel = self.mel_scale(spec)

        # apply logC compression
        logmel = torch.log10(1 + mel * 10000)

        # normalize
        logmel = self.znorm(logmel)

        return logmel
